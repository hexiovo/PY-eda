"""Independent operators/oracles, boundaries and delivery integration for driver."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from eda_toolkit.config import get_preset, resolve_config
from eda_toolkit.core_decompose import separate_tonic_phasic
from eda_toolkit.core_features import extract_window_features, list_feature_definitions
from eda_toolkit.driver_decompose import (adjoint_driver, bateman_kernel, forward_driver,
    objective_gradient, solve_segment, tonic_basis)
from eda_toolkit.metrics_driver import DRIVER_IDS, driver_bursts, driver_window_metrics, linear_window


def settings(**changes):
    return {**get_preset("single_driver")["decomposition"]["driver"], **changes}


def dense_kernel(n, kernel):
    matrix = np.zeros((n, n))
    for row in range(n):
        for col in range(row+1):
            if row-col < len(kernel):
                matrix[row, col] = kernel[row-col]
    return matrix


def processed(y, fs=4.):
    y = np.asarray(y, float)
    mask = np.ones(len(y), bool)
    return {"time_s": np.arange(len(y))/fs, "clean_uS": y,
            "sampling_rate_hz": fs,
            "metadata": {"recording_id": "r", "subject_id": "s", "session_id": "v", "channel": "eda", "fs_hz": fs},
            "quality": {key: mask.copy() for key in ("processing_eligible", "analysis_eligible", "detector_eligible", "eligible")}}


def window(start, end, kind="fixed", onset=None):
    return {"recording_id": "r", "window_id": "w", "start_s": start, "end_s": end,
            "kind": kind, "event_id": "e" if kind == "event_response" else None,
            "event_onset_s": onset, "condition": "A"}


def hand_bundle(values):
    result = processed(values, fs=1.)
    result.update(method="bateman_sparse", driver_uS_s=np.array(values, float), max_gap_s=1.5)
    result["quality"]["driver_eligible"] = np.ones(len(values), bool)
    return result


def test_kernel_analytic_shape_normalization_and_tail_bound():
    s = settings(tau_rise_s=.5, tau_decay_s=2.)
    h, info = bateman_kernel(8., s)
    t = np.arange(len(h))/8
    t_peak = .5*2*np.log(4)/1.5
    normalization = np.exp(-t_peak/2)-np.exp(-t_peak/.5)
    np.testing.assert_allclose(h, (np.exp(-t/2)-np.exp(-t/.5))/normalization, rtol=2e-15, atol=1e-15)
    assert h[0] == 0 and np.min(h) >= 0 and np.max(h) <= 1.
    assert h[-1] <= s["kernel_tail_tolerance"]
    assert info["peak_time_s"] == pytest.approx(t_peak)


@pytest.mark.parametrize("seed", [17, 1803, 271026])
def test_generated_convolution_adjoint_and_gradient_against_dense_oracle(seed):
    rng = np.random.default_rng(seed)
    s = settings(tau_rise_s=.3, tau_decay_s=1.3, tonic_knot_s=3.)
    n, fs = 55, 4.
    h, _ = bateman_kernel(fs, s)
    H = dense_kernel(n, h)
    a, b = rng.uniform(0, .1, n), rng.normal(size=n)
    np.testing.assert_allclose(forward_driver(a,h), H@a, atol=2e-15)
    np.testing.assert_allclose(adjoint_driver(b,h), H.T@b, atol=1e-14)
    assert np.dot(forward_driver(a,h), b) == pytest.approx(np.dot(a, adjoint_driver(b,h)), abs=2e-14)
    B, knots = tonic_basis(n, fs, s["tonic_knot_s"])
    # Independently interpolate each unit knot vector, not the sparse indexing algorithm.
    dense_B = np.column_stack([np.interp(np.arange(n)/fs, knots, np.eye(len(knots))[j]) for j in range(len(knots))])
    np.testing.assert_allclose(B.toarray(), dense_B, atol=1e-15)
    x = np.r_[a, rng.normal(2,.1,len(knots))]
    y = H@a + dense_B@x[n:] + rng.normal(0,.02,n)
    fun = lambda z: objective_gradient(z,y,h,B,s)[0]
    numeric = optimize._numdiff.approx_derivative(fun,x,method="3-point").ravel()
    actual = objective_gradient(x,y,h,B,s)[1]
    np.testing.assert_allclose(actual,numeric,rtol=2e-6,atol=2e-8)


def test_convex_solution_against_independent_bounded_least_squares_oracle():
    s = settings(sparsity=0., ridge=.02, tonic_smoothness=.5, tonic_knot_s=8.,
                 function_tolerance=1e-15, gradient_tolerance=1e-10)
    n, fs = 100, 4.
    h, _ = bateman_kernel(fs,s)
    H = dense_kernel(n,h)
    knots = np.linspace(0,(n-1)/fs,int(np.ceil((n-1)/fs/8.))+1)
    B = np.column_stack([np.interp(np.arange(n)/fs,knots,np.eye(len(knots))[j]) for j in range(len(knots))])
    mass = np.zeros(n);mass[[20,60]]=[.2,.4]
    y = H@mass + B@np.linspace(2,2.1,len(knots))
    k = len(knots)
    D = np.diff(np.eye(k),n=2,axis=0)
    A = np.block([[H,B], [np.sqrt(s['ridge'])*np.eye(n),np.zeros((n,k))],
                  [np.zeros((k-2,n)),np.sqrt(s['tonic_smoothness'])*D]])
    target = np.r_[y,np.zeros(n+k-2)]
    oracle=optimize.lsq_linear(A,target,bounds=(np.r_[np.zeros(n),np.full(k,-np.inf)],np.full(n+k,np.inf)),tol=1e-12,max_iter=2000)
    assert oracle.success
    result=solve_segment(y,fs,s)
    assert result['diagnostics']['converged']
    np.testing.assert_allclose(result['driver_uS_s']/fs,oracle.x[:n],atol=3e-5,rtol=1e-3)
    np.testing.assert_allclose(result['tonic_uS'],B@oracle.x[n:],atol=2e-6,rtol=1e-6)
    assert result['diagnostics']['objective']==pytest.approx(.5*np.sum((A@oracle.x-target)**2),abs=1e-9)


def test_known_impulses_offset_invariance_and_reconstruction():
    s=settings(); fs=4.; n=481
    h,_=bateman_kernel(fs,s)
    mass=np.zeros(n);mass[[100,220,300]]=[.3,.5,.4]
    # np.convolve is an independent direct convolution of the declared synthetic model.
    y=2.+np.arange(n)*.0001+np.convolve(mass,h)[:n]
    result=solve_segment(y,fs,s)
    shifted=solve_segment(y+5,fs,s)
    assert result['diagnostics']['converged'] and shifted['diagnostics']['converged']
    assert np.min(result['driver_uS_s'])>=0
    assert result['driver_uS_s'].sum()/fs==pytest.approx(1.2,abs=.01)  # intentional L1 shrinkage
    np.testing.assert_allclose(shifted['driver_uS_s'],result['driver_uS_s'],atol=2e-4)
    np.testing.assert_allclose(result['tonic_uS']+result['phasic_uS']+result['residual_uS'],y,atol=1e-15)
    assert result['diagnostics']['reconstruction_rmse_uS']<.001


def test_failed_solver_short_segments_and_budgets_never_become_zero_driver():
    config=get_preset('single_driver')
    config['decomposition']['driver']['max_iterations']=1
    data=processed(2+np.sin(np.arange(481)/20)*.2)
    result=separate_tonic_phasic(data,config)
    assert result['diagnostics']['status']=='unavailable'
    assert np.isnan(result['driver_uS_s']).all()
    assert not result['quality']['driver_eligible'].any()
    short=separate_tonic_phasic(processed(np.ones(20)),get_preset('single_driver'))
    assert short['diagnostics']['segments'][0]['reason']=='driver_segment_too_short'
    config=get_preset('single_driver');config['decomposition']['driver']['max_segment_samples']=100
    with pytest.raises(ValueError,match='预算'):
        separate_tonic_phasic(data,config)
    config=get_preset('single_driver');config['compute']['memory_limit_mb']=.001
    with pytest.raises(ValueError,match='内存'):
        separate_tonic_phasic(data,config)


def test_segments_are_independent_and_masks_cache_and_cancel_are_preserved(tmp_path):
    config=get_preset('single_driver')
    s=config['decomposition']['driver'];h,_=bateman_kernel(4.,s)
    mass=np.zeros(300);mass[120]=.4
    segment=2+np.convolve(mass,h)[:300]
    data=processed(np.r_[segment,segment])
    data['time_s'][300:]+=10
    saved=copy.deepcopy(data)
    first=separate_tonic_phasic(data,config)
    data['clean_uS'][300:]+=10
    second=separate_tonic_phasic(data,config)
    np.testing.assert_array_equal(first['driver_uS_s'][:300],second['driver_uS_s'][:300])
    assert first['diagnostics']['status']=='succeeded'
    assert not first['quality']['driver_eligible'][:64].any()
    assert not first['quality']['driver_eligible'][236:364].any()
    np.testing.assert_array_equal(saved['quality']['eligible'],data['quality']['eligible'])
    from eda_toolkit.pipeline import _save_cache,_read_cache
    _save_cache({'bundle':first,'tables':{},'summary':{'status':'succeeded'}},tmp_path)
    restored=_read_cache(tmp_path)['bundle']
    np.testing.assert_array_equal(restored['driver_uS_s'],first['driver_uS_s'])
    np.testing.assert_array_equal(restored['quality']['driver_eligible'],first['quality']['driver_eligible'])
    with pytest.raises(InterruptedError):
        separate_tonic_phasic(data,config,cancel_callback=lambda:True)


def test_linear_support_integral_threshold_crossings_and_bursts():
    data=hand_bundle([0,1,0,0,2,0,0])
    config=get_preset('single_driver');config['decomposition']['driver']['burst_threshold_uS_s']=.5
    bursts=driver_bursts(data,config)
    assert bursts['status'].tolist()==['valid','valid']
    np.testing.assert_allclose(bursts.start_s,[.5,3.25])
    np.testing.assert_allclose(bursts.end_s,[1.5,4.75])
    rows=driver_window_metrics(data,window(0,6,'event_response',0),config,bursts)
    expected={'driver_integral_uS':3.,'driver_time_mean_uS_s':.5,'driver_max_uS_s':2.,
              'driver_active_time_proportion':2.5/6,'driver_burst_count':2.,'driver_burst_rate_per_min':20.,
              'event_driver_integral_uS':3.,'event_driver_max_uS_s':2.,'event_driver_peak_latency_s':4.}
    for fid,value in expected.items():
        assert rows[fid]['status']=='valid'
        assert rows[fid]['value']==pytest.approx(value,abs=1e-14)
    clipped=linear_window(data['time_s'],data['driver_uS_s'],np.ones(7,bool),.5,1.5,1.5,.5)
    assert clipped=={'integral':.75,'duration':1.,'active':1.}
    data['quality']['driver_eligible'][3]=False
    bursts=driver_bursts(data,config)
    assert 'censored' in set(bursts.status)
    config['windows']['min_coverage']=.99
    rows=driver_window_metrics(data,window(0,6),config,bursts)
    assert all(row['value'] is None for row in rows.values())


def test_zero_driver_missing_driver_and_event_latency():
    config=get_preset('single_driver');data=hand_bundle([0]*10)
    rows=driver_window_metrics(data,window(0,9,'event_response',0),config,driver_bursts(data,config))
    assert rows['driver_integral_uS']['value']==0.
    assert rows['driver_burst_count']['value']==0
    assert rows['event_driver_peak_latency_s']['status']=='unavailable'
    data['method']='filter_separation'
    assert all(r['reason']=='requires_bateman_sparse' for r in driver_window_metrics(data,window(0,9),config,pd.DataFrame()).values())


@pytest.mark.parametrize('changes',[{'tau_rise_s':2.}, {'tonic_knot_s':1.}, {'ridge':0.},
    {'sparsity':-1.}, {'kernel_tail_tolerance':.9}, {'max_iterations':True}, {'burst_threshold_uS_s':0.}])
def test_invalid_driver_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        resolve_config({'decomposition':{'driver':changes}})


def test_feature_selection_frozen_parameters_and_signal_exports(tmp_path):
    config=get_preset('single_driver')
    assert len(list_feature_definitions(config))==133
    assert not DRIVER_IDS.intersection(list_feature_definitions(get_preset()))
    config['synchrony']['enabled']=True
    assert len(list_feature_definitions(config))==144
    config['synchrony']['enabled']=False
    config['features']['ids']=sorted(DRIVER_IDS)
    from eda_toolkit.contracts import make_record
    from eda_toolkit.pipeline import _process_one
    s=config['decomposition']['driver'];h,_=bateman_kernel(4.,s)
    mass=np.zeros(481);mass[[120,260]]=[.3,.4]
    y=2+np.arange(481)*.0001+np.convolve(mass,h)[:481]
    record=make_record(np.arange(481)/4,y,{'recording_id':'r','subject_id':'s','session_id':'v','channel':'eda','unit':'uS','fs_hz':4.,'clock_domain':'recording_local'})
    result=_process_one(record,{},config,lambda:False,lambda *_:None)
    assert 'driver_uS_s' in result['tables']['signals']
    assert {'driver_bursts','driver_diagnostics'}<=result['tables'].keys()
    features=result['tables']['features']
    assert set(features.feature_id)<=DRIVER_IDS
    assert features.loc[features.feature_id.eq('driver_integral_uS'),'status'].eq('valid').any()
    config['output']['formats']=['csv','json','xlsx']
    from eda_toolkit.core_export import export_run,verify_manifest
    export_run({'summary':{'status':'succeeded'},'tables':result['tables']},{'resolved_config':config},tmp_path,config)
    assert verify_manifest(tmp_path)['valid']


def test_gui_driver_enablement_parameters_and_switch_guard(tmp_path,monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QMessageBox
    from PySide6.QtCore import Qt
    from eda_toolkit.ui.app import create_window
    cfg=get_preset();cfg['workspace']['project_directory']=str(tmp_path/'project')
    warnings=[]
    monkeypatch.setattr(QMessageBox,'warning',lambda *args:warnings.append(str(args[-1])))
    view,state=create_window(cfg)
    try:
        w=state['widgets'];table=w['feature_registry_table'];method=w['decomp_method']
        row=next(i for i in range(table.rowCount()) if table.item(i,2).text()=='driver_integral_uS')
        assert not table.item(row,0).flags() & Qt.ItemFlag.ItemIsUserCheckable
        method.setCurrentIndex(method.findData('bateman_sparse'))
        assert table.item(row,0).flags() & Qt.ItemFlag.ItemIsUserCheckable
        table.setCurrentCell(row,0);w['select_current_feature_only'].click()
        assert state['config']['features']['ids']==['driver_integral_uS']
        method.setCurrentIndex(method.findData('filter_separation'))
        assert method.currentData()=='bateman_sparse' and warnings
        parameters=json.loads(w['driver_parameters'].toPlainText());parameters['sparsity']=.02
        w['driver_parameters'].setPlainText(json.dumps(parameters));w['driver_apply'].click()
        assert state['config']['decomposition']['driver']['sparsity']==.02
        assert state['feature_registry']['driver_integral_uS']['parameters']['sparsity']==.02
    finally:
        view.close();state['app'].processEvents()
