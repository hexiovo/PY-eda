"""预先固定的独立AR(1)重复零模型诊断；不作为真实EDA/任意任务资格。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import math
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from eda_toolkit.config import resolve_config
from eda_toolkit.core_export import atomic_json, atomic_csv
from eda_toolkit.core_sync import estimate_sync_null
from eda_toolkit.capabilities import source_fingerprint, get_capabilities

TRIALS, B, SEED, ALPHA = 30, 39, 20261025, .05
RHO, FS, N, BURN_IN = .8, 4., 321, 1000


def wilson(k, n):
    if not n:
        return [None, None]
    z, p = 1.959963984540054, k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def main():
    root = Path(__file__).resolve().parents[1] / 'artifacts' / 'verification' / 'surrogate_diagnostic'
    root.mkdir(parents=True, exist_ok=True)
    config = resolve_config({'pipeline_id':'group_sync', 'synchrony':{
        'analysis_context_s':12., 'window_s':20., 'step_s':20., 'max_lag_s':2., 'lag_step_s':1.,
        'min_coverage':.95, 'null_replicates':B, 'max_attempts':1000, 'min_shift_s':3.,
        'shared_clock_verified':True}})
    plan = {'planned_independent_trials':TRIALS, 'replicates_per_trial':B, 'base_seed':SEED,
            'rho':RHO, 'sampling_rate_hz':FS, 'samples':N, 'burn_in_samples':BURN_IN,
            'low_tail_cutoff':ALPHA, 'signal_provenance':'independent stationary Gaussian AR1 simulations; not physiological recordings',
            'plan_revision':2, 'revision_reason':'Initial coverage=1 admitted no observations: 20-second half-open 4Hz point windows have 19.75 seconds exposure (79/80). Use predeclared 0.95 eligibility; exact observed/null sample-mask compatibility remains enforced. Initial unavailable-only run retained, no tail results guided this change.',
            'scope':'bounded exploratory surrogate-tail diagnostic; not universal Type-I error control or scientific qualification',
            'source_fingerprint':source_fingerprint(), 'resolved_config':config}
    atomic_json(root/'predeclared_plan.json',plan)
    rows, failures = [], []
    for trial in range(TRIALS):
        generators = [np.random.default_rng(s) for s in np.random.SeedSequence([SEED,trial]).spawn(3)]
        signals = {f'r{index}::eda':lfilter([1.],[1.,-RHO],rng.normal(size=N+BURN_IN))[-N:]
                   for index,rng in enumerate(generators[:2])}
        metadata = {f'r{index}::eda':{'recording_id':f'r{index}','subject_id':f's{index}','session_id':'S',
                    'channel':'eda','clock_domain':'reference','fs_hz':FS} for index in range(2)}
        membership = pd.DataFrame([{'group_id':'G','session_id':'S','subject_id':f's{index}',
                                   'recording_id':f'r{index}','channel':'eda'} for index in range(2)])
        aligned = {'time_s':np.arange(N)/FS, 'signals':signals,
                   'eligible':{key:np.ones(N,dtype=bool) for key in signals}, 'metadata':metadata,
                   'membership':membership,'basis':'phasic','unit':'arbitrary_simulated',
                   'alignment_report':{'formal_synchrony_allowed':True}}
        try:
            summary, tails = estimate_sync_null(aligned,None,config,int(generators[2].integers(0,2**31)))
            if tails.empty or not tails['accepted_replicates'].eq(B).all() or tails['surrogate_tail_probability'].isna().any():
                failures.append({'trial':trial,'reason':'requested_worlds_or_tail_unavailable','summary':summary.to_dict('records')})
            for row in tails.to_dict('records'):
                rows.append({'trial':trial, **row})
        except Exception as error:
            failures.append({'trial':trial,'reason':str(error)})
        if (trial+1)%5==0:
            print(f'completed {trial+1}/{TRIALS}',flush=True)
    table = pd.DataFrame(rows)
    estimates = []
    if not table.empty:
        for window, group in table.groupby('window_id'):
            valid = group.loc[group.surrogate_tail_probability.notna()]
            k = int(valid.surrogate_tail_probability.le(ALPHA).sum())
            estimates.append({'window_id':window,'completed_trials':len(valid),'planned_trials':TRIALS,
                              'tails_at_or_below_cutoff':k,'empirical_low_tail_frequency':k/len(valid) if len(valid) else None,
                              'binomial_wilson_95_interval':wilson(k,len(valid))})
    atomic_csv(root/'trial_tails.csv',table)
    atomic_json(root/'diagnostic.json',{'plan':plan,'environment':get_capabilities(),'window_diagnostics':estimates,
                                     'failed_trials':failures,'method_readiness':'experimental_uncalibrated',
                                     'limitation':'30 independent toy trials give wide uncertainty; no real device/task validation or general calibration claim'})
    print({'trials':TRIALS,'failed_trials':len(failures),'window_diagnostics':estimates})
    return 0 if not failures else 1


if __name__=='__main__':
    raise SystemExit(main())
