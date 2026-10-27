"""Local convex Bateman sparse deconvolution; see docs/driver-method.md.

This is a distinct research model, not a Ledalab/CDA/cvxEDA implementation.
The optimized nonnegative impulse masses a are in uS; driver=a/dt is uS/s.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
from scipy import optimize, signal, sparse

from .config import DEFAULT_CONFIG
from .core_preprocess import _bool_array, _full_config, _sampling_rate
from .errors import fail


def driver_settings(config):
    return {**DEFAULT_CONFIG["decomposition"]["driver"],
            **config.get("decomposition", {}).get("driver", {})}


def bateman_kernel(fs, settings):
    """Causal, continuously peak-normalized kernel with an explicit tail bound."""
    rise, decay = float(settings["tau_rise_s"]), float(settings["tau_decay_s"])
    if not (math.isfinite(fs) and fs > 0 and 0 < rise < decay):
        raise ValueError("bateman_requires_positive_fs_and_rise_less_than_decay")
    peak_t = rise * decay * math.log(decay / rise) / (decay - rise)
    peak = math.exp(-peak_t / decay) - math.exp(-peak_t / rise)
    duration = -decay * math.log(float(settings["kernel_tail_tolerance"]) * peak)
    size = int(math.ceil(duration * fs)) + 1
    if size > int(settings["max_segment_samples"]):
        fail("driver_kernel_budget", "Bateman 核长超过 driver 样本预算。", category="resource", stage="decomposition")
    time = np.arange(size, dtype=float) / fs
    kernel = (np.exp(-time / decay) - np.exp(-time / rise)) / peak
    return kernel, {"peak_time_s": peak_t, "normalization": "continuous_peak_1",
                    "kernel_samples": size, "kernel_support_s": (size - 1) / fs,
                    "tail_amplitude_bound": float(settings["kernel_tail_tolerance"])}


def tonic_basis(n, fs, knot_s):
    """Sparse linear interpolation basis including both physical segment endpoints."""
    duration = (n - 1) / fs
    count = max(2, int(math.ceil(duration / knot_s)) + 1)
    knots = np.linspace(0., duration, count)
    time = np.arange(n) / fs
    left = np.clip(np.searchsorted(knots, time, side="right") - 1, 0, count - 2)
    fraction = (time - knots[left]) / (knots[left + 1] - knots[left])
    rows = np.repeat(np.arange(n), 2)
    columns = np.column_stack((left, left + 1)).ravel()
    data = np.column_stack((1 - fraction, fraction)).ravel()
    return sparse.csr_matrix((data, (rows, columns)), shape=(n, count)), knots


def forward_driver(mass, kernel):
    return signal.fftconvolve(mass, kernel, mode="full")[:len(mass)]


def adjoint_driver(values, kernel):
    return signal.fftconvolve(values, kernel[::-1], mode="full")[len(kernel)-1:len(kernel)-1+len(values)]


def objective_gradient(x, y, kernel, basis, settings):
    n = len(y)
    mass, coefficients = x[:n], x[n:]
    residual = forward_driver(mass, kernel) + basis @ coefficients - y
    curvature = np.diff(coefficients, n=2)
    alpha, ridge, gamma = (float(settings[key]) for key in ("sparsity", "ridge", "tonic_smoothness"))
    value = .5 * np.dot(residual, residual) + alpha * np.sum(mass) + .5 * ridge * np.dot(mass, mass)
    value += .5 * gamma * np.dot(curvature, curvature)
    grad_mass = adjoint_driver(residual, kernel) + alpha + ridge * mass
    grad_tonic = np.asarray(basis.T @ residual).ravel()
    grad_tonic[:-2] += gamma * curvature
    grad_tonic[1:-1] -= 2 * gamma * curvature
    grad_tonic[2:] += gamma * curvature
    return float(value), np.concatenate((grad_mass, grad_tonic))


def solve_segment(y, fs, settings, *, cancel_callback=None):
    y = np.asarray(y, float)
    if y.ndim != 1 or len(y) < 2 or not np.all(np.isfinite(y)):
        raise ValueError("driver_solver_requires_finite_segment")
    if len(y) > int(settings["max_segment_samples"]):
        fail("driver_segment_budget", "连续段超过 driver 样本预算；请明确降采样或调整预算。", category="resource", stage="decomposition")
    kernel, kernel_info = bateman_kernel(fs, settings)
    basis, knots = tonic_basis(len(y), fs, settings["tonic_knot_s"])
    initial = np.concatenate((np.zeros(len(y)), np.full(len(knots), np.median(y))))

    def objective(x):
        if cancel_callback is not None and cancel_callback():
            raise InterruptedError("用户请求取消 driver 求解。")
        return objective_gradient(x, y, kernel, basis, settings)

    result = optimize.minimize(objective, initial, method="L-BFGS-B", jac=True,
                               bounds=[(0., None)] * len(y) + [(None, None)] * len(knots),
                               options={"maxiter": int(settings["max_iterations"]),
                                        "maxfun": int(settings["max_iterations"]) * 5,
                                        "gtol": float(settings["gradient_tolerance"]),
                                        "ftol": float(settings["function_tolerance"]), "maxls": 40})
    objective_value, gradient = objective(result.x)
    mass = result.x[:len(y)]
    projected = gradient.copy()
    active = mass <= 0
    projected[:len(y)][active] = np.minimum(projected[:len(y)][active], 0.)
    projected_norm = float(np.max(np.abs(projected)))
    tolerance = float(settings["kkt_tolerance"]) * max(1., float(np.max(np.abs(y))))
    converged = bool(result.success and np.all(np.isfinite(result.x)) and
                     np.isfinite(objective_value) and projected_norm <= tolerance and np.all(mass >= 0))
    tonic = np.asarray(basis @ result.x[len(y):])
    phasic = forward_driver(mass, kernel)
    residual = y - tonic - phasic
    diagnostics = {**kernel_info, "method": "bateman_sparse", "converged": converged,
                   "status": "succeeded" if converged else "unavailable",
                   "reason": "" if converged else "driver_solver_not_converged",
                   "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
                   "iterations": int(result.nit), "objective": objective_value,
                   "projected_gradient_inf": projected_norm, "projected_gradient_limit": tolerance,
                   "reconstruction_rmse_uS": float(np.sqrt(np.mean(residual ** 2))),
                   "tonic_knot_count": len(knots), "driver_unit": "uS/s", "mass_unit": "uS",
                   "sampling_rate_hz": fs, "parameters": copy.deepcopy(settings)}
    return {"driver_uS_s": mass * fs, "tonic_uS": tonic, "phasic_uS": phasic,
            "residual_uS": residual, "diagnostics": diagnostics}


def separate_driver(processed, config, *, cancel_callback=None):
    full = _full_config(config)
    settings = driver_settings(config)
    time, source = np.asarray(processed["time_s"], float), np.asarray(processed["clean_uS"], float)
    if time.ndim != 1 or source.shape != time.shape or not len(time) or not np.all(np.isfinite(time)):
        fail("processed_shape", "driver 需要等长的一维有限时间轴与 clean 信号。", stage="decomposition")
    quality = copy.deepcopy(processed.get("quality", {}))
    mask = _bool_array(quality.get("processing_eligible", quality.get("eligible")), len(time), "processing_eligible")
    mask &= np.isfinite(source)
    for key in ("interpolated", "artifact", "edge_affected"):
        mask &= ~_bool_array(quality.get(key), len(time), key)
    if not np.any(mask):
        fs = processed.get("sampling_rate_hz") or full["mapping"].get("sampling_rate_hz")
        if not fs:
            fail("sampling_rate_unknown", "driver 需要已知采样率。", stage="decomposition")
        segments, max_gap = [], 1.5 / float(fs)
    else:
        fs, segments, max_gap = _sampling_rate(time, mask, full,
                    metadata=processed.get("timebase_metadata", processed.get("metadata")), require_regular=True)
    fs = float(fs)
    if fs * settings["tau_rise_s"] < 1:
        fail("driver_sampling_resolution", "采样间隔必须不大于 driver 上升时间常数。", category="configuration", stage="decomposition")
    # Conservative O(n) working-set estimate includes optimizer history and FFT workspace.
    estimate = (len(time) * 256 + min(len(time), settings["max_segment_samples"]) * 2048) / 1024**2
    if estimate > float(full["compute"]["memory_limit_mb"]):
        fail("driver_memory_budget", "driver 工作内存估计超过配置上限。", category="resource", stage="decomposition")
    kernel, kernel_info = bateman_kernel(fs, settings)
    radius = len(kernel) - 1
    arrays = {name: np.full(len(time), np.nan) for name in ("driver_uS_s", "tonic_uS", "phasic_uS", "residual_uS")}
    eligible = np.zeros(len(time), bool)
    edge = _bool_array(quality.get("edge_affected"), len(time), "edge_affected").copy()
    diagnostics = []
    for start, stop in segments:
        if cancel_callback is not None and cancel_callback():
            raise InterruptedError("用户请求取消 driver 分解。")
        row = {"start": int(start), "end": int(stop), "start_s": float(time[start]),
               "end_s": float(time[stop-1]), "samples": stop-start}
        if stop-start <= 2*radius+2 or time[stop-1]-time[start] < full["preprocess"]["min_segment_duration_s"]:
            diagnostics.append({**row, "status": "skipped", "reason": "driver_segment_too_short",
                                "required_samples_gt": 2*radius+2})
            continue
        result = solve_segment(source[start:stop], fs, settings, cancel_callback=cancel_callback)
        diagnostics.append({**row, **result["diagnostics"]})
        if not result["diagnostics"]["converged"]:
            continue
        for name in arrays:
            arrays[name][start:stop] = result[name]
        edge[start:start+radius] = True
        edge[stop-radius:stop] = True
        eligible[start+radius:stop-radius] = True
    analysis_input = _bool_array(quality.get("analysis_eligible", quality.get("eligible")), len(time), "analysis_eligible")
    detector_input = _bool_array(quality.get("detector_eligible", quality.get("eligible")), len(time), "detector_eligible")
    quality.update(edge_affected=edge, analysis_eligible=analysis_input & eligible,
                   eligible=analysis_input & eligible, driver_eligible=analysis_input & eligible,
                   detector_eligible=detector_input & eligible, segments=list(segments))
    success = sum(row["status"] == "succeeded" for row in diagnostics)
    report = {"method": "bateman_sparse", "version": "local-convex-bateman-v1",
              "basis": "clean_uS", "driver_unit": "uS/s", "unit": "uS",
              "status": "unavailable" if not success else "succeeded" if success == len(diagnostics) else "partial",
              "parameters": settings, "segments": diagnostics, "sampling_rate_hz": fs,
              "residual_definition": "clean_uS - tonic_uS - phasic_uS",
              "boundary_policy": "exclude_full_kernel_support_at_both_segment_ends",
              "edge_radius_samples": radius, "estimated_working_mb": estimate, **kernel_info}
    bundle = copy.deepcopy(processed)
    bundle.update(arrays)
    bundle.update(driver=arrays["driver_uS_s"], quality=quality, diagnostics=report,
                  method="bateman_sparse", basis="clean_uS", detector_eligible=quality["detector_eligible"],
                  sampling_rate_hz=fs, max_gap_s=max_gap, segments=list(segments))
    return bundle


def driver_diagnostics_table(bundle):
    return pd.DataFrame(bundle.get("diagnostics", {}).get("segments", []))
