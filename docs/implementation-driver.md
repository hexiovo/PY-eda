# Driver features and v0.3.0 delivery plan

## Background and context

The user requested actual driver metrics, a new Windows executable, and a Git push to https://github.com/hexiovo/PY-eda.git. That repository is empty, public, and writable by the connected account. The current filter separation returns no driver. Existing common metrics must remain numerically unchanged under the default filter method.

## Accepted scope and approach

Implement an explicitly named `bateman_sparse` research method with SciPy, not a claim of Ledalab/CDA/cvxEDA compatibility. For each uninterrupted eligible uniform segment, fit `y = H a + B c + residual`, `a >= 0`. H is a causal peak-normalized biexponential Bateman kernel; B is linear interpolation over tonic knots. Minimize `0.5||Ha+Bc-y||² + alpha*sum(a) + 0.5*ridge*||a||² + 0.5*gamma*||D2 c||²`. The driver rate is `a/dt` in uS/s; integrated driver is in uS under this declared kernel normalization. Optimize with bounded L-BFGS-B and an independently checked analytic gradient. No external GPL algorithm code is copied and no new runtime dependency is required.

The kernel constants, tonic knot spacing, penalties, solver budget/tolerances, kernel truncation and burst threshold are explicit configuration. Discard unsuccessful or failed projected-gradient checks. Keep diagnostics. Fit segments independently; gaps, artifacts, interpolation and preprocessing edges are not silently bridged. Mask a full kernel support at both segment boundaries. Enforce sample and compute-memory limits before allocating.

Add nine features: driver integral, time mean, maximum, active-time proportion, complete burst count/rate, and event-response integral/maximum/peak latency. Output processed driver, a burst catalogue and segment diagnostics. Save driver arrays through cache/resume. Enable the metric family only with the driver-producing method, show inactive entries in the GUI, and provide an explicit driver preset and parameter editor.

## Acceptance criteria

- Current filter separation and existing metrics retain their original numerical behavior.
- Independent dense convolution, adjoint/gradient, constrained least-squares oracle, analytical kernel and synthetic recovery tests pass; failed convergence, budgets, short records, irregular grids, gaps and empty support are tested.
- Driver is nonnegative; reconstruction residual is explicit; integrals and durations use declared support; invalid values are missing, never substituted with zero.
- New IDs appear in GUI, frozen definitions, export and resumed output. Driver parameters are saved with runs.
- Full tests and Windows build pass. Frozen/source outputs agree on synthetic fixtures. Package help and dependencies with the EXEs and provide a ZIP.
- Initialize local Git, inspect an explicit source/document/test/build-script inventory for accidental data/credentials/large artifacts, commit and push to the requested empty main branch without force. Exclude local environments, results, caches and generated binaries from source Git. Publish the portable ZIP as a release asset if an authenticated upload capability is available; otherwise provide the local ZIP and report upload status accurately.

## Validation and evidence

Record the 190-test baseline and a deterministic extraction baseline before edits. Use the existing numerical suite plus a new driver test module. Differential oracle: explicit dense H/B and independent scipy.optimize.lsq_linear for alpha=0, and KKT/projected-gradient checks for sparse penalty >0. Seeded domains cover sample rates, kernel constants, nonnegative impulses and tonic offsets. No reference equality to Ledalab is claimed. Save evidence under artifacts/verification/driver-v0.3.0; publish a concise nonlocal validation summary in docs.

## Sources

- Bateman/deconvolution model context: https://pmc.ncbi.nlm.nih.gov/articles/PMC2904901/
- Solver contract: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

The literature motivates the model family; this repository's stated objective, normalizations and tests define this implementation.
