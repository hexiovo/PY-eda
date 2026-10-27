# Execution Plan Addendum — Continuous P00–P14 Development

Date: 2026-10-25. Planner: GPT-6.1 Sol/high. Project: D:\program\EDA.

## Background and authority

Reuse the existing planning revision 1.1, R01–R26, RV01–RV15 and A00–A14.
The human authorized continuous development through every phase, then explicitly
returned planning to GPT-6.1 Sol. This overrides historical planning-only and
per-phase-stop wording for this run without changing those historical files.
HexiPlan uses recommended execution, a large-project cadence, Luna implementation,
independent Sol integration inspection and a final Astra coverage audit. Routine
progression needs no additional human reply. Missing representative real data
limits scientific qualification; it does not stop software engineering.

## Scope and acceptance

Deliver nine CSV templates and dictionaries, general mapping, reversible QC,
segmented preprocessing, independent filter separation/SCR, four window families,
single-person features, alignment/synchrony/surrogates/pseudodyads, structured
statistics including random-intercept mixed models, plots/exports, five presets,
CLI and Chinese GUI sharing the same controller, bounded CPU execution, Chinese
help/PDF/current screenshots and portable Windows GUI/CLI executables.
Conditional cvxEDA/Ledalab/GPU remain unavailable until their own licensing,
runtime, numerical and performance gates pass. No enabled placeholder is a pass.

## Ownership and dependency order

The coordinator owns contracts/config/errors/capabilities and controller
integration. Bootstrap owns packaging metadata, environment and entry routing.
Intake owns baseline evidence and HexiPlan RUN. Implementation owners separately
own import/QC/templates and preprocessing/decomposition/features. Later work
uses nonconflicting owners for synchrony/statistics, export/plots, GUI and CLI.
Schema/method changes return to the planner. No concurrent same-file edits.

P00 creates atomic requirements, method evidence, license and acceptance registers.
Prior citations remain prior assertions until checked for the actual selected
implementation. P01 supplies contracts, isolated dependency resolution and an
early Qt/native frozen probe. Software preparation is not misreported as P00.
P02–P06 build a real headless import→QC→preprocess→decompose→SCR→features path.
P07–P08 add independent-anchor alignment, group metrics/nulls and design-aware
statistics. P09–P10 complete lifecycle/CLI/GUI. P11–P14 measure resources,
document, freeze and consolidate engineering/scientific evidence.

## Implementation decisions

Prefer observed Python 3.14.8 x64 in .venv. Essential-wheel failure permits an
evidenced Python 3.12 fallback. Runtime dependencies: NumPy, SciPy, pandas,
PySide6, Matplotlib, statsmodels, openpyxl and reportlab; pytest for development,
PyInstaller for builds. No NeuroKit2/CVXPY/CVXOPT/torch/CuPy in the base route.
Core modules use ordinary dictionaries/arrays/tables and have no Qt imports.
Numerical functions do not write files. Algorithm docstrings are Chinese and
state units, parameters, failure conditions, missingness and original sources.

Read/hash/parse the same input bytes, preserve leading-zero identifiers and
uninterpreted columns, reject invalid time without sorting/deleting it. Require
declared units/rate when absent; positive declared resistance converts via
1e6/R_ohm. Keep missing amplitude. Acquisition gaps split support.

QC facts overlap. Manual exclusion wins; restore only overrides automatic
candidates. Append-only revisions retain operator/reason/time/parent. Restore
never turns missing or untrusted-time samples into observations.

Resolve clock maps before time-dependent processing. Interpolate only bounded,
flanked short gaps, no extrapolation. Smoothing and Butterworth SOS zero-phase
filters operate within continuous support. Validate Nyquist, pad length and usable
segment duration. Derive/report a conservative edge radius from numerical settling
and tolerance; it is not a scientific duration guarantee. Resample with anti-aliasing
and conservative support propagation. Physical and transformed branches stay distinct.

Base decomposition is independently named SciPy low/high-pass filter separation,
research-example cutoff 0.05 Hz/order 2. Always compute residual=input−tonic−phasic;
driver is absent. No false NeuroKit compatibility claim.

SCR uses per-segment trough-to-peak candidates, first plateau maximum, recording-wide
relative reference before windowing, explicit rise/interval rules and deterministic
amplitude/earlier-peak tie breaking. Recovery is L+(1−f)A at the first descending
crossing, interpolated in time. Subsequent response/gap/boundary censors recovery.
Research examples (0.05 uS, 1 s spacing, 0–10 s rise, half recovery) are configurable.

Windows use [start,end). Exposure sums clipped adjacent eligible intervals without
gap bridges/tail extrapolation. SCR rate=60×onset count/detector-eligible seconds.
Sample summaries and time-weighted integrals have different named definitions.
No response with valid exposure gives count zero and missing detected-amplitude
mean. Event baseline/onset/peak/recovery windows are independently configurable;
bad required support invalidates the event and competing attribution is ambiguous.

Alignment fits t_ref=a+b*t_local, b>0 and fs_ref=fs_local/b using independent anchors.
One anchor supports offset only; two do not establish empirical residual precision.
Report uncertainty, span and extrapolation. Never infer clocks by EDA correlation.
Synchrony preserves signed Pearson/Spearman, lag zero, lag surfaces, valid support
and fixed membership. Positive lag means A precedes B for corr(A(t),B(t+lag)).
Coherent participant-level circular shifts move masks, avoid seams and constrain
relative shifts; each surrogate repeats the observed search. Cap attempts and never
stop by significance. Tail probabilities/Monte Carlo uncertainty are exploratory
until calibrated. Matched pseudodyads require exchangeable task/timeline/role metadata.

Statistics supports descriptive, one-sample, paired, Welch, independent-unit
correlation and random-intercept LMM. Safe structured terms replace formulas.
Validate hierarchy/aggregation/covariates and planned correction families. Report
estimate/effect/SE/CI/statistic/df/p/adjusted p/counts/exclusions/diagnostics.
Failed models are not silently simplified and failed tests do not shrink families.

The controller reserves unique directories, snapshots inputs/config/annotations,
commits artifacts atomically with checksums, supports cancellation and validates
completed stages on resume. Changed inputs cannot resume silently. Required failed
group members block dependent inference. New reduced cohorts require a new design.
CSV/JSON/HTML/PNG/XLSX/PDF exports preserve unavailable as null/status/reason.
Exit codes: 0 success, 2 input/config, 3 capability, 4 runtime, 5 partial, 130 cancelled.

GUI pages cover project/import/QC/preprocess/analysis/synchrony/statistics/results.
All nine templates have view/download/import actions. Real mapping/QC/config/run
functions support plan preview, one-file/batch/retry, logs/progress/cancel and help.
Only the Qt main thread touches widgets. Active snapshots and future drafts differ.

Measure serial before parallel; stable seeds depend on record/stage/replicate,
not completion order. Enforce memory/pairs/windows/null limits without reducing
requested sampling or repetitions. Bundle help/templates/presets/fonts and Qt/native
dependencies. Freeze both one-folder executables; record actual lock/SBOM/notices.
Clean-machine qualification requires a real clean Windows environment, not this
development machine. Real device/task scientific validation remains pending.

## Integration verification

1. P00/P01: complete traceability, strict config, Qt-free core import, wheel resolution,
   early freeze feasibility.
2. P02–P06: Chinese/BOM/mapping/001/NA/units, malformed clocks, QC precedence/history,
   short/long gaps, Nyquist, no peaks/censoring, hand exposure/count/integrals,
   unchanged originals through a shared integration fixture.
3. P07/P08: known affine clocks/coupling/lag/constant inputs, infeasible shifts,
   incomplete-group refusal, pseudodyad metadata, reference contrasts/LMM and repeated
   null simulations with uncertainty. Uncalibrated regimes remain experimental.
4. P09/P10: shared GUI/CLI numerical outputs, cancel/resume/changed-input refusal,
   partial exit codes and real enabled actions.
5. P11–P14: measured serial/parallel agreement, real help/PDF/screenshots, frozen
   end-to-end commands, honest clean-system/science limitations and independent review.

Reuse valid evidence. Necessary failure-directed checks are appropriate; repeating
passed unrelated checks is not. Finish implementation defects within scope before
handoff. The final audit is coverage evidence, not a substitute for executed tests.
