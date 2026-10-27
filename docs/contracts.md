# Data and function contracts

Planner: GPT-6.1 Sol/high. Schema and preset version: 1.0.

The core uses dictionaries, NumPy arrays and pandas tables. It never imports Qt,
reads widget state or modifies source recordings. All analysis times are seconds;
physical conductance is microSiemens (`uS`). Unknown device units require an
explicit declaration. Only the standard `eda_uS` column declares its own unit.

`record` contains `metadata`, `time_s`, `eda_uS`, `sample_index`, and `extras`.
The arrays have equal length. Time is finite and strictly increasing. Missing
amplitude remains NaN and is never replaced by zero. Metadata contains identity,
source/hash/unit, time origin/clock, rate and mapping revision. A declared rate
and a rate inferred from timestamps are distinct provenance fields.

`qc` contains Boolean `facts`, `eligible`, half-open index `segments`, `intervals`,
`summary` and `annotation_revision`. Facts preserve original_missing, nonfinite,
auto_candidate, manual_exclude, manual_restore, interpolated, edge_affected and
time_untrusted independently. Restoring a candidate cannot restore an original
missing observation. Manual exclusions always win.

`processed` retains identity/raw data and adds clean_uS, quality and operations.
`bundle` adds tonic_uS, phasic_uS, residual_uS, driver (None for filter separation),
method/version, basis/units and detector eligibility. Transform branches must not
relabel a z-score or logarithm as physical microSiemens.

Exposure is the sum of eligible adjacent time intervals, clipped to each window.
No gap is bridged and the last point is not extrapolated. Point counts and seconds
are separate. Intervals/windows use `[start,end)`; onset determines continuous-window
SCR assignment. Empty successful tables retain their column definitions.

Public functions are specified in `execution_plan.md` and the module docstrings.
The single controller is `run_pipeline(config,input_manifest=None,
progress_callback=None,cancel_callback=None)`. Both frontends call this entry.

Configuration sections are input, mapping, qc, preprocess, decomposition, scr,
windows, features, synchrony, statistics, compute and output. JSON duplicate keys,
nonstandard numeric literals and unknown fields are errors. The only unconstrained
field-name maps are declared mapping columns/constants. Statistical designs use
structured terms and never executable formulas. Canonical resolved JSON is hashed.

Run statuses: planned, running, succeeded, partial, failed, cancelled, skipped,
not_applicable. Measurement validity: valid, unavailable, censored, ambiguous.
Each unavailable result has a reason. A partial batch is not a successful complete
cohort. A cancelled run keeps completed artifacts without turning into success.

Errors expose category, code, stage, record_id, field, message and remediation.
Unavailable optional algorithms are registered capabilities; selecting one fails
explicitly. All default thresholds are research examples. Engineering/reference
checks do not establish scientific validity for an untested device or task.
