# Focused repair plan — Sol approved 2026-10-25

This addendum implements the independent review counterexamples without changing
the release scope. Owners retain their existing files. The coordinator owns
schemas/controller, the synchrony owner owns core_sync, and the numerical owner
owns SCR feature changes. Ordinary recommended execution applies.

## Surrogate analysis support

Add synchrony.analysis_intervals_s=null, analysis_context_s=0.0 and
pseudodyad_exchangeability=false. group_sync research preset uses 60 seconds of
context at both ends. Explicit disjoint half-open reference-second intervals take
precedence. Otherwise use the aligned common grid minus the declared context.
Retain all context arrays; generate observed/null windows only inside the resolved
analysis intervals. Too-short intervals are unavailable, never automatically shrunk.
Publish resolved intervals/context and the support policy.

Construct each participant's required sample-index union for all analysis windows,
lags and seam-exclusion width. Admissible shifts have seams outside required support
and identical moved versus observed masks at every required index. Matching counts
alone is insufficient. A participant uses one shift for all channels/windows/metrics.
Zero absolute shift is allowed. Tested pair endpoints must meet the declared minimum
circular relative shift. Uniform participant candidate proposals with joint rejection
define the conditional sampler; never select shifts from correlations.

Preflight empty candidates (shift_support_infeasible), provably impossible joint
constraints (joint_shift_infeasible), and full-coverage seam impossibility before
correlation attempts. Enumerate small candidate spaces; use bounded geometry-only
search for large spaces and distinguish feasibility_unresolved from impossible.
Resource limits/cancellation remain. Insufficient accepted B yields no tail estimate;
never silently reduce requested B. Record candidate/world/proposal counts, with-
replacement sampling and support policy version.

## Search tails and methods

two-sided: max(abs(r)) across lags, surrogate >= observed.
greater: max(r), surrogate >= observed.
less: min(r), surrogate <= observed.

Observed and surrogate repeat the same search, preserve selected signed r/lag and
deterministic ties. Generic fields are observed_statistic/null_mean_statistic/
null_sd_statistic/statistic/tail. Populate max_abs fields only for two-sided analyses.
The result remains surrogate_tail_probability/experimental_uncalibrated, with Monte
Carlo uncertainty; add-one arithmetic does not establish exchangeability/calibration.

none means not requested; circular_shift follows the support procedure; pseudodyad
must perform actual donor reassignment. Selecting one never runs another method.

Supported pseudodyads use complete matched independent blocks (group_id,session_id).
Require the explicit exchangeability assertion, task and stimulus timeline/strata,
unique role/member_slot and compatible complete slot sets. Task-relative time or
declared task_time_origin_s is required; global clock labels do not imply task phase.
For each stratum/slot independently permute donor blocks in a surrogate world, using
one donor assignment across all windows/lags/metrics/edges. Reject self-pairs, original
partners, same donor interaction block, support mismatch or changed target design.
Preserve donor subject/block/record provenance while computing target identities.
Insufficient metadata/blocks/support/assignments returns precise unavailable, never
circular fallback. Track unique worlds and finite support separately from with-
replacement Monte Carlo resolution. Independent groups may share a session label.

## SCR rate exposure

scr.onset_left_context_s=0.0; onset_right_context_s=null resolves to max_rise_s.
An explicit right context cannot be shorter than max_rise_s. These are transparent
research definitions with task/device qualification pending.

detection_search_eligible preserves processing-qualified support for trough/peak/
recovery searches. onset_eligible marks times with complete continuous left/right
search context. Do not shorten search arrays to calculate onset exposure. Keep all
SCR catalogue rows, onset_eligible and qualification_reason. Primary counts/rates/
amplitude summaries use qualified onsets; other candidates remain for review.
Recovery censoring affects recovery metrics, not qualified amplitude or count.

Rate exposure sums adjacent qualified onset-support intervals clipped to each
analysis window, with no gap bridging or final-sample extrapolation. Event zero
response is available only when the configured onset-search interval has its
required onset support; otherwise the event response is unavailable.

## Focused verification

Reuse existing independent tests. Full coverage must preflight infeasible without
thousands of attempts. Buffered continuous data must accept requested B with exact
support. Impossible joint shifts remain unavailable. Negative coupling must give
correct distinct signed/two-sided searches. Pseudodyad dispatch must not call circular
shift; test missing metadata and a complete matched synthetic donor-world design.
For support 0–40 seconds and R=10 seconds, onset exposure is 30 seconds. Onset 29 /
peak 35 remains qualified; onset 35 / peak 38 remains catalogue-only. Censored
recovery does not erase qualified amplitude/count. No unrelated full revalidation.
