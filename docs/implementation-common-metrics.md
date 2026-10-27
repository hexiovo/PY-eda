# Common EDA metrics addition

## Background and context

The user approved adding the common metrics identified in the preceding review. The existing registry has 101 full-capability metrics (90 single-record metrics). Core SCR and event metrics already exist. This change adds amplitude transforms and within-record normalization, sample distribution summaries, and SCR timing dispersion. It does not implement a new driver-producing decomposition algorithm.

## Acceptance criteria

- Preserve the original 33 features and existing extended numerical results.
- Expose every new metric through the shared registry, GUI selection, frozen definitions and exports.
- Keep original amplitudes; transform individual observations before aggregation. Invalid trials remain missing; valid nonresponses remain zero for square-root/log transforms.
- Normalize against the complete current recording for the same subject/session/channel, across conditions, never across people or recordings. Save reference counts, mean, sample SD and maximum with transformed observations. SCR max correction is amplitude/max, not min-max scaling.
- Constant references cannot produce a Z score; all-zero references cannot produce max correction. Missing, censored and insufficient data retain explicit reasons.
- Unit tests, integration tests, full regression suite and package build must pass. Verify GUI catalog coverage and exported tables.

## Implementation details

- Add 34 registry entries: 18 clean/tonic distribution summaries (min, max, P05, P25, P75, P95, adjusted skewness, unbiased excess kurtosis, unscaled MAD); four SCR amplitude summaries (sqrt, ln(1+A/1uS), recording Z, recording maximum fraction); four zero-inclusive event-series equivalents; eight timing summaries (median/SD for rise, configured recovery, 50% and 63% recovery).
- Use existing NumPy/SciPy dependencies. Linear sample percentiles, sample SD (ddof=1), skewness with bias=False (n>=3), Fisher excess kurtosis with bias=False (n>=4), MAD=median(abs(x-median(x))). Constant skew/kurtosis are unavailable.
- Put shared pure mathematics and table annotation in a small metrics module. Integrate with core_features and metrics_trials. Reuse existing families so GUI filtering stays consistent.
- Extend trial_table and scr_events_extended with transformed values and normalization reference metadata; keep raw SCR output intact.
- Update the feature dictionary, method notes and help source. Historical release evidence remains historical; source changes do not silently update existing portable executables.

## Testing strategy

Capture baseline numerical tests and a deterministic extraction fixture. Check analytical examples, zero/no-response behavior, constant references, insufficient samples, invalid windows, right censoring, reference scope across conditions and isolation across records/channels. Check scale/shift invariants on seeded inputs. Compare old outputs exactly. Run end-to-end selected-ID exports and frozen definitions, the full pytest suite, and a wheel build using local dependencies. Record evidence under artifacts/verification/common-metrics.
