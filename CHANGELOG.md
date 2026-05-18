# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - 2026-05-17

### Fixed

- **Duplicate `antenna_id` silent overwrite in legacy CSV translator** (`correlator/mapping.py`). When a legacy `bf_weights_generator` CSV contained duplicate `ant64` values, `_translate_bf_weights_legacy_csv` would silently keep whichever row pandas happened to produce last. It now raises `ValueError` listing the offending `antenna_id` values.

- **Bool-token parser silently failing closed** (`correlator/mapping.py`). `_parse_bool` previously coerced unrecognized strings (e.g., `"1"`, `"yes"`, `"y"`) to `False` via a fallthrough path. It now accepts the full set `{true, 1, yes, y, t}` as True and `{false, 0, no, n, f, ""}` as False, and raises `ValueError` listing any unrecognized tokens.

### Changed

- **`slot_table()` docstring corrected** (`correlator/mapping.py`). The docstring previously claimed the table was "trimmed to 64 rows". The default layout is 6 SNAPs × 12 ADCs = 72 slots, matching CAsMan hardware reality. The docstring now states 72 rows and notes that `Array64Config` in `bf_weights_generator` is a legacy artifact.

### Added

- **Post-translate beamforming assertion** (`correlator/mapping.py`). After translating a legacy CSV, `_translate_bf_weights_legacy_csv` now asserts that at least one row has `include_in_beamforming == 1`. A zero-count result indicates a malformed or fully-disabled source CSV and raises `ValueError` rather than returning a silently unusable mapping.
