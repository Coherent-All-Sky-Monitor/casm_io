# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - 2026-07-30

### Added

- **Triggered stream-layout voltage dumps** (`voltage/reader.py`, `voltage/configs/dada_format_stream.json`). `VoltageReader` reads dumps written to `stream_0 ... stream_5` (6 sub-bands x 512 channels, 484.375 -> 390.656 MHz after the 2026-03-27 band shift) as well as the legacy `chan0_1023 / chan1024_2047 / chan2048_3071` layout, choosing between them from the subdirectories in `data_dir`. A dump split over several files is ordered by the `OBS_OFFSET` in the filename and read as one timeline, so `n_time` counts from the start of the dump.

- **`filled_subbands` on `FullBandResult`**. Only the streams around a trigger are written; sub-bands with no data are zero-filled with a warning and their indices returned, so a caller can record which part of the band is zeros. A dump with no data at all still raises.

- **`allow_gaps` on `read_subband()` and `read_full_band()`**. Several dumps can share a `UTC_START`, so files matched by the timestamp prefix that do not run end to end in `OBS_OFFSET` raise by default, with the size of the gap and the full `UTC_START_OBSOFFSET` prefix that selects one dump. `allow_gaps=True` stitches across with a warning instead.

- **Cross-stream `OBS_OFFSET` alignment check**. The streams of one dump share an `OBS_OFFSET`; `read_full_band()` raises when they disagree rather than stitching different dumps into one band.

- **Bounds-checked antenna CSV extraction** (`voltage/reader.py`). `snap_id` / `adc` outside the hardware layout now raise, naming the offending rows. A negative `adc` previously wrapped round and returned a different antenna's voltages.

## [1.0.0] - 2026-05-18

### Fixed

- **Duplicate `antenna_id` silent overwrite in legacy CSV translator** (`correlator/mapping.py`). When a legacy `bf_weights_generator` CSV contained duplicate `ant64` values, `_translate_bf_weights_legacy_csv` would silently keep whichever row pandas happened to produce last. It now raises `ValueError` listing the offending `antenna_id` values.

- **Bool-token parser silently failing closed** (`correlator/mapping.py`). `_parse_bool` previously coerced unrecognized strings (e.g., `"1"`, `"yes"`, `"y"`) to `False` via a fallthrough path. It now accepts the full set `{true, 1, yes, y, t}` as True and `{false, 0, no, n, f, ""}` as False, and raises `ValueError` listing any unrecognized tokens.

### Changed

- **`slot_table()` docstring corrected** (`correlator/mapping.py`). The docstring previously claimed the table was "trimmed to 64 rows". The default layout is 6 SNAPs × 12 ADCs = 72 slots, matching CAsMan hardware reality. The docstring now states 72 rows and notes that `Array64Config` in `bf_weights_generator` is a legacy artifact.

### Added

- **Post-translate beamforming assertion** (`correlator/mapping.py`). After translating a legacy CSV, `_translate_bf_weights_legacy_csv` now asserts that at least one row has `include_in_beamforming == 1`. A zero-count result indicates a malformed or fully-disabled source CSV and raises `ValueError` rather than returning a silently unusable mapping.
