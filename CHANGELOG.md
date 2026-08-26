# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - 2026-07-30

### Added

- **`correlate(inputs=...)` selects the inputs to correlate**. (snap, adc) pairs against the snap dict, column indices against an array; the labels come back in `.inputs` in the requested order. The visibility cube and the einsum both go as the square of the input count, so this is what makes a native-resolution correlation affordable: 4 of 24 inputs over the full band is 600 MB and 2.3 s for 0.05 s of dump, against 14.2 MB per sample (432 GB/s) unselected. Only the requested columns are stacked, so an unused ADC is never copied.

- **`correlate()` takes a stream of gulps** (`voltage/correlate.py`). Passing `reader.iter_full_band(...)` instead of an array returns a generator of `CorrelateResult`, one per gulp, so a dump too long to unpack correlates with the same call as a short one. Integration bins are carried across the gulp boundaries and `time_s` is a global axis, making the result bit-identical to a single whole-dump call for any gulp size (verified on a 4 GB dump and in tests). Gulps may be `FullBandResult`s, snap dicts, or arrays; the stream is lazy.

- **Sub-band selection, time offsets and seconds-based reads** (`voltage/reader.py`). `read_full_band()` takes `subbands=` (a contiguous stream selection), `time_offset=`, and `seconds=`/`offset_seconds=` alongside the sample counts. Reads go through `np.memmap`, so only the requested samples and SNAP slots are pulled off disk.

- **`iter_full_band()`** yields `read_full_band()` results in RAM-sized chunks (gulp defaults to a fraction of MemAvailable), so long dumps are processed without holding the unpacked complex64 — 8x the disk size — in memory at once.

- **`casm_io.voltage.correlate()`**. Visibilities from voltages at the native 32.768 us resolution (`tint_s=None`) or any integration time; multiplies pairwise before averaging by construction. Accepts the raw snap dict (stacked in (snap, adc) order, labels in `.inputs`) or a per-antenna array. Returns `CorrelateResult`.

- **`casm-autocorr` console script** (`autocorr.py`). Accumulates a dump's mean power per SNAP/ADC in gulps, reads the correlator integration covering the dump window, and over-plots the two bandpasses per SNAP (dB relative to median), with progress and time ranges shown in UTC and Pacific time.

- **Quickstart notebook** (`examples/voltage_quickstart.ipynb`): dump from Python (casm_t2 `dump_voltages`), read sub-bands/SNAPs, correlate, per-SNAP autocorrelation figures. `examples/voltage_dumps.py` reworked around it: raw SNAP layout by default, `--streams/--snaps/--gulp`, correlation opt-in via `--correlate`/`--out`.

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
