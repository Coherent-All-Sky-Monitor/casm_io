# Correlator Visibilities

## Reading visibilities

```python
from casm_io import VisibilityReader, load_format
from casm_io.correlator.mapping import AntennaMapping

fmt = load_format("layout_64ant")
reader = VisibilityReader("/data/casm/visibilities_64ant", "2026-01-27-20:38:33", fmt)

# Inspect before reading
print(reader.n_files)            # number of .dat files found
print(reader.available_indices)  # [0, 1, 2, ...]
print(reader.missing_indices)    # any gaps in the sequence
print(reader.time_span)          # (start_unix, end_unix) — raw floats
print(reader.time_span_str())                        # human-readable UTC
print(reader.time_span_str("America/Los_Angeles"))   # human-readable Pacific

# Read all data (shows inline progress bar)
result = reader.read(freq_order='ascending')
print(result['vis'].shape)       # (T, 3072, 8256) complex64
print(result['freq_mhz'][:3])   # [375.000, 375.031, 375.061]
```

## Reading a fixed number of files

```python
# Read first 5 files (zero-fills any missing files with a warning)
result = reader.read(nfiles=5)

# nfiles and time_end are mutually exclusive
result = reader.read(
    time_start="2026-01-28T04:00:00",
    time_end="2026-01-28T14:00:00",
    time_tz="America/Los_Angeles",
)
# When time_tz is not UTC, the reader echoes both local and UTC times:
#   Requested (America/Los_Angeles): 2026-01-28 04:00:00 -> 2026-01-28 14:00:00
#   Requested (UTC):                 2026-01-28 12:00:00 -> 2026-01-28 22:00:00
```

## Verbosity and progress

All readers show inline progress bars and status messages by default. Pass `verbose=False` to silence:

```python
result = reader.read(nfiles=5, verbose=False)
```

### Time conversion helpers

`time_span_str(tz)` converts the observation time span to any timezone:

```python
reader.time_span_str()                        # '2026-01-27 20:38:33 UTC -> ...'
reader.time_span_str("America/Los_Angeles")   # '2026-01-27 12:38:33 PST -> ...'
```

The underlying helpers are also available directly:

```python
from casm_io._time import unix_to_iso, unix_to_datetime, format_time_span

unix_to_iso(1738099113.0, "America/Los_Angeles")   # '2026-01-28 12:38:33 PST'
unix_to_datetime(1738099113.0, "UTC")               # datetime(2026, 1, 28, 20, 38, 33, tz=UTC)
```

## Extracting specific baselines

```python
ant = AntennaMapping.load("/path/to/antenna_layout.csv")
ref_idx = ant.packet_index(antenna_id=4)
target_idxs = [ant.packet_index(a) for a in [5, 6, 7]]

result = reader.read(ref=ref_idx, targets=target_idxs)
print(result['vis'].shape)  # (T, 3072, 3) complex64
```

## Format configs

Two built-in formats:
- `"layout_32ant"` — 64 inputs (32 ant x 2 pol), 34.36s integrations
- `"layout_64ant"` — 128 inputs (64 ant x 2 pol), 137.44s integrations

```python
fmt = load_format("layout_64ant")           # built-in
fmt = load_format("/path/to/custom.json")   # custom JSON
```

New format: add a JSON file to `casm_io/correlator/configs/`. No code changes needed.

## Antenna mapping CSV

Required columns: `antenna_id`, `snap_id`, `adc`, `packet_index`.
Optional: `grid_code`, `kernel_index`, `pol`, `x_m`, `y_m`, `z_m`, `functional`.

Legacy column names (`antenna`, `snap`, `packet_idx`) are auto-detected.

## Frequency order

Default `freq_order='descending'` (native 468.75 -> 375 MHz). Pass `'ascending'` to reverse.
