# Correlator Visibilities

## Reading visibilities

```python
from casm_io import VisibilityReader, load_format

fmt = load_format("layout_64ant")
reader = VisibilityReader("/data/casm/visibilities_64ant", "2026-01-27-20:38:33", fmt)

# Inspect before reading
print(reader.n_files)            # number of .dat files found
print(reader.available_indices)  # [0, 1, 2, ...]
print(reader.missing_indices)    # any gaps in the sequence

# Read all data
result = reader.read(freq_order='ascending')
print(result.vis.shape)          # (T, 3072, 8256) complex64
print(result.freq_mhz[:3])      # [375.000, 375.031, 375.061]
print(result.time_unix[:3])      # unix timestamps per integration
print(result.metadata)           # format info, files used, etc.
```

## Result attributes

`reader.read()` returns a `VisibilityResult` dataclass:

| Attribute | Type | Description |
|-----------|------|-------------|
| `vis` | `np.ndarray` | `(T, F, n_baselines)` or `(T, F, n_targets)` complex64 |
| `freq_mhz` | `np.ndarray` | Frequency axis in MHz |
| `time_unix` | `np.ndarray` | Unix timestamps per integration |
| `metadata` | `dict` | Format info, files used, missing_files, freq_order |
| `ref` | `int` or `None` | Reference input index (if baseline extraction used) |
| `targets` | `list` or `None` | Target input indices (if baseline extraction used) |

Both access styles work:

```python
result.vis           # attribute access
result['vis']        
```

## Reading a fixed number of files

```python
# Read first 5 files (zero-fills any missing files with a warning)
result = reader.read(nfiles=5)

# Skip files before reading
result = reader.read(nfiles=5, skip_nfiles=10)  # reads files 10-14

# Combine with time_start
result = reader.read(
    time_start="2026-01-28T04:00:00",
    time_tz="America/Los_Angeles",
    nfiles=5,
)
```

### Parameter rules

| Combo | Valid? |
|-------|--------|
| `nfiles=5` | Yes — read files 0-4 |
| `nfiles=5, skip_nfiles=10` | Yes — read files 10-14 |
| `time_start=..., nfiles=5` | Yes — start from that time, read 5 files |
| `time_start=..., nfiles=5, skip_nfiles=3` | Yes — start from time, skip 3, read 5 |
| `time_start=..., time_end=...` | Yes — exact time window |
| `nfiles=..., time_end=...` | No — mutually exclusive |
| `skip_nfiles=...` without `nfiles` | No — skip requires nfiles |

## Reading by time range

```python
result = reader.read(
    time_start="2026-01-28T04:00:00",
    time_end="2026-01-28T14:00:00",
    time_tz="America/Los_Angeles",
)
# Output echoes both local and UTC:
#   Requested (America/Los_Angeles): 2026-01-28 04:00:00 -> 2026-01-28 14:00:00
#   Requested (UTC):                 2026-01-28 12:00:00 -> 2026-01-28 22:00:00
```

`time_tz` defaults to `"UTC"`. Naive strings are interpreted in the given timezone. 

## Time span and timezone conversion

```python
# Raw unix floats
reader.time_span                                     # (start_unix, end_unix)

# Human-readable in any timezone
reader.time_span_str()                               # '2026-01-27 20:38:33 UTC -> ...'
reader.time_span_str("America/Los_Angeles")          # '2026-01-27 12:38:33 PST -> ...'
```

### Converting timestamps from results

```python
from casm_io._time import unix_to_iso, unix_to_datetime, format_time_span

result = reader.read(nfiles=5)

# Single timestamp
unix_to_iso(result.time_unix[0])                         # '2026-01-28 00:41:50 UTC'
unix_to_iso(result.time_unix[0], "America/Los_Angeles")  # '2026-01-27 16:41:50 PST'

# As datetime object
unix_to_datetime(result.time_unix[0], "America/Los_Angeles")

# Format a range
format_time_span(result.time_unix[0], result.time_unix[-1], "America/Los_Angeles")

# Convert entire array
times_pt = [unix_to_iso(t, "America/Los_Angeles") for t in result.time_unix]
```

## Verbosity and progress

All reads show inline progress bars and time span info (UTC + Pacific) by default:

```
Data span (UTC): 2026-02-28 00:41:50 UTC -> 2026-03-02 23:33:16 UTC
Data span (PT):  2026-02-27 16:41:50 PST -> 2026-03-02 15:33:16 PST
Files available: 58 (indices 0-57)
Reading 5 files starting from index 0
Time range (UTC): 2026-02-28 00:41:50 UTC -> 2026-02-28 06:48:20 UTC
Time range (PT):  2026-02-27 16:41:50 PST -> 2026-02-27 22:48:20 PST
  Reading files [==============================] 5/5
```

Pass `verbose=False` to silence:

```python
result = reader.read(nfiles=5, verbose=False)
```

## Extracting specific baselines

```python
from casm_io.correlator.mapping import AntennaMapping

ant = AntennaMapping.load("/path/to/antenna_layout.csv")
ref_idx = ant.packet_index(antenna_id=4)
target_idxs = [ant.packet_index(a) for a in [5, 6, 7]]

result = reader.read(ref=ref_idx, targets=target_idxs)
print(result.vis.shape)   # (T, 3072, 3) complex64
print(result.ref)          # 4
print(result.targets)      # [5, 6, 7]
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
