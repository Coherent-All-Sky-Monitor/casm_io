# Correlator Visibilities

## `read_visibilities` (recommended entry point)

Scans `data_root` recursively for `visibilities_*` subdirectories, discovers all observations overlapping the requested time range, reads and stitches them, and warns about gaps. Use this for any time-range query spanning one or more observations.

```python
from casm_io.correlator import read_visibilities, load_format

fmt = load_format("layout_64ant")
result = read_visibilities(
    time_start="2026-05-16 11:30:00",
    time_end="2026-05-16 14:30:00",
    time_tz="America/Los_Angeles",  # recommended for OVRO operations
    data_root="/mnt",               # default; scans for visibilities_* dirs
    fmt=fmt,
    freq_order="descending",        # recommended (CASM native; default)
    verbose=True,                   # recommended for interactive use
)
```

### Parameters

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `time_start` | str or datetime | required | ISO format or datetime object |
| `time_end` | str or datetime | required | ISO format or datetime object |
| `time_tz` | str | `"UTC"` | Recommended: `"America/Los_Angeles"` for OVRO ops |
| `data_root` | str | `"/mnt"` | Scanned recursively for `visibilities_*` dirs; ignored when `data_dir` is set |
| `data_dir` | str or None | `None` | Explicit directory; bypasses `data_root` scan |
| `fmt` | VisibilityFormat or None | `None` | Required for headerless files before 2026-03-04; auto-detected from header otherwise |
| `ref` | int or None | `None` | Reference correlator input index for baseline extraction |
| `targets` | list[int] or None | `None` | Target input indices; defaults to all non-ref inputs when `ref` is set |
| `freq_order` | str | `"descending"` | Recommended: `"descending"` (CASM native, highest channel first) |
| `channels` | tuple(int, int) or None | `None` | `(ch_start, ch_end)` in native descending order; exclusive end; mutually exclusive with `freq_range_mhz` |
| `freq_range_mhz` | tuple(float, float) or None | `None` | `(freq_lo, freq_hi)` in MHz; mutually exclusive with `channels` |
| `verbose` | bool | `True` | Recommended `True` for interactive use; prints time spans, gap warnings, shapes |

### Return value

`VisibilityResult` with:

| Field | Type | Description |
|-------|------|-------------|
| `vis` | ndarray | `(T, F, n_bl)` complex64; or `(T, F, n_targets)` when `ref` is set |
| `freq_mhz` | ndarray | `(F,)` float64 in MHz |
| `time_unix` | ndarray | `(T,)` float64 Unix timestamps |
| `metadata` | dict | `observations`, `data_dirs`, `gaps`, `files`, format fields |

Both access styles work: `result.vis` and `result["vis"]`.

`metadata["gaps"]` is a list of dicts (each with `start_unix`, `end_unix`, `duration_s`) for any time ranges within the request that had no data. `metadata["observations"]` lists the `base_str` identifiers of each observation used.

### Frequency slicing

Two mutually exclusive options for reading a channel subset from disk (faster than loading the full band):

```python
# By frequency range in MHz
result = read_visibilities(..., freq_range_mhz=(440, 465))

# By channel index in native descending order (exclusive end)
result = read_visibilities(..., channels=(631, 1443))
```

In native descending order, channel 0 is the highest frequency. `freq_range_mhz=(440, 465)` maps to the channel range covering 440-465 MHz, which internally becomes a `(ch_start, ch_end)` pair.

For a post-load numpy slice when the full array is already in memory:

```python
mask = (freq_mhz >= 440) & (freq_mhz <= 465)
vis_sub = vis[:, mask, :]
```

### Baseline extraction

```python
from casm_io.correlator import AntennaMapping

ant = AntennaMapping.load("/path/to/layout.csv")
ref_pkt  = ant.packet_index(antenna_id=10)
tgt_pkts = [ant.packet_index(a) for a in [1, 2, 3, 5]]

result = read_visibilities(
    ...,
    ref=ref_pkt,
    targets=tgt_pkts,
)
# result.vis shape: (T, F, 4) complex64
# Conjugation is applied automatically so each baseline is V(ref, target)
```

When `ref` is set and `targets` is omitted, all other inputs are used as targets (full row of the correlation matrix).

### Single baseline from the full matrix

```python
from casm_io.correlator import triu_flat_index

nsig = fmt.nsig    # e.g. 128 for layout_64ant
i, j = ant.packet_index(5), ant.packet_index(12)
bl_idx = triu_flat_index(nsig, min(i, j), max(i, j))
v_pair = result.vis[:, :, bl_idx]   # (T, F) complex64
# Conjugate if i > j to get V(i, j):
if i > j:
    v_pair = v_pair.conj()
```

## `VisibilityReader` (file-level control)

Use `VisibilityReader` when you need to work with a single known observation directory, read a fixed number of files, or skip files.

```python
from casm_io.correlator import VisibilityReader, load_format

# Files with headers (post-2026-03-04): fmt is auto-detected
reader = VisibilityReader("/mnt/nvme4/data/casm/visibilities_64ant", "2026-03-27-07:56:53")

# Older headerless files: pass fmt explicitly
fmt = load_format("layout_64ant")
reader = VisibilityReader("/mnt/nvme4/data/casm/visibilities_64ant", "2026-01-27-20:38:33", fmt)

print(reader.n_files)           # number of .dat files found
print(reader.available_indices) # [0, 1, 2, ...]
print(reader.missing_indices)   # gaps in the file sequence
print(reader.time_span)         # (start_unix, end_unix)
print(reader.time_span_str("America/Los_Angeles"))
```

### `VisibilityReader.read()`

```python
result = reader.read(
    time_start="2026-01-28 04:00:00",
    time_end="2026-01-28 06:00:00",
    time_tz="America/Los_Angeles",
    freq_order="descending",
    verbose=True,
)
```

All parameters from `read_visibilities` apply, plus:

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `nfiles` | int or None | `None` | Read exactly this many files; mutually exclusive with `time_end` |
| `skip_nfiles` | int | `0` | Skip this many files before reading; requires `nfiles` |

Parameter combinations:

| Combo | Valid |
|-------|-------|
| `nfiles=5` | Yes: files 0-4 |
| `nfiles=5, skip_nfiles=10` | Yes: files 10-14 |
| `time_start=..., nfiles=5` | Yes: start from that time, read 5 files |
| `time_start=..., time_end=...` | Yes: exact window |
| `nfiles=..., time_end=...` | No: mutually exclusive |
| `skip_nfiles=...` without `nfiles` | No: skip requires nfiles |

Missing files within a `nfiles` request are zero-filled with a warning. Missing files within a time-range request raise `RuntimeError`.

## Format configurations

```python
from casm_io.correlator import load_format

fmt = load_format("layout_64ant")            # built-in
fmt = load_format("/path/to/custom.json")    # custom
```

Built-in formats:

| Name | Inputs | Integration time | File duration |
|------|--------|-----------------|---------------|
| `layout_32ant` | 64 (32 ant x 2 pol) | ~0.54 s | 34.36 s |
| `layout_64ant` | 128 (64 ant x 2 pol) | ~1.07 s | 137.44 s |

Both formats cover 3072 channels: `layout_32ant` spans 468.75-375.03 MHz, `layout_64ant` 484.375-390.66 MHz after the 2026-03-27 band shift. The exact top frequency reads from the file header where one is present; see the band-shift note below.

`VisibilityFormat` fields: `nsig`, `nchan`, `dt_raw_s`, `ntime_per_file`, `chan_bw_mhz`, `freq_top_mhz`, `freq_bottom_mhz`, `n_baselines`, `file_duration_s`.

```python
fmt.get_frequency_axis(order="descending")   # (nchan,) float64 in MHz
fmt.freq_to_channel(freq_mhz)               # native descending channel index
fmt.freq_range_to_channels(lo, hi)          # (ch_start, ch_end) exclusive
```

## Baseline indexing

Visibilities are stored as a flattened upper-triangular matrix (diagonal = autocorrelations). The flat index ordering matches `np.triu_indices(nsig)`.

```python
from casm_io.correlator import triu_flat_index, triu_to_ij

n_bl = fmt.n_baselines          # nsig*(nsig+1)//2

# Flat index for inputs (i, j); requires i <= j
bl_idx = triu_flat_index(nsig, i, j)

# Reverse: flat index back to (i, j)
i, j = triu_to_ij(nsig, bl_idx)
```

Autocorrelation for input `p`: `triu_flat_index(nsig, p, p)`.

Cross-correlation between antenna IDs `a` and `b`:

```python
pi, pj = ant.packet_index(a), ant.packet_index(b)
bl_idx = triu_flat_index(nsig, min(pi, pj), max(pi, pj))
v = vis[:, :, bl_idx]
if pi > pj:
    v = v.conj()   # orient as V(a, b) = V*(b, a)
```

## Verbose output

With `verbose=True` (default), reads print:

```
Requested (UTC): 2026-05-16 18:30:00 UTC -> 2026-05-16 21:30:00 UTC
Requested (PT):  2026-05-16 11:30:00 PDT -> 2026-05-16 14:30:00 PDT
Scanning /mnt ...
  Found: visibilities_64ant/
  visibilities_64ant/: 2 observations (2026-05-16-18:04:12 -> 2026-05-16-19:31:57)

Matching observations: 2
  [1] 2026-05-16-18:04:12  (82 files)
  [2] 2026-05-16-19:31:57  (70 files)

Reading 2026-05-16-18:04:12 (2026-05-16 18:30:00 UTC -> ...) ...
  Reading files [==============================] 82/82
...
Final output shape: (227, 3072, 8256) (complex64)
```

Pass `verbose=False` to silence all output.

## Gotchas

**Headerless old files**: `VisibilityReader` raises `ValueError` if there is no 4096-byte file header and no `fmt` argument. Pass `fmt=load_format("layout_64ant")` for files written before 2026-03-04.

**Band shift 2026-03-27**: the top of the correlator band shifted from 468.75 MHz to 484.375 MHz (voltage dumps shifted with it). For post-shift files, the reader reads `FREQ_START` from the file header and overrides `fmt.freq_top_mhz` automatically. If you pass an explicit `fmt` that disagrees with the header, you get a warning and the header value wins.

**`channels` and `freq_range_mhz` are mutually exclusive**. Passing both raises `ValueError`.

**`nfiles` and `time_end` are mutually exclusive** in `VisibilityReader.read()`. Passing both raises `ValueError`.

**Frequency order**: native order is descending (highest channel first). Channel 0 is the highest frequency. `freq_range_to_channels(lo, hi)` returns `(ch_start, ch_end)` where `ch_start` corresponds to `freq_hi` because higher frequency = lower channel index.

## Low-level utilities

These are exported from `casm_io.correlator` for scripting and tooling.

```python
from casm_io.correlator import (
    discover_files,
    discover_observations,
    write_npz,
    read_npz,
)

# List all .dat files in a data_dir for a given base_str
idx_to_path = discover_files(data_dir, base_str)
# returns dict[int -> str] mapping file index to absolute path

# Scan a data_dir and return metadata for each observation found
obs_list = discover_observations(data_dir, fmt=None, verbose=False)
# returns list of dicts, each with: base_str, n_files, time_start,
# time_end, fmt, data_dir; sorted by time_start.
# Observations with no readable header and no explicit fmt are skipped.

# Save and reload visibility arrays as compressed NPZ
write_npz("vis.npz", vis, freq_mhz, time_unix)
result = read_npz("vis.npz")  # returns dict with vis, freq_mhz, time_unix
```

`discover_observations` is what `read_visibilities` calls internally. Use it directly only if you need to inspect what is on disk before committing to a read.

## Time utilities

```python
from casm_io._time import unix_to_iso, unix_to_datetime, format_time_span

unix_to_iso(result.time_unix[0])                          # '2026-05-16 18:30:02 UTC'
unix_to_iso(result.time_unix[0], "America/Los_Angeles")   # '2026-05-16 11:30:02 PDT'
format_time_span(result.time_unix[0], result.time_unix[-1], "America/Los_Angeles")
```
