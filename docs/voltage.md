# Voltage DADA Files

## Reading a full voltage dump

```python
from casm_io import VoltageReader

reader = VoltageReader("/mnt/nvme3/data/casm/voltage_dumps", "2026-02-17-21:10:43")
print(reader.subbands_found)  # [0, 1, 2]

# Read all 3 subbands, stitch to full band, extract per-antenna
result = reader.read_full_band(
    antenna_csv="/path/to/antenna_layout.csv",
    freq_order='ascending',
    n_time=1000,
)
print(result.voltages.shape)      # (1000, 3072, 16) complex64
print(result.freq_mhz[[0, -1]])   # [375.000, 468.719]
print(result.utc_start)            # '2026-02-17-21:10:43'
print(result.antenna_df.head())    # antenna mapping DataFrame
```

Every CSV row is bounds-checked against the layout before extraction: `snap_id`
and `adc` outside the hardware range raise, naming the row, rather than silently
handing back a different antenna.

## Triggered dumps (stream_0 ... stream_5)

Triggered dumps use per-stream directories with 512 channels each instead of the
three 1024-channel `chan*` directories. `VoltageReader` picks the layout from the
subdirectories it finds, so the calls are the same:

```python
reader = VoltageReader("/mnt/nvme3/data/casm/cand_dumps", "2026-03-16-00:30:11")
print(reader.subbands_found)  # [0, 1, 2] — streams 3-5 weren't written

result = reader.read_full_band(n_time=1000, snaps=[0, 2, 4])
print(result.voltages[0].shape)   # (1000, 3072, 12)
```

Streams that weren't written are zero-filled with a warning; the read only fails
if no stream has data. Their indices come back in `result.filled_subbands`, so a
script can record which part of the band is zeros:

```python
if result.filled_subbands:
    print(f"no data for streams {result.filled_subbands} — zeros")
```

Stream 0 is the top of the band, as with the legacy layout.

The streams of one dump carry the same `OBS_OFFSET` in their first file. If they
don't, the timestamp prefix has picked up different dumps in different stream
directories and `read_full_band()` raises, naming the streams and their offsets.

A dump longer than about 10 seconds is split over several files. They are found
by timestamp prefix, ordered by the `OBS_OFFSET` in the filename and read as one
timeline, so `n_time` counts from the start of the dump. Files sharing a
UTC_START can be separate dumps, so when consecutive files don't run end to end
the read raises with the size of the gap. Extend the timestamp with the offset
to pick one dump:

```python
reader = VoltageReader(dump_dir, "2026-03-16-00:30:11_0000044756219904")
```

Pass `allow_gaps=True` to `read_subband()` or `read_full_band()` to stitch across
the gap anyway (a warning replaces the error). The samples either side are then
not contiguous in time.

Pass `config=` (a path to a JSON format config, or a dict) to override the layout
choice.

## Result attributes

### `FullBandResult` (from `read_full_band()`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `voltages` | `np.ndarray` or `dict` | `(n_time, 3072, n_ant)` if antenna_csv provided, else `{snap_id: (n_time, 3072, n_adc)}` |
| `header` | `dict` | Parsed DADA header from first subband |
| `freq_mhz` | `np.ndarray` | Full 3072-channel frequency axis |
| `utc_start` | `str` | UTC_START from header |
| `antenna_df` | `DataFrame` or `None` | Antenna mapping if CSV was provided |
| `filled_subbands` | `list[int]` | Sub-band indices zero-filled because no data was found; empty for a complete read |

### `SubbandResult` (from `read_subband()`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `voltages` | `dict` | `{snap_id: (n_time, n_chan_per_sub, n_adc) complex64}` |
| `header` | `dict` | Parsed DADA header |
| `freq_mhz` | `np.ndarray` | Frequency axis for this subband (1024 channels) |

Both access styles work:

```python
result.voltages          # attribute access
result['voltages']       # dict-style (backwards compatible)
```

## Reading a single subband

```python
result = reader.read_subband(0, n_time=100)
print(result.voltages[0].shape)   # (100, 1024, 12) — SNAP 0 data
print(result.freq_mhz.shape)      # (1024,)
print(result.header['UTC_START'])  # header field access
```

## Verbosity and progress

Reads show inline progress bars by default:

```
  Reading subbands [===========>                  ] 1/3
  Unpacking SNAPs  [==============================] 6/6
```

Pass `verbose=False` to silence:

```python
result = reader.read_subband(0, verbose=False)
result = reader.read_full_band(verbose=False)
```

## Time conversion for voltage data

```python
from casm_io._time import unix_to_iso

# UTC_START from header
print(result.utc_start)

# If you need datetime objects
from datetime import datetime
dt = datetime.fromisoformat(result.utc_start)
```

## Header trust

Default `trust_header=False` substitutes known-good defaults for unreliable header fields.

**Trusted**: `UTC_START`, `TSAMP`, `NCHAN`, `NBIT`, `NDIM`, `ENCODING`, `RESOLUTION`, `HDR_SIZE`, `BW`, `UDP_NANT`, `STREAM_SUBBAND_ID`, `FREQ`

**Untrusted**: `NANT`, `FILE_SIZE`, `SOURCE`, `NPOL`, `PICOSECONDS`, `START_CHANNEL`, `END_CHANNEL`

## Frequency order

Default `freq_order='descending'` (native 468.75 -> 375 MHz). Pass `'ascending'` to reverse.
