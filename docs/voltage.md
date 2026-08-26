# Voltage DADA Files

Two layouts exist on disk: triggered dumps written to per-stream directories
(current), and the legacy three-subband dumps from before March 2026.
`VoltageReader` picks the layout from the subdirectories it finds, so the calls
are identical. Two walkthroughs exist: the notebook
[examples/voltage_quickstart.ipynb](../examples/voltage_quickstart.ipynb)
(trigger from Python via casm_t2's `dump_voltages`, read, correlate) and the
script [examples/voltage_dumps.py](../examples/voltage_dumps.py) for batch use.
`casm-autocorr` compares a dump's autocorrelations against the correlator.

## Reading a triggered dump

Triggered dumps are made with `casm-voltage-dump` (lives in casm_t2) and land in
`/mnt/nvme4/data/casm/cand_dumps/stream_0 ... stream_5` — six sub-bands of 512
channels, frequency descending, stream 0 at the top of the band. Files are named
`<UTC_START>_<OBS_OFFSET>.<filenum>.dada`. Add `--gather DIR` to pull all six
streams into a working directory (the other node's by rsync, the local ones as
symlinks) and print the reader call ready-made; `dump_voltages(dir,
past_duration=...)` does the same from Python.

```python
from casm_io import VoltageReader

reader = VoltageReader("/mnt/nvme4/data/casm/cand_dumps",
                       "2026-07-29-21:06:34_0166501757706240")
print(reader.subbands_found)      # [0, 1, 2, 3, 4, 5]

# Stitch all streams to the full band and extract per-antenna
result = reader.read_full_band(
    antenna_csv="/home/casm/software/dev/antenna_layouts/current",
    seconds=1,                    # or n_time=30518 samples at TSAMP = 32.768 us
)
print(result.voltages.shape)      # (30518, 3072, n_ant) complex64, n_ant = CSV rows
print(result.freq_mhz[[0, -1]])   # [484.375, 390.656] — descending, band shifted 2026-03-27
print(result.utc_start)           # '2026-07-29-21:06:34'
print(result.antenna_df.head())   # antenna mapping DataFrame, one row per output antenna
```

Every CSV row is bounds-checked against the layout before extraction: `snap_id`
and `adc` outside the hardware range raise, naming the row, rather than silently
handing back a different antenna. An antenna whose SNAP was not read comes back
as zeros with a message.

Without `antenna_csv` you get the raw layout: `{snap_id: (n_time, 3072, 12)}`,
12 being the ADC inputs on a SNAP board. `snaps=` picks which SNAP slots to
unpack — the config's `active_snaps` by default, which is all 11 SNAP slots for
stream dumps and `[0, 2, 4]` for the legacy layout.

```python
result = reader.read_full_band(n_time=1000, snaps=[0, 2, 4])
print(result.voltages[0].shape)   # (1000, 3072, 12) — SNAP 0
```

`subbands=` reads only some streams (the selection must be contiguous so the
stitched frequency axis has no hole), and `time_offset=` / `offset_seconds=`
start the read inside the dump. Reads go through `np.memmap`, so only the
requested samples and SNAP slots are pulled off disk:

```python
result = reader.read_full_band(subbands=[1, 2], seconds=0.5, offset_seconds=1.0)
print(result.voltages[0].shape)   # (15259, 1024, 12) — 468.75-437.53 MHz
```

For dumps too long to unpack in memory at once (complex64 is 8x the disk
size), `iter_full_band()` yields the same reads in RAM-sized chunks — the
gulp defaults to a fraction of the memory currently free:

```python
power = 0
for chunk in reader.iter_full_band(seconds=30, snaps=[0]):
    v = chunk.voltages[0]         # each chunk is a FullBandResult
    power = power + np.sum(v.real**2 + v.imag**2, axis=0)
```

## Visibilities from voltages

`correlate()` multiplies the inputs pairwise and then averages — in that
order, which is what makes it a correlation. `tint_s=None` keeps the native
32.768 us resolution; any other integration time rounds to whole samples.
The raw snap dict is stacked along the input axis in (snap, adc) order and
the labels come back in `.inputs`; a per-antenna array keeps its own order.

```python
from casm_io.voltage import correlate

out = correlate(result.voltages, tint_s=0.01)
out.vis           # (n_bin, n_chan, n_input, n_input) complex64
out.time_s        # bin-centre seconds from the start of the array
out.inputs        # [(snap, adc), ...] for dict input, None for array input
```

### Correlating a dump too long to hold

Hand `correlate()` the iterator instead of the array and it returns a
generator of the same results, one per gulp:

```python
KEEP = [(0, 0), (0, 1), (3, 0), (3, 1)]     # the (snap, adc) inputs you want

vis, time_s = [], []
for out in correlate(reader.iter_full_band(seconds=0.05, snaps=[0, 3]),
                     inputs=KEEP):
    vis.append(out.vis)                     # (n_bin, n_chan, 4, 4)
    time_s.append(out.time_s)
vis = np.concatenate(vis)
time_s = np.concatenate(time_s)
```

That is a correlation with **no averaging at all**: native 32.768 us time
resolution and all 3072 channels. `inputs=` is what makes it affordable, the
cube and the work to build it both going as the square of the input count.
For 0.05 s of 4 inputs that is `(1526, 3072, 4, 4)`, 600 MB in 2.3 s; the
same read without `inputs=` is 24 inputs, 14.2 MB per sample, 432 GB per
second of dump.

`seconds=` here bounds **how much dump is read**, nothing to do with
averaging. It is in this example only to keep the output small: dropping it
reads the whole dump, and a 4 s dump with no averaging is 48 GB for those
same 4 inputs. Read a window with `seconds=`/`offset_seconds=`, average with
`tint_s=`, or write each gulp to disk; the three are independent choices.

Integration bins are carried across the gulp boundaries and `time_s` counts
from the start of the stream, so the visibilities are bit-identical to one
whole-dump call whatever the gulp size, including a gulp shorter than one
integration. Nothing is read until the generator is advanced.

`inputs=` takes (snap, adc) pairs against the snap dict and column indices
against an array read with `antenna_csv=`. The pairs come back in
`out.inputs`, in the order you asked for, and only those columns are ever
stacked.

### Averaging in time

Add `tint_s=`. It rounds to whole samples and `out.tint_samples` reports
what was used: 1 ms is 30.5 samples, so it becomes 31 (1.0158 ms).

```python
for out in correlate(reader.iter_full_band(seconds=0.05, snaps=[0, 3]),
                     inputs=KEEP, tint_s=0.001):
    vis.append(out.vis)                     # (49, 3072, 4, 4), 19.3 MB
```

### Averaging in frequency

Frequency is axis 1, and `correlate()` does not touch it, so bin it yourself
inside the loop. Whole band:

```python
    vis.append(out.vis.mean(axis=1))        # (n_bin, 4, 4)
```

or in groups of `n_f` channels, keeping some spectral resolution:

```python
    v = out.vis
    vis.append(v.reshape(v.shape[0], v.shape[1] // n_f, n_f, *v.shape[2:])
                .mean(axis=2))              # n_f=16 -> (1526, 192, 4, 4)
```

The two are independent: `tint_s` averages time, the reshape averages
frequency, and either can be used without the other. Average the
visibilities, never the voltages. Note that band-averaging a cross-baseline
decoheres across 93.75 MHz unless the delay has been taken out first.

Reduce inside the loop or write each gulp to a
`np.lib.format.open_memmap` on disk. Appending the raw `out.vis` for a long
dump just rebuilds the whole cube in RAM and undoes the point of streaming.

Only the streams around the trigger are written. Streams with no data are
zero-filled with a warning; the read fails only if no stream has data at all.
Their indices come back in `result.filled_subbands`, so a script can record which
part of the band is zeros:

```python
if result.filled_subbands:
    print(f"no data for streams {result.filled_subbands} — zeros")
```

The streams of one dump carry the same `OBS_OFFSET` in their first file. If they
don't, the timestamp prefix has picked up different dumps in different stream
directories and `read_full_band()` raises, naming the streams and their offsets.

A dump longer than about 10 seconds is split over several files. They are found
by timestamp prefix, ordered by the `OBS_OFFSET` in the filename and read as one
timeline, so `n_time` counts from the start of the dump. Several dumps can share
a `UTC_START`, so when consecutive files don't run end to end the read raises
with the size of the gap. The full `UTC_START_OBSOFFSET` prefix selects one dump:

```python
reader = VoltageReader(dump_dir, "2026-07-29-21:06:34_0166501757706240")
```

Pass `allow_gaps=True` to `read_subband()` or `read_full_band()` to stitch across
the gap anyway (a warning replaces the error). The samples either side are then
not contiguous in time.

Pass `config=` (a path to a JSON format config, or a dict) to override the layout
choice.

## Legacy dumps (before March 2026)

Pre-band-shift dumps live in `/mnt/nvme3/data/casm/voltage_dumps` as three
1024-channel directories (`chan0_1023`, `chan1024_2047`, `chan2048_3071`)
covering 468.75 -> 375.03 MHz. All three subbands must be present; unlike the
stream layout, a missing one raises.

```python
reader = VoltageReader("/mnt/nvme3/data/casm/voltage_dumps", "2026-02-17-21:10:43")
print(reader.subbands_found)      # [0, 1, 2]

result = reader.read_full_band(
    antenna_csv="/path/to/antenna_layout.csv",
    freq_order='ascending',
    n_time=1000,
)
print(result.voltages.shape)      # (1000, 3072, n_ant) complex64
print(result.freq_mhz[[0, -1]])   # [375.031, 468.750]
print(result.utc_start)           # '2026-02-17-21:10:43'
```

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
| `freq_mhz` | `np.ndarray` | Frequency axis for this subband (512 channels for a stream, 1024 for the legacy layout) |

Both access styles work:

```python
result.voltages          # attribute access
result['voltages']       # dict-style (backwards compatible)
```

## Reading a single subband

```python
result = reader.read_subband(0, n_time=100)
print(result.voltages[0].shape)    # (100, 512, 12) — SNAP 0, one stream
print(result.freq_mhz[[0, -1]])    # [484.375, 468.781] for stream 0
print(result.header['UTC_START'])  # header field access
```

Legacy subbands are 1024 channels wide, so the same call returns `(100, 1024, 12)`.

## Verbosity and progress

Reads show inline progress bars by default:

```
  Reading subbands [==========>                   ] 2/6
  Extracting antennas [=============>            ] 24/48
```

The legacy layout also shows an `Unpacking SNAPs` bar; stream reads print one
line per file instead, with the sample count taken from that file.

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

Default `freq_order='descending'`, the native order: 484.375 -> 390.656 MHz for
dumps taken after the 2026-03-27 band shift, 468.75 -> 375.03 MHz for legacy
dumps. Pass `'ascending'` to reverse. Either way the axis is built from the
sub-band index and the format config, never from the header — `FREQ_START` is
off-grid for stream 0, and `START_CHANNEL` is one sub-band width low for
pre-shift data (the reader prints a note when it disagrees).
