# casm_io

Unified I/O library for CASM (Coherent All-Sky Monitor) data products at OVRO.

- **Correlator visibilities** — binary `.dat` files
- **Voltage DADA dumps** — 4+4 bit complex `.dada` files (3-subband)
- **Filterbank files** — SIGPROC `.fil` format (read/write + quick-look plots)
- **Candidates** — FRB search candidate lists (Hella T1)

## Install

```bash
source ~/software/dev/casm_venvs/casm_offline_env/bin/activate
cd /home/casm/software/dev/casm_io
pip install --no-build-isolation -e .
```

## Quick Start

### Correlator visibilities

**Recommended: `read_visibilities()`** — give it a time range and it figures out everything:

```python
from casm_io.correlator import read_visibilities

# Just time range — auto-discovers data directory and observations
result = read_visibilities("2026-03-24 06:00", "2026-03-24 08:00")
print(result.vis.shape)          # (52, 3072, 8256) complex64

# With frequency channel slicing 
result = read_visibilities(
    "2026-03-24 06:00", "2026-03-24 08:00",
    channels=(500, 506),         # native descending channel indices
)
print(result.vis.shape)          # (52, 6, 8256) complex64

# Or specify frequency range in MHz
result = read_visibilities(
    "2026-03-24 06:00", "2026-03-24 08:00",
    freq_range_mhz=(420, 425),   # auto-converted to channel indices
    freq_order="ascending",
)

# Timezone support
result = read_visibilities(
    "2026-03-23 23:00", "2026-03-24 01:00",
    time_tz="America/Los_Angeles",
    channels=(500, 506),
)

# Baseline extraction
result = read_visibilities(
    "2026-03-24 06:00", "2026-03-24 08:00",
    ref=0, targets=[1, 2, 3],
    channels=(500, 506),
)

# Both access styles work
result.vis                       # attribute access
result['vis']                    # dict-style access
result.metadata['gaps']          # any data gaps in the time range
result.metadata['observations']  # which observations were used
```

**Low-level: `VisibilityReader`** — when you need file-level control:

```python
from casm_io.correlator import VisibilityReader, load_format

# New files (since March 4 2026) — format auto-detected from header
reader = VisibilityReader("/data/casm/visibilities_64ant", "2026-03-05-08:02:39")
result = reader.read(nfiles=5, channels=(500, 506))

# Old files — pass fmt explicitly
fmt = load_format("layout_64ant")
reader = VisibilityReader("/data/casm/visibilities_64ant", "2026-01-27-20:38:33", fmt)
result = reader.read(nfiles=5, skip_nfiles=10)
```

### Voltage DADA files

```python
from casm_io import VoltageReader

reader = VoltageReader("/mnt/nvme3/data/casm/voltage_dumps", "2026-02-17-21:10:43")
result = reader.read_full_band(antenna_csv="/path/to/antenna_layout.csv")
print(result.voltages.shape)     # (ntime, 3072, 16) complex64
print(result.utc_start)          # '2026-02-17-21:10:43'
```

### Filterbank files

```python
from casm_io import FilterbankFile

fb = FilterbankFile("/path/to/beam.fil")   # prints backend and shape
print(fb.nchans, fb.nsamples)              # header info, no data loaded yet
print(fb.data.shape)                       # (nsamples, nchans) — loaded on access
print(fb.backend_used)                     # "sigpyproc" or "standalone"

fb = FilterbankFile("/path/to/beam.fil", verbose=False)  # silence output
```

### Antenna mapping

```python
from casm_io.correlator import AntennaMapping

# Load the canonical layout CSV (resolves $CASM_LAYOUT_CSV / current symlink
# when called with no argument) or pass an explicit path.
ant = AntennaMapping.load("/path/to/antenna_layout.csv")
ant.packet_index(antenna_id=5)         # 30
ant.snap_adc(antenna_id=5)             # (2, 6)
ant.format_antenna(5)                  # 'Ant 5 | S2A6 -> input 30'
ant.active_antennas()                  # functional==1 rows

# Mark antennas inactive at runtime without editing the CSV
ant_clean = ant.with_inactive([3, 7])  # drop ants 3 and 7
ant_subset = ant.with_only([1, 2, 5])  # keep only these

# Dense per-slot helpers (default n_snaps=6, n_adc=12 → 72 slots, the
# full CAsMan hardware reality):
ant.positions_64()        # (72, 3) ENU, zeros for unwired slots
ant.active_mask_64()      # (72,) bool: wired & functional & in beamforming
ant.antenna_ids_64()      # (72,) int: antenna_id or -1
```

`AntennaMapping.load` accepts two CSV schemas:

1. **Canonical** (what `casm-build-layout` writes): columns
   `antenna_id`/`snap_id`/`adc`/`packet_index` plus optional
   `x_m`/`y_m`/`z_m`/`functional`/`include_in_beamforming`. Legacy
   column aliases (`antenna`, `snap`, `packet_idx`, `feng_id`,
   `feng_idx`, `x`/`y`/`z`) are auto-renamed.
2. **Legacy `bf_weights_generator`** (old fixtures): columns
   `pos_id`/`snap_A`/`adc_A`/`ant64`/`x_east_m`/`y_north_m`/`z_up_m`.
   Auto-detected when all four marker columns are present and
   translated in place (`antenna_id = ant64 + 1`,
   `packet_index = snap_A*12 + adc_A`). Boolean columns
   (`include_in_beamforming`, `installed`) accept any of
   `true/false`, `1/0`, `yes/no`, `y/n`, `t/f` (case-insensitive); the
   translator raises on unrecognized tokens or duplicate antenna_ids.

### Candidates

```python
from casm_io import CandidateReader

cands = CandidateReader("/path/to/t1_candidates.txt")
print(cands.n_candidates, cands.snr_range, cands.dm_range)
print(cands.df.head())
```

**plot_candidate** — plot a dedispersed waterfall and time series for a single candidate:

```python
from casm_io.candidates import CandidateReader, plot_candidate
from casm_io.filterbank import FilterbankFile

fb = FilterbankFile("/path/to/injection.fil", verbose=False)
cr = CandidateReader("/path/to/candidates.out")
best = int(cr.df["snr"].values.argmax())
plot_candidate(fb, best, cr, output_path="best_candidate.png")
```

`nsub` (default 64) controls the number of frequency subbands in the waterfall. `margin_factor` (default 2.0) sets the time window around the candidate in seconds.

**CandidateMatcher** — match hella candidates against a known injection using DM and time windows. Band parameters are read directly from the filterbank header.

```python
from casm_io.candidates import CandidateReader, CandidateMatcher
from casm_io.filterbank import FilterbankFile

fb = FilterbankFile("/path/to/injection.fil", verbose=False)
cr = CandidateReader("/path/to/candidates.out")
matcher = CandidateMatcher(fb)
result = matcher.match(cr.df, dm_true=100.0, fwhm_samples=5.0)
if result["detected"]:
    best = result["best"]
    print(f"SNR={best['snr']:.1f}  DM={best['dm']:.1f}")
```

## Documentation

- [Correlator visibilities](docs/correlator.md)
- [Voltage DADA files](docs/voltage.md)
- [Filterbank files](docs/filterbank.md)
- [Candidates](docs/candidates.md)

## Testing

```bash
python -m pytest tests/ -v
```
