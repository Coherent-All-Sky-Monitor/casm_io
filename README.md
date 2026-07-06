# casm_io

Unified I/O library for CASM (Coherent All-Sky Monitor) data products at OVRO. Provides correlator visibilities, voltage DADA dumps, filterbank files, and FRB candidates behind a consistent Python API. No fringe-stopping, calibration, or beamforming logic lives here.

## Install

```bash
source ~/software/dev/casm_venvs/casm_offline_env/bin/activate
cd /home/casm/software/dev/casm_io
pip install --no-build-isolation -e .
```

## Correlator visibilities

`read_visibilities` auto-discovers every `visibilities_*` subdirectory under `data_root`, finds all observations overlapping the requested time range, stitches them, and warns about gaps.

```python
from casm_io.correlator import read_visibilities, load_format

fmt = load_format("layout_64ant")   # required for headerless files pre-2026-03-04
result = read_visibilities(
    time_start="2026-05-16 11:30:00",
    time_end="2026-05-16 14:30:00",
    time_tz="America/Los_Angeles",  # recommended for OVRO operations
    data_root="/mnt",               # scanned recursively for visibilities_* dirs
    fmt=fmt,
    freq_order="descending",        # recommended (CASM native; default)
    verbose=True,                   # recommended for interactive use
)
vis       = result.vis        # (T, F, n_bl) complex64
freq_mhz  = result.freq_mhz   # (F,) float64, descending
time_unix = result.time_unix  # (T,) float64 Unix timestamps
```

Variations:

```python
# Frequency range slice (post-load, no re-read needed)
mask = (freq_mhz >= 440) & (freq_mhz <= 465)
vis_sub = vis[:, mask, :]

# Or read only those channels from disk (faster for large obs)
result = read_visibilities(..., freq_range_mhz=(440, 465))

# Channel index slice (native descending order; exclusive end)
result = read_visibilities(..., channels=(631, 1443))

# All baselines toward a reference input
result = read_visibilities(..., ref=10, targets=[0, 1, 2, 5])

# Single (i, j) baseline from the full matrix
from casm_io.correlator.baselines import triu_flat_index
bl_idx = triu_flat_index(nsig, min(i, j), max(i, j))
v_pair = result.vis[:, :, bl_idx]   # (T, F) complex64
```

`channels` and `freq_range_mhz` are mutually exclusive. See [docs/correlator.md](docs/correlator.md) for full parameter reference.

## Antenna mapping

```python
from casm_io.correlator import AntennaMapping

# No path -> canonical layout: $CASM_LAYOUT_CSV, then $CASM_LAYOUT_DIR/current
ant = AntennaMapping.load()
# Or pin an explicit CSV:
ant = AntennaMapping.load("/path/to/antenna_layout.csv")
ant.packet_index(antenna_id=5)    # correlator input index
ant.snap_adc(antenna_id=5)        # (snap_id, adc)
ant.format_antenna(5)             # 'Ant 5 | S2A6 -> input 30'
ant.active_antennas()             # list of functional antenna IDs

# Mark antennas inactive without editing the CSV (returns new object)
ant_clean = ant.with_inactive([3, 7])
ant_subset = ant.with_only([1, 2, 5])

# Dense 72-slot table (6 SNAPs x 12 ADCs)
ant.slot_table()                  # DataFrame, 72 rows; -1 for unwired slots
ant.positions_64()                # (72, 3) ENU array, zeros for unwired
ant.active_mask_64()              # (72,) bool: wired & functional & beamforming
ant.antenna_ids_64()              # (72,) int: antenna_id or -1
```

See [docs/antenna_mapping.md](docs/antenna_mapping.md) for CSV schema details and the dual-schema auto-detection logic.

## Voltage DADA files

```python
from casm_io import VoltageReader

reader = VoltageReader("/mnt/nvme3/data/casm/voltage_dumps", "2026-02-17-21:10:43")
result = reader.read_full_band(antenna_csv="/path/to/antenna_layout.csv")
print(result.voltages.shape)   # (ntime, 3072, 16) complex64
```

See [docs/voltage.md](docs/voltage.md).

## Filterbank files

```python
from casm_io import FilterbankFile

fb = FilterbankFile("/path/to/beam.fil")
print(fb.nchans, fb.nsamples)
data = fb.data                 # (nsamples, nchans) float32, lazy load
```

See [docs/filterbank.md](docs/filterbank.md).

## Candidates

```python
from casm_io import CandidateReader

cands = CandidateReader("/path/to/t1_candidates.txt")
print(cands.n_candidates, cands.snr_range, cands.dm_range)
```

See [docs/candidates.md](docs/candidates.md).

## Constants

```python
from casm_io.constants import OVRO_LAT_DEG, OVRO_LON_DEG, OVRO_ELEV_M, C_LIGHT_M_S
```

## Testing

```bash
python -m pytest tests/ -v
```

## Documentation

- [Sign & ordering conventions](docs/CONVENTIONS.md) — shared contract for all CASM offline packages
- [Correlator visibilities](docs/correlator.md)
- [Antenna mapping](docs/antenna_mapping.md)
- [Voltage DADA files](docs/voltage.md)
- [Filterbank files](docs/filterbank.md)
- [Candidates](docs/candidates.md)
