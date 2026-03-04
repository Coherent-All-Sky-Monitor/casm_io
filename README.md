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
pip install -e .
```

## Quick Start

### Correlator visibilities

```python
from casm_io import VisibilityReader, load_format

fmt = load_format("layout_64ant")
reader = VisibilityReader("/data/casm/visibilities_64ant", "2026-01-27-20:38:33", fmt)

# Check time span
print(reader.time_span_str())                        # UTC
print(reader.time_span_str("America/Los_Angeles"))   # Pacific

# Read first 5 files
result = reader.read(nfiles=5)
print(result.vis.shape)          # (T, 3072, 8256) complex64
print(result.freq_mhz[:3])      # [468.750, 468.719, 468.689]

# Skip files
result = reader.read(nfiles=5, skip_nfiles=10)

# Both access styles work
result.vis                       # attribute access (new)
result['vis']                    # dict-style access (backwards compatible)
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

### Candidates

```python
from casm_io import CandidateReader

cands = CandidateReader("/path/to/t1_candidates.txt")
print(cands.n_candidates, cands.snr_range, cands.dm_range)
print(cands.df.head())
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
