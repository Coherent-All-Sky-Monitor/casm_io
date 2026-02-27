# casm_io

Unified reader and writer for all CASM (Coherent All-Sky Monitor) data products at OVRO.

- **Correlator visibilities** — binary `.dat` files (pre/post Jan 27 2026 formats)
- **Voltage DADA dumps** — 4+4 bit complex `.dada` files (3-subband, 4096-byte headers)
- **Filterbank files** — SIGPROC `.fil` format (read/write + quick-look plots)
- **Candidates** — FRB search candidate lists (Hella T1/T2/T3)

## Install

```bash
source ~/software/dev/casm_venvs/casm_offline_env/bin/activate
cd /home/casm/software/dev/casm_io
pip install -e .
```

Verify:
```bash
python -c "import casm_io; print(casm_io.__version__)"
```

## Quick Start

### Correlator visibilities

```python
from casm_io.correlator import read_visibilities, load_format

fmt = load_format("post_jan27_2026")
data = read_visibilities("/data/casm/visibilities_64ant", "2026-01-27-20:38:33", fmt)
print(data['vis'].shape)       # (T, 3072, 8256) complex64
print(data['freq_mhz'][:3])   # [375.000, 375.031, 375.061]
```

### Voltage DADA files

```python
from casm_io.voltage import read_voltage_dump

result = read_voltage_dump(
    "/mnt/nvme3/data/casm/voltage_dumps",
    "2026-02-17-21:10:43",
    antenna_csv="/path/to/antenna_layout_current.csv",
)
print(result['voltages'].shape)      # (ntime, 3072, 16) complex64
print(result['freq_mhz'][[0, -1]])   # [468.719, 375.000]
```

### Filterbank files

```python
from casm_io.filterbank import read_filterbank

result = read_filterbank("/path/to/beam.fil")
print(result['data'].shape)        # (nsamples, nchans)
print(result['backend_used'])      # "sigpyproc" or "standalone"
```

### Candidates

```python
from casm_io.candidates import read_t1_candidates

df = read_t1_candidates("/path/to/t1_candidates.txt")
print(df.shape)
```

## Documentation

- [Correlator visibilities](docs/correlator.md) — format configs, antenna mapping, baseline extraction, frequency order
- [Voltage DADA files](docs/voltage.md) — single-subband reading, header trust, frequency order
- [Filterbank files](docs/filterbank.md) — plotting functions, CLI script, backend traceability
- [Candidates](docs/candidates.md) — T1 column table, usage

## Testing

```bash
python -m pytest tests/ -v
find casm_io -name '*.py' -exec python -m py_compile {} \;
python -c "import casm_io; print(casm_io.__version__)"
```
