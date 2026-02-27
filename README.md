# casm_io

Unified reader and writer for all CASM (Coherent All-Sky Monitor) data products at OVRO.

Three data products:
- **Correlator visibilities** — binary `.dat` files (pre/post Jan 27 2026 formats)
- **Voltage DADA dumps** — 4+4 bit complex `.dada` files (3-subband, 4096-byte headers)
- **Filterbank files** — SIGPROC `.fil` format (read/write + quick-look plots)

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

## Correlator Visibilities

```python
from casm_io.correlator import read_visibilities, load_format
from casm_io.correlator.mapping import AntennaMapping

# Load format config (binary file layout) and antenna mapping (hardware)
fmt = load_format("post_jan27_2026")
ant = AntennaMapping.load("/path/to/antenna_layout_current.csv")

# Read all baselines
data = read_visibilities(
    "/data/casm/visibilities_64ant",
    "2026-01-27-20:38:33",
    fmt,
    freq_order='ascending',
)
print(data['vis'].shape)       # (T, 3072, 8256) complex64
print(data['freq_mhz'][:3])   # [375.000, 375.031, 375.061]

# Extract specific baselines using antenna mapping
ref_idx = ant.packet_index(antenna_id=4)   # -> 27
target_idxs = [ant.packet_index(a) for a in [5, 6, 7]]

data = read_visibilities(
    "/data/casm/visibilities_64ant",
    "2026-01-27-20:38:33",
    fmt,
    ref=ref_idx,
    targets=target_idxs,
    time_start="2026-01-28T04:00:00",
    time_end="2026-01-28T14:00:00",
    time_tz="America/Los_Angeles",
)
print(data['vis'].shape)       # (T, 3072, 3) complex64

# Inspect mapping
print(ant.format_antenna(5))   # "Ant 5 | S2A6 -> input 30"
print(ant.active_antennas())   # [1, 2, 3, ..., 16]
```

### Format configs

Two built-in formats ship with the package:
- `"pre_jan27_2026"` — 64 inputs (32 ant), 34.36s integrations
- `"post_jan27_2026"` — 128 inputs (64 ant), 137.44s integrations

Load by name or by path to a custom JSON:
```python
fmt = load_format("post_jan27_2026")       # built-in
fmt = load_format("/path/to/custom.json")  # custom
```

### Antenna mapping CSV

The antenna mapping CSV maps hardware (SNAP board, ADC channel) to antennas and correlator inputs. Required columns: `antenna_id`, `snap_id`, `adc`, `packet_index`. Optional: `grid_code`, `kernel_index`, `pol`, `x_m`, `y_m`, `z_m`, `functional`.

Legacy CSVs using `antenna` and `snap` column names are auto-detected.

When CAsMan is ready:
```bash
casman export-mapping --format casm_io > antenna_mapping_2027.csv
```

## Voltage DADA Files

```python
from casm_io.voltage import read_voltage_dump

# Read all 3 subbands, stitch to full band, extract per-antenna
result = read_voltage_dump(
    "/mnt/nvme3/data/casm/voltage_dumps",
    "2026-02-17-21:10:43",
    antenna_csv="/path/to/antenna_layout_current.csv",
    freq_order='ascending',
    trust_header=False,   # default: only trust verified header fields
    n_time=1000,
)
print(result['voltages'].shape)      # (1000, 3072, 16) complex64
print(result['freq_mhz'][[0, -1]])   # [375.000, 468.719]
print(result['header']['UTC_START']) # "2026-02-17-21:10:43"
```

Read a single subband file:
```python
from casm_io.voltage import read_dada_file

raw = read_dada_file(
    "/mnt/nvme3/data/casm/voltage_dumps/chan0_1023/2026-02-17-21:10:43_CHAN0_1023_0000000000000000.000000.dada",
    n_time=100,
)
print(raw['voltages'][0].shape)  # (100, 1024, 12) - SNAP 0 data
```

### Header trust

DADA headers contain fields that are sometimes wrong. By default `trust_header=False` only uses verified fields (`UTC_START`, `TSAMP`, `NCHAN`, etc.) and substitutes known-good defaults for unreliable fields (`NANT`, `FILE_SIZE`, `SOURCE`, etc.). Set `trust_header=True` to use all header values as-is.

## Filterbank Files

```python
from casm_io.filterbank import read_filterbank, write_filterbank

# Read
result = read_filterbank("/path/to/beam.fil")
print(result['data'].shape)        # (nsamples, nchans)
print(result['backend_used'])      # "sigpyproc" or "standalone"

# Write
info = write_filterbank("output.fil", result['data'], result['header'], nbits=8)
print(info['backend_used'])        # "sigpyproc" or "standalone"
```

### Quick-look plots

```python
from casm_io.filterbank.plotting import (
    plot_bandpass,
    plot_timeseries,
    plot_dynamic_spectrum,
    plot_dedispersed_waterfall,
)

plot_bandpass(result['data'], result['header'], scale='db', output_path="bandpass.png")

plot_timeseries(result['data'], result['header'], output_path="timeseries.png")

plot_dynamic_spectrum(
    result['data'], result['header'],
    dm=500.0,
    time_range=(1.2, 1.8),
    output_path="frb_candidate.png",
)

# 3-panel FRB inspection: waterfall + timeseries + bandpass
plot_dedispersed_waterfall(
    result['data'], result['header'],
    dm=500.0,
    output_path="frb_inspection.png",
)
```

### Backend traceability

Both `read_filterbank` and `write_filterbank` return a `backend_used` field (`"sigpyproc"` or `"standalone"`). Use this to trace which code path ran if debugging issues.

## Frequency order

All readers default to `freq_order='descending'` (native/raw channel order, 468.75 -> 375 MHz). Pass `freq_order='ascending'` to get 375 -> 468.75 MHz.

## Project structure

```
casm_io/
    __init__.py          # version + convenience re-exports
    constants.py         # OVRO location, frequency band, timing
    correlator/
        configs/         # pre_jan27_2026.json, post_jan27_2026.json
        formats.py       # VisibilityFormat, load_format()
        mapping.py       # AntennaMapping (from CSV)
        baselines.py     # upper-triangular indexing utilities
        reader.py        # read_visibilities(), discover_files()
        writer.py        # write_npz(), read_npz()
    voltage/
        configs/         # dada_format.json
        header.py        # parse_dada_header(), trusted/untrusted fields
        unpack.py        # unpack_4bit() (4+4 bit complex)
        reader.py        # read_dada_file(), read_voltage_dump()
    filterbank/
        header.py        # SIGPROC header parser/writer
        reader.py        # read_filterbank() (sigpyproc + standalone)
        writer.py        # write_filterbank() (sigpyproc + standalone)
        plotting.py      # bandpass, timeseries, dynamic spectrum, FRB waterfall
```

## Adding new hardware configs

**New correlator format** (e.g., 256-antenna): Add a JSON file to `casm_io/correlator/configs/` with `nsig`, `dt_raw_s`, `nchan`, frequency params. Load with `load_format("my_new_format")` or `load_format("/path/to/file.json")`.

**New antenna mapping**: Create a CSV with required columns (`antenna_id`, `snap_id`, `adc`, `packet_index`). No code changes needed. Works for 16, 64, or 256 antennas.

