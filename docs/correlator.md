# Correlator Visibilities

## Reading visibilities

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
```

## Extracting specific baselines

```python
# Use antenna mapping to get correlator input indices
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

## Format configs

Two built-in formats ship with the package:
- `"pre_jan27_2026"` — 64 inputs (32 ant), 34.36s integrations
- `"post_jan27_2026"` — 128 inputs (64 ant), 137.44s integrations

Load by name or by path to a custom JSON:
```python
fmt = load_format("post_jan27_2026")       # built-in
fmt = load_format("/path/to/custom.json")  # custom
```

New correlator format (e.g., 256-antenna): add a JSON file to `casm_io/correlator/configs/` with `nsig`, `dt_raw_s`, `nchan`, frequency params. No code changes needed.

## Antenna mapping CSV

The antenna mapping CSV maps hardware (SNAP board, ADC channel) to antennas and correlator inputs. Required columns: `antenna_id`, `snap_id`, `adc`, `packet_index`. Optional: `grid_code`, `kernel_index`, `pol`, `x_m`, `y_m`, `z_m`, `functional`.

Legacy CSVs using `antenna` and `snap` column names are auto-detected.

When CAsMan is ready:
```bash
casman export-mapping --format casm_io > antenna_mapping_2027.csv
```

## Frequency order

All correlator readers default to `freq_order='descending'` (native/raw channel order, 468.75 -> 375 MHz). Pass `freq_order='ascending'` to get 375 -> 468.75 MHz.
