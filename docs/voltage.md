# Voltage DADA Files

## Reading a full voltage dump

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

## Reading a single subband

```python
from casm_io.voltage import read_dada_file

raw = read_dada_file(
    "/mnt/nvme3/data/casm/voltage_dumps/chan0_1023/2026-02-17-21:10:43_CHAN0_1023_0000000000000000.000000.dada",
    n_time=100,
)
print(raw['voltages'][0].shape)  # (100, 1024, 12) - SNAP 0 data
```

## Header trust

DADA headers contain fields that are sometimes wrong. By default `trust_header=False` only uses verified fields and substitutes known-good defaults for unreliable fields. Set `trust_header=True` to use all header values as-is.

**Trusted fields**: `UTC_START`, `TSAMP`, `NCHAN`, `NBIT`, `NDIM`, `ENCODING`, `RESOLUTION`, `HDR_SIZE`, `BW`, `UDP_NANT`, `STREAM_SUBBAND_ID`, `FREQ`

**Untrusted fields**: `NANT`, `FILE_SIZE`, `SOURCE`, `NPOL`, `PICOSECONDS`, `START_CHANNEL`, `END_CHANNEL`

## Frequency order

Voltage readers default to `freq_order='descending'` (native 468.75 -> 375 MHz). Pass `freq_order='ascending'` to get 375 -> 468.75 MHz.
