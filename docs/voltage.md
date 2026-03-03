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
print(result['voltages'].shape)      # (1000, 3072, 16) complex64
print(result['freq_mhz'][[0, -1]])   # [375.000, 468.719]
```

## Reading a single subband

```python
result = reader.read_subband(0, n_time=100)
print(result['voltages'][0].shape)   # (100, 1024, 12) — SNAP 0 data
print(result['freq_mhz'].shape)      # (1024,)
```

## Header trust

Default `trust_header=False` substitutes known-good defaults for unreliable header fields.

**Trusted**: `UTC_START`, `TSAMP`, `NCHAN`, `NBIT`, `NDIM`, `ENCODING`, `RESOLUTION`, `HDR_SIZE`, `BW`, `UDP_NANT`, `STREAM_SUBBAND_ID`, `FREQ`

**Untrusted**: `NANT`, `FILE_SIZE`, `SOURCE`, `NPOL`, `PICOSECONDS`, `START_CHANNEL`, `END_CHANNEL`

## Frequency order

Default `freq_order='descending'` (native 468.75 -> 375 MHz). Pass `'ascending'` to reverse.
