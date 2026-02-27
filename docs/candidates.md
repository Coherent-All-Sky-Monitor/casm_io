# Candidates

Read FRB search candidate lists produced by Hella.

## Usage

```python
from casm_io.candidates import read_t1_candidates

df = read_t1_candidates("/path/to/t1_candidates.txt")
print(df.head())
print(df.shape)
```

## T1 columns

| Column | Type | Description |
|--------|------|-------------|
| `snr` | float | Signal-to-noise ratio |
| `sample_index` | int | Filter output sample index |
| `integration_index` | int | Integration time index |
| `mjd` | float | MJD timestamp |
| `boxcar_width` | int | Boxcar filter width index |
| `dm_index` | int | DM trial index |
| `dm` | float | Dispersion measure (pc/cm^3) |
| `beam_index` | int | Beam index |
