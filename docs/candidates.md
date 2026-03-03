# Candidates

Read FRB search candidate lists produced by Hella.

## Usage

```python
from casm_io import CandidateReader

cands = CandidateReader("/path/to/t1_candidates.txt")
print(cands.n_candidates)   # 1234
print(cands.snr_range)      # (6.5, 42.3)
print(cands.dm_range)       # (10.0, 3000.0)
print(cands.df.head())
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
