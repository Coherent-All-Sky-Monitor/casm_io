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

On the correlator nodes the lists are written to
`/mnt/nvme4/data/casm/hella_cands/cands_<UTC_START>.dat.<n>`, with `<n>` running
0-3 for each observation.

## T1 columns

Columns are renamed from the Hella header row (`SNR SAMP_START TIME_START WIDTH DM_IDX DM BEAM_IDX`):

| Column | Type | Description |
|--------|------|-------------|
| `snr` | float | Signal-to-noise ratio (`SNR`) |
| `sample_index` | int | Filter output sample index (`SAMP_START`) |
| `time_start` | float | Candidate start time (`TIME_START`) |
| `boxcar_width` | int | Boxcar filter width index (`WIDTH`) |
| `dm_index` | int | DM trial index (`DM_IDX`) |
| `dm` | float | Dispersion measure, pc/cm^3 (`DM`) |
| `beam_index` | int | Beam index (`BEAM_IDX`) |
