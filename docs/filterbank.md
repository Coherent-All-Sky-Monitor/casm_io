# Filterbank Files

## Reading and writing

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

## Quick-look plots

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

# 2-panel FRB inspection: timeseries + waterfall
plot_dedispersed_waterfall(
    result['data'], result['header'],
    dm=500.0,
    output_path="frb_inspection.png",
)
```

## CLI script

A CLI script for filterbank inspection is provided in `examples/inspect_filterbank.py`:

```bash
# Dedispersed 2-panel plot with time range
python examples/inspect_filterbank.py \
    --input beam.fil --dm 125.0 --time-range 15.5 17.5

# Raw dynamic spectrum (no dedispersion)
python examples/inspect_filterbank.py \
    --input beam.fil --dm 125.0 --no-dedisperse

# Custom output path and dB scale
python examples/inspect_filterbank.py \
    --input beam.fil --dm 125.0 --output frb.png --scale db
```

Options:
- `--input` (required): path to `.fil` file
- `--dm` (float, default 0.0): dispersion measure in pc/cm^3
- `--time-range START END`: time window in seconds
- `--no-dedisperse`: show raw dynamic spectrum even when `--dm` is set
- `--output`: output PNG path (auto-generated from input filename if omitted)
- `--scale`: `linear` (default) or `db`

## Backend traceability

Both `read_filterbank` and `write_filterbank` return a `backend_used` field (`"sigpyproc"` or `"standalone"`). Use this to trace which code path ran if debugging issues.
