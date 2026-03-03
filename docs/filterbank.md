# Filterbank Files

## Reading

```python
from casm_io import FilterbankFile

fb = FilterbankFile("/path/to/beam.fil")

# Header is parsed on init — data is NOT loaded yet
print(fb.nchans, fb.nsamples)
print(fb.freq_mhz[:3])
print(fb.time_s[-1])
print(fb.backend_used)    # "sigpyproc" or "standalone"

# Data loads on first access
data = fb.data             # (nsamples, nchans)
```

## Writing

```python
from casm_io import write_filterbank

info = write_filterbank("output.fil", data, fb.header, nbits=8)
print(info['backend_used'])
```

## Quick-look plots

```python
from casm_io.filterbank.plotting import (
    plot_bandpass,
    plot_timeseries,
    plot_dynamic_spectrum,
    plot_dedispersed_waterfall,
)

fb = FilterbankFile("beam.fil")

plot_bandpass(fb.data, fb.header, scale='db', output_path="bandpass.png")
plot_timeseries(fb.data, fb.header, output_path="timeseries.png")

plot_dynamic_spectrum(fb.data, fb.header, dm=500.0, time_range=(1.2, 1.8),
                      output_path="waterfall.png")

# 2-panel FRB inspection: timeseries + waterfall
plot_dedispersed_waterfall(fb.data, fb.header, dm=500.0,
                           output_path="frb_inspection.png")
```

## Backend traceability

Both `FilterbankFile` and `write_filterbank` expose `backend_used` (`"sigpyproc"` or `"standalone"`). Check this when debugging read/write issues.
