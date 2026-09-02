# Filterbank Files

## Reading

```python
from casm_io import FilterbankFile

fb = FilterbankFile("/path/to/beam.fil")
# Prints: Opened beam.fil (sigpyproc): 3072 chans, 50000 samples

# Header is parsed on init — data is NOT loaded yet
print(fb.nchans, fb.nsamples)
print(fb.freq_mhz[:3])
print(fb.time_s[-1])
print(fb.backend_used)    # "sigpyproc" or "standalone"

# Data loads lazily on first access
data = fb.data             # (nsamples, nchans)
# Prints: Loading filterbank data...
#         Loaded: (50000, 3072) float32
```

Silence output with `verbose=False`:

```python
fb = FilterbankFile("/path/to/beam.fil", verbose=False)
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

## Splitting a time range

```python
from casm_io.filterbank import split_filterbank

result = split_filterbank(
    "/path/to/beam.fil",
    "/path/to/beam_split.fil",
    start_sample=50,      # first sample to extract (0-indexed), default 0
    nsamples=80,          # None reads to end of file
    beam=None,            # beam index for a multibeam file; defaults to 0
    verbose=True,
)
print(result["nsamples_written"], result["header"], result["filepath"])
```

Reads `[start_sample, start_sample + nsamples)` via seek + `fromfile`, so the whole file is never loaded into memory. `tstart` in the output header is shifted forward by `start_sample * tsamp / 86400` (days) and `nsamples`/`nbeams` are updated to match. `start_sample`/`nsamples` out of range raise `ValueError`. Sub-byte filterbanks (`nbits < 8`) are not supported.

## Requantizing bit depth

```python
from casm_io.filterbank import requantize_filterbank

result = requantize_filterbank(
    "/path/to/beam_32bit.fil",
    "/path/to/beam_8bit.fil",
    target_nbits=8,       # 8 or 16
    sigma_clip=4.0,       # std devs mapped to half the output range
    beam=None,            # beam index for a multibeam file
    verbose=True,
)
print(result["backend_used"], result["per_channel_stats"])
```

Converts between bit depths (e.g. 32-bit float to 8-bit unsigned int) with a per-channel linear mapping: each channel's median maps to the output mid-range value (127.5 for 8-bit) and `+/- sigma_clip` standard deviations span the full output range. This preserves the bandpass shape, since channels with higher power keep higher mean values in the output. Channels with zero standard deviation ("dead" channels) are written as all zeros. `per_channel_stats` returns `median`, `std`, and `scale` arrays per channel. The on-disk header is rebuilt from the raw SIGPROC header rather than sigpyproc's parsed header, to avoid derived keys (`accel`, `bandwidth`, `fbottom`, `ftop`, `obs_time`, `period`) that break round-tripping.

## Backend traceability

Both `FilterbankFile` and `write_filterbank` expose `backend_used` (`"sigpyproc"` or `"standalone"`). Check this when debugging read/write issues.
