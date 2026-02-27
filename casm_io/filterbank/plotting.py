"""
Filterbank quick-look plotting utilities.

These are data inspection tools, not scientific analysis.
All plots use the Agg backend, return fig objects, and optionally save to file.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .header import get_frequency_axis, get_time_axis


def _dm_delay_samples(dm: float, freq_mhz: np.ndarray, tsamp: float) -> np.ndarray:
    """
    Compute per-channel DM delay in samples.

    Delay = 4.148808 * DM * (1/f_lo^2 - 1/f^2) / tsamp
    Reference: lowest frequency channel.
    """
    k_dm = 4.148808e3  # MHz^2 s pc^-1 cm^3
    f_ref = np.min(freq_mhz)
    delays_s = k_dm * dm * (1.0 / f_ref**2 - 1.0 / freq_mhz**2)
    return np.round(delays_s / tsamp).astype(int)


def _dedisperse_sigpyproc(data: np.ndarray, dm: float, header: dict) -> np.ndarray:
    """
    Dedisperse using sigpyproc (circular shift, preserves length).

    Parameters
    ----------
    data : np.ndarray
        (nsamples, nchans) array.
    dm : float
        Dispersion measure in pc/cm^3.
    header : dict
        Filterbank header (needs fch1, foff, nchans, nbits, tsamp, tstart).

    Returns
    -------
    np.ndarray
        Dedispersed (nsamples, nchans) array (same length as input).
    """
    from sigpyproc.header import Header
    from sigpyproc.block import FilterbankBlock

    hdr = Header(
        filename="",
        data_type="filterbank",
        nchans=header.get("nchans", data.shape[1]),
        foff=header.get("foff", -1.0),
        fch1=header.get("fch1", 0.0),
        nbits=header.get("nbits", 32),
        tsamp=header.get("tsamp", 1.0),
        tstart=header.get("tstart", 60000.0),
        nsamples=data.shape[0],
    )
    # sigpyproc expects (nchans, nsamples)
    block = FilterbankBlock(data.T.astype(np.float32), hdr)
    dd_block = block.dedisperse(dm)
    return dd_block.data.T


def _dedisperse_standalone(data: np.ndarray, dm: float, freq_mhz: np.ndarray, tsamp: float) -> np.ndarray:
    """
    Dedisperse by shifting channels and trimming edges.

    Parameters
    ----------
    data : np.ndarray
        (nsamples, nchans) array.
    dm : float
        Dispersion measure in pc/cm^3.
    freq_mhz : np.ndarray
        Frequency axis in MHz.
    tsamp : float
        Sample time in seconds.

    Returns
    -------
    np.ndarray
        Dedispersed (nsamples_trimmed, nchans) array.
    """
    delays = _dm_delay_samples(dm, freq_mhz, tsamp)
    max_delay = np.max(np.abs(delays))
    nsamples = data.shape[0]
    out_len = nsamples - max_delay

    if out_len <= 0:
        raise ValueError(
            f"DM={dm} requires {max_delay} samples delay but data has "
            f"only {nsamples} samples"
        )

    out = np.zeros((out_len, len(freq_mhz)), dtype=data.dtype)
    for i, d in enumerate(delays):
        shift = max_delay - d
        out[:, i] = data[shift : shift + out_len, i]
    return out


def _dedisperse(data: np.ndarray, dm: float, freq_mhz: np.ndarray, tsamp: float,
                header: dict | None = None) -> np.ndarray:
    """
    Dedisperse data by shifting channels.

    Tries sigpyproc first (circular shift, preserves length).
    Falls back to standalone (trim edges) if sigpyproc unavailable.

    Parameters
    ----------
    data : np.ndarray
        (nsamples, nchans) array.
    dm : float
        Dispersion measure in pc/cm^3.
    freq_mhz : np.ndarray
        Frequency axis in MHz.
    tsamp : float
        Sample time in seconds.
    header : dict, optional
        Filterbank header. Required for sigpyproc path.

    Returns
    -------
    np.ndarray
        Dedispersed array. Same length if sigpyproc, trimmed if standalone.
    """
    if header is not None:
        try:
            return _dedisperse_sigpyproc(data, dm, header)
        except (ImportError, Exception):
            pass
    return _dedisperse_standalone(data, dm, freq_mhz, tsamp)


def _time_range_to_slice(
    time_range: tuple, header: dict, nsamples: int,
) -> tuple[int, int]:
    """Convert time_range (seconds or MJD) to sample indices."""
    t0, t1 = time_range
    tsamp = header.get("tsamp", 1.0)

    # If values are large (> 40000), assume MJD
    if t0 > 40000:
        tstart_mjd = header.get("tstart", 0.0)
        t0_s = (t0 - tstart_mjd) * 86400.0
        t1_s = (t1 - tstart_mjd) * 86400.0
    else:
        t0_s, t1_s = t0, t1

    s0 = max(0, int(t0_s / tsamp))
    s1 = min(nsamples, int(t1_s / tsamp))
    return s0, s1


def plot_bandpass(
    data: np.ndarray,
    header: dict,
    scale: str = "linear",
    output_path: str | None = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot time-averaged spectrum (bandpass).

    Parameters
    ----------
    data : np.ndarray
        (nsamples, nchans) data.
    header : dict
        Filterbank header.
    scale : str
        'linear' or 'db'.
    output_path : str, optional
        Save path.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    freqs = get_frequency_axis(header)
    bandpass = data.astype(np.float32).mean(axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    if scale.lower() == "db":
        bandpass = 10 * np.log10(np.maximum(bandpass, 1e-10))
        ylabel = "Power (dB)"
    else:
        ylabel = "Mean Counts"

    ax.plot(freqs, bandpass, "b-", lw=0.8)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Bandpass ({scale.capitalize()})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_timeseries(
    data: np.ndarray,
    header: dict,
    scale: str = "linear",
    output_path: str | None = None,
    figsize: tuple = (14, 6),
    max_points: int = 2000,
) -> plt.Figure:
    """
    Plot band-averaged power vs time.

    Parameters
    ----------
    data : np.ndarray
        (nsamples, nchans) data.
    header : dict
        Filterbank header.
    scale : str
        'linear' or 'db'.
    output_path : str, optional
        Save path.
    figsize : tuple
        Figure size.
    max_points : int
        Max points to plot (downsamples for speed).

    Returns
    -------
    matplotlib.figure.Figure
    """
    times = get_time_axis(header, len(data))
    ts = data.astype(np.float32).mean(axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    if scale.lower() == "db":
        ts = 10 * np.log10(np.maximum(ts, 1e-10))
        ylabel = "Power (dB)"
    else:
        ylabel = "Mean Counts"

    ds = max(1, len(ts) // max_points)
    ax.plot(times[::ds], ts[::ds], "b-", lw=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Band-Averaged Power ({scale.capitalize()})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_dynamic_spectrum(
    data: np.ndarray,
    header: dict,
    scale: str = "linear",
    dm: float | None = None,
    time_range: tuple | None = None,
    output_path: str | None = None,
    figsize: tuple = (14, 8),
    cmap: str = "viridis",
    vmin_pct: float = 1,
    vmax_pct: float = 99,
    max_time_bins: int = 2000,
    max_freq_bins: int = 1000,
) -> plt.Figure:
    """
    Plot dynamic spectrum (waterfall).

    Parameters
    ----------
    data : np.ndarray
        (nsamples, nchans) data.
    header : dict
        Filterbank header.
    scale : str
        'linear' or 'db'.
    dm : float, optional
        If provided, dedisperse before plotting.
    time_range : tuple, optional
        (start, end) in seconds or MJD. Slices before plotting.
    output_path : str, optional
        Save path.
    figsize, cmap, vmin_pct, vmax_pct, max_time_bins, max_freq_bins
        Plot parameters.

    Returns
    -------
    matplotlib.figure.Figure
    """
    freqs = get_frequency_axis(header)
    tsamp = header.get("tsamp", 1.0)
    plot_data = data

    # Time slicing
    if time_range is not None:
        s0, s1 = _time_range_to_slice(time_range, header, len(data))
        plot_data = plot_data[s0:s1]

    # Dedispersion
    if dm is not None and dm > 0:
        plot_data = _dedisperse(plot_data, dm, freqs, tsamp, header=header)

    times = get_time_axis(header, len(plot_data))
    if time_range is not None and time_range[0] <= 40000:
        times = times + time_range[0]

    # Ensure frequency is ascending (low freq at bottom, high at top)
    if len(freqs) > 1 and freqs[0] > freqs[-1]:
        freqs = freqs[::-1]
        plot_data = plot_data[:, ::-1]

    # Downsample for plotting
    t_factor = max(1, len(plot_data) // max_time_bins)
    f_factor = max(1, len(freqs) // max_freq_bins)
    data_ds = plot_data[::t_factor, ::f_factor].astype(np.float32)
    times_ds = times[::t_factor]
    freqs_ds = freqs[::f_factor]

    fig, ax = plt.subplots(figsize=figsize)
    if scale.lower() == "db":
        data_ds = 10 * np.log10(np.maximum(data_ds, 1e-10))
        label = "Power (dB)"
    else:
        label = "Counts"

    vmin = np.percentile(data_ds, vmin_pct)
    vmax = np.percentile(data_ds, vmax_pct)

    im = ax.imshow(
        data_ds.T, aspect="auto", origin="lower",
        extent=[times_ds[0], times_ds[-1], freqs_ds[0], freqs_ds[-1]],
        cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (MHz)")
    title = f"Dynamic Spectrum ({scale.capitalize()})"
    if dm is not None:
        title += f" | DM={dm:.1f}"
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=label)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_dedispersed_waterfall(
    data: np.ndarray,
    header: dict,
    dm: float,
    time_range: tuple | None = None,
    output_path: str | None = None,
    figsize: tuple = (14, 12),
    cmap: str = "viridis",
) -> plt.Figure:
    """
    3-panel FRB inspection plot: waterfall + timeseries + bandpass.

    Parameters
    ----------
    data : np.ndarray
        (nsamples, nchans) data.
    header : dict
        Filterbank header.
    dm : float
        Dispersion measure for dedispersion.
    time_range : tuple, optional
        (start, end) in seconds or MJD.
    output_path : str, optional
        Save path.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for waterfall.

    Returns
    -------
    matplotlib.figure.Figure
    """
    freqs = get_frequency_axis(header)
    tsamp = header.get("tsamp", 1.0)
    plot_data = data

    if time_range is not None:
        s0, s1 = _time_range_to_slice(time_range, header, len(data))
        plot_data = plot_data[s0:s1]

    dd_data = _dedisperse(plot_data, dm, freqs, tsamp, header=header)
    times = get_time_axis(header, len(dd_data))
    if time_range is not None and time_range[0] <= 40000:
        times = times + time_range[0]

    # Ensure frequency is ascending (low freq at bottom, high at top)
    if len(freqs) > 1 and freqs[0] > freqs[-1]:
        freqs = freqs[::-1]
        dd_data = dd_data[:, ::-1]

    dd_float = dd_data.astype(np.float32)
    ts = dd_float.mean(axis=1)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.05)

    # Timeseries (top)
    ax_ts = fig.add_subplot(gs[0])
    ds = max(1, len(ts) // 2000)
    ax_ts.plot(times[::ds], ts[::ds], "k-", lw=0.5)
    ax_ts.set_ylabel("Counts")
    ax_ts.set_title(f"DM = {dm:.2f} pc/cm\u00b3")
    ax_ts.set_xlim(times[0], times[-1])
    ax_ts.tick_params(labelbottom=False)

    # Waterfall (bottom)
    ax_wf = fig.add_subplot(gs[1])
    t_factor = max(1, len(dd_data) // 2000)
    f_factor = max(1, len(freqs) // 1000)
    wf = dd_float[::t_factor, ::f_factor]
    freqs_ds = freqs[::f_factor]
    vmin, vmax = np.percentile(wf, [1, 99])
    ax_wf.imshow(
        wf.T, aspect="auto", origin="lower",
        extent=[times[0], times[-1], freqs_ds[0], freqs_ds[-1]],
        cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax_wf.set_xlabel("Time (s)")
    ax_wf.set_ylabel("Frequency (MHz)")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
