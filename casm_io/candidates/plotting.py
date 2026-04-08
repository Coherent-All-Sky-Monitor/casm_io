"""Candidate inspection plots."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from casm_io.filterbank.plotting import (
    _dm_delay_samples,
    _dedisperse,
)
from casm_io.filterbank.header import get_frequency_axis, get_time_axis


def plot_candidate(
    fb,
    candidate_index: int,
    reader,
    margin_factor: float = 2.0,
    output_path: str | None = None,
    figsize: tuple = (14, 12),
    cmap: str = "viridis",
    nsub: int = 64,
) -> plt.Figure:
    """
    Plot dedispersed waterfall + timeseries for a single candidate.

    Parameters
    ----------
    fb : FilterbankFile
        Loaded filterbank file.
    candidate_index : int
        Positional index in ``reader.df`` to plot (iloc-based).
        The original row number from the file is preserved via the
        DataFrame index and shown in the plot title.
    reader : CandidateReader
        Candidate list (sort however you want, don't reset_index).
    margin_factor : float
        Time window in seconds around the candidate (default 2.0).
    output_path : str, optional
        If provided, save figure to this path.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for waterfall.
    nsub : int
        Number of subbands for waterfall (default 64). Lower values
        average more channels, improving visibility of low-SNR pulses.

    Returns
    -------
    matplotlib.figure.Figure
    """
    row = reader.df.iloc[candidate_index]
    candidate_number = row.name
    dm = float(row['dm'])
    sample_idx = int(row['sample_index'])
    boxcar = int(row['boxcar_width'])
    boxcar_samples = max(1, 2**boxcar)  # boxcar index -> actual sample width
    snr = float(row['snr'])

    header = fb.header
    freqs = get_frequency_axis(header)
    tsamp = header.get("tsamp", 1.0)
    nsamples = fb.data.shape[0]

    # Dedisperse the FULL filterbank first, then slice.
    # This avoids edge artifacts at high DM where the sweep can be
    # comparable to the data slice length.
    dd_full = _dedisperse(fb.data, dm, freqs, tsamp, header=header)

    # Ensure frequency ascending for display
    if len(freqs) > 1 and freqs[0] > freqs[-1]:
        freqs = freqs[::-1]
        dd_full = dd_full[:, ::-1]

    # Display window: +/- margin_factor seconds around candidate (default 2s)
    margin = max(256, int(margin_factor / tsamp))

    s0 = max(0, sample_idx - margin)
    s1 = min(len(dd_full), sample_idx + margin)
    dd_slice = dd_full[s0:s1].astype(np.float32)

    times = np.arange(s0, s1) * tsamp
    t_cand = sample_idx * tsamp

    # Subband the raw dedispersed slice
    nchan = dd_slice.shape[1]
    nchan_per_sub = nchan // nsub
    nchan_use = nsub * nchan_per_sub
    wf_sub = dd_slice[:, :nchan_use].reshape(
        dd_slice.shape[0], nsub, nchan_per_sub).mean(axis=2)
    freqs_sub = freqs[:nchan_use].reshape(nsub, nchan_per_sub).mean(axis=1)

    # Per-subband normalization (robust: median + MAD)
    for i in range(nsub):
        med = np.median(wf_sub[:, i])
        mad = np.median(np.abs(wf_sub[:, i] - med))
        sigma = mad * 1.4826
        if sigma > 0:
            wf_sub[:, i] = (wf_sub[:, i] - med) / sigma

    # Timeseries from full-resolution normalized data, with boxcar smoothing
    ts_times = times.copy()
    ts = wf_sub.mean(axis=1)
    if boxcar_samples > 1:
        kernel = np.ones(boxcar_samples) / boxcar_samples
        ts = np.convolve(ts, kernel, mode="same")

    # Time-scrunch waterfall only (not timeseries) for visibility
    wf_times = times.copy()
    if boxcar_samples > 1:
        n_time_bins = wf_sub.shape[0] // boxcar_samples
        if n_time_bins > 0:
            wf_sub = wf_sub[:n_time_bins * boxcar_samples].reshape(
                n_time_bins, boxcar_samples, nsub).mean(axis=1)
            wf_times = wf_times[:n_time_bins * boxcar_samples].reshape(
                n_time_bins, boxcar_samples).mean(axis=1)

    # Waterfall color scale from robust stats
    vmin = np.percentile(wf_sub, 1)
    vmax = np.percentile(wf_sub, 99)

    # --- Plot ---
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.05)

    # Timeseries (top)
    ax_ts = fig.add_subplot(gs[0])
    ax_ts.plot(ts_times, ts, "k-", lw=0.5)
    ax_ts.axvline(t_cand, color='r', ls='--', lw=0.8, alpha=0.7)
    ax_ts.set_ylabel("Counts")
    ax_ts.set_title(
        f"Candidate #{candidate_number} | SNR={snr:.1f} | DM={dm:.2f} pc/cm\u00b3 | "
        f"width={boxcar} | sample={sample_idx}"
    )
    ax_ts.set_xlim(ts_times[0], ts_times[-1])
    ax_ts.tick_params(labelbottom=False)

    # Waterfall (bottom)
    ax_wf = fig.add_subplot(gs[1])
    ax_wf.imshow(
        wf_sub.T, aspect="auto", origin="lower",
        extent=[wf_times[0], wf_times[-1], freqs_sub[0], freqs_sub[-1]],
        cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax_wf.axvline(t_cand, color='r', ls='--', lw=0.8, alpha=0.7)
    ax_wf.set_xlabel("Time (s)")
    ax_wf.set_ylabel("Frequency (MHz)")

    fig.set_constrained_layout(True)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
