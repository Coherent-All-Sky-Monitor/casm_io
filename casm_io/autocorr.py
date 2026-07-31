"""Compare voltage-dump and correlator autocorrelations per SNAP and ADC.

Data inspection tool, not scientific analysis: reads a triggered voltage
dump in gulps, accumulates the mean power per (snap, adc, channel), reads
the correlator visibilities covering the dump window, and over-plots the
two autocorrelation bandpasses — one figure per SNAP, one panel per ADC.

The two are not expected to be identical: the voltage dump is a few
seconds, the correlator integration is ~137 s, and the absolute scales
differ (4-bit voltage units vs correlator counts), so each curve is
normalised by its own median. The bandpass shapes should agree.

Run as ``casm-autocorr``; see ``--help``.
"""

import argparse
import os
from datetime import datetime, timezone

import numpy as np

from .voltage.reader import (VoltageReader, default_gulp_samples,
                             mem_available_bytes)
from .correlator.reader import read_visibilities
from .correlator.baselines import triu_flat_index
from .correlator.mapping import AntennaMapping
from ._progress import print_progress
from ._time import format_time_span, unix_to_datetime

TSAMP_S = 32.768e-6  # voltage sample time
_LOCAL_TZ = "America/Los_Angeles"


def psrdada_to_unix(t: str) -> float | None:
    """Unix time from a PSRDADA UTC string like 2026-07-30-22:25:22.958."""
    for fmt in ("%Y-%m-%d-%H:%M:%S.%f", "%Y-%m-%d-%H:%M:%S"):
        try:
            dt = datetime.strptime(t, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, TypeError):
            continue
    return None


def voltage_autocorr(
    reader, n_time=None, gulp_samples=30518, snaps=None, subbands=None,
    verbose=True,
):
    """Mean |v|^2 per SNAP over time, read in gulps to bound memory.

    Returns (spec, freq_mhz, header, n_used) where spec is
    {snap: (n_chan, n_adc) float64} and n_used the samples averaged.
    """
    acc = {}
    n_used = 0
    freq_mhz = None
    header = None
    while n_time is None or n_used < n_time:
        gulp = gulp_samples
        if n_time is not None:
            gulp = min(gulp, n_time - n_used)
        res = reader.read_full_band(
            n_time=gulp, time_offset=n_used, snaps=snaps, subbands=subbands,
            verbose=verbose and n_used == 0,
        )
        nt = next(iter(res.voltages.values())).shape[0]
        if nt == 0:
            break
        if freq_mhz is None:
            freq_mhz = res.freq_mhz
            header = res.header
            dump_start = psrdada_to_unix(
                header.get("DUMP_UTC_START", header.get("UTC_START", ""))
            )
            if verbose and dump_start is not None and n_time is not None:
                span = n_time * TSAMP_S
                print(f"Reading (UTC): "
                      f"{format_time_span(dump_start, dump_start + span)}")
                print(f"Reading (PT):  "
                      f"{format_time_span(dump_start, dump_start + span, _LOCAL_TZ)}")
        for s, v in res.voltages.items():
            if s not in acc:
                acc[s] = np.zeros(v.shape[1:], dtype=np.float64)
            acc[s] += np.sum(v.real ** 2 + v.imag ** 2, axis=0)
        n_used += nt
        if verbose:
            suffix = ""
            if dump_start is not None:
                now = unix_to_datetime(
                    dump_start + n_used * TSAMP_S, _LOCAL_TZ,
                )
                suffix = f"at {now.strftime('%H:%M:%S.%f')[:-3]} PT"
            print_progress(n_used, n_time or n_used, prefix="Voltage autocorr",
                           suffix=suffix)
        if nt < gulp:
            break

    if verbose and n_time is not None and 0 < n_used < n_time:
        print()  # close the progress bar line after a short dump
    if n_used == 0:
        raise RuntimeError("No voltage samples read")
    spec = {s: a / n_used for s, a in acc.items()}
    return spec, freq_mhz, header, n_used


def autos_from_vis(vis, packet_indices):
    """Time-averaged autocorrelations from an upper-triangular vis array.

    vis has shape (n_time, n_chan, n_baselines) with the baseline axis
    flattened upper-triangular including the diagonal; packet_indices are
    correlator input indices. Returns {packet_index: (n_chan,) float64}.
    """
    n_bl = vis.shape[2]
    nsig = int(round((np.sqrt(8 * n_bl + 1) - 1) / 2))
    if nsig * (nsig + 1) // 2 != n_bl:
        raise ValueError(f"{n_bl} baselines is not triangular")
    autos = {}
    for p in packet_indices:
        if not 0 <= p < nsig:
            raise ValueError(f"packet_index {p} outside 0-{nsig - 1}")
        autos[p] = vis[:, :, triu_flat_index(nsig, p, p)].real.mean(axis=0)
    return autos


def _db_rel_median(curve):
    """Bandpass in dB relative to its median, so shapes compare across
    instruments with wildly different absolute scales (and one strong RFI
    line cannot flatten the rest of the band)."""
    nonzero = curve[curve > 0]
    if nonzero.size == 0:
        return np.zeros_like(curve)
    floor = 1e-6 * np.median(nonzero)
    return 10 * np.log10(np.maximum(curve, floor) / np.median(nonzero))


def plot_snap_autocorrs(
    spec, freq_v, vis_autos, freq_vis, mapping_df, out_dir,
    voltage_label="voltage", vis_label="visibility",
):
    """One figure per SNAP: 12 ADC panels, both bandpasses over-plotted.

    Returns the list of files written.
    """
    import matplotlib.pyplot as plt

    # (snap, adc) -> (antenna_id, packet_index) for the panel titles
    wired = {
        (int(r["snap_id"]), int(r["adc"])): (int(r["antenna_id"]),
                                             int(r["packet_index"]))
        for _, r in mapping_df.iterrows()
    }

    written = []
    for snap in sorted(spec):
        n_adc = spec[snap].shape[1]
        fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
        for adc in range(n_adc):
            ax = axes.flat[adc]
            ax.plot(freq_v, _db_rel_median(spec[snap][:, adc]),
                    lw=0.6, color="C0", label=voltage_label)
            title = f"ADC {adc}"
            if (snap, adc) in wired:
                ant, packet = wired[(snap, adc)]
                title += f"  ant {ant}"
                if packet in vis_autos:
                    ax.plot(freq_vis, _db_rel_median(vis_autos[packet]),
                            lw=0.6, color="C1", alpha=0.7, label=vis_label)
            else:
                title += "  (not in CSV)"
            ax.set_title(title, fontsize=9)
            if adc == 0:
                ax.legend(fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("Frequency (MHz)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Power (dB rel. median)")
        fig.suptitle(f"SNAP {snap} autocorrelations")
        fig.tight_layout()
        path = os.path.join(out_dir, f"autocorr_snap{snap}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("data_dir",
                        help="voltage dump directory with stream_N folders")
    parser.add_argument("timestamp",
                        help="dump filename prefix, e.g. "
                             "2026-07-29-21:06:34_0187972821983232")
    parser.add_argument("--vis-root", required=True,
                        help="directory scanned for visibilities_* folders")
    parser.add_argument("--csv", default=None,
                        help="antenna layout CSV (default: $CASM_LAYOUT_CSV "
                             "or the layout dir 'current' symlink)")
    parser.add_argument("--seconds", type=float, default=2.0,
                        help="voltage seconds to average")
    parser.add_argument("--gulp", type=float, default=None,
                        help="voltage seconds per read (default: sized from "
                             "the RAM currently available)")
    parser.add_argument("--streams", default=None,
                        help="comma-separated sub-band indices, e.g. 0,1")
    parser.add_argument("--snaps", default=None,
                        help="comma-separated SNAP slots to read "
                             "(default: the SNAPs wired in the CSV)")
    parser.add_argument("--out-dir", default=".",
                        help="directory for the plots")
    args = parser.parse_args()

    subbands = [int(x) for x in args.streams.split(",")] if args.streams else None
    snaps = [int(x) for x in args.snaps.split(",")] if args.snaps else None

    mapping = AntennaMapping.load(args.csv)
    if snaps is None:
        snaps = sorted(mapping.dataframe["snap_id"].unique().tolist())
        print(f"SNAPs from CSV: {snaps}")

    # Voltage side
    reader = VoltageReader(args.data_dir, args.timestamp)
    print(f"streams found: {reader.subbands_found}")

    cfg = reader._cfg
    n_chan = (len(subbands) if subbands else cfg["n_subbands"]) \
        * cfg["n_chan_per_subband"]
    n_inputs = len(snaps) * cfg["n_adc_per_snap"]
    if args.gulp is not None:
        gulp_samples = max(int(round(args.gulp / TSAMP_S)), 1)
    else:
        gulp_samples = default_gulp_samples(n_chan, n_inputs)
        avail = mem_available_bytes()
        print(f"gulp: {gulp_samples * TSAMP_S:.2f} s "
              f"({gulp_samples * n_chan * n_inputs * 8 / 1e9:.1f} GB unpacked, "
              f"{avail / 1e9:.0f} GB available)" if avail else
              f"gulp: {gulp_samples * TSAMP_S:.2f} s")

    spec, freq_v, header, n_used = voltage_autocorr(
        reader,
        n_time=int(round(args.seconds / TSAMP_S)),
        gulp_samples=gulp_samples,
        snaps=snaps, subbands=subbands,
    )
    print(f"voltage autocorr: {len(spec)} SNAPs x {next(iter(spec.values())).shape}"
          f" from {n_used} samples ({n_used * TSAMP_S:.2f} s)")
    print(f"voltage freq: {freq_v[0]:.4f} -> {freq_v[-1]:.4f} MHz")

    # Correlator side: the single integration covering the dump. The reader
    # only returns integrations that fit inside the requested window and one
    # integration is ~137 s, so ask for a generous window and keep only the
    # integration whose span contains the middle of the dump.
    t_start = psrdada_to_unix(header.get("DUMP_UTC_START",
                                         header.get("UTC_START", "")))
    if t_start is None:
        raise RuntimeError("Dump header has no parseable DUMP_UTC_START")
    t_stop = psrdada_to_unix(header.get("DUMP_UTC_STOP", "")) or t_start
    dump_mid = 0.5 * (t_start + t_stop)
    result = read_visibilities(
        unix_to_datetime(dump_mid - 300), unix_to_datetime(dump_mid + 300),
        data_root=args.vis_root, verbose=True,
    )
    times = np.asarray(result.time_unix, dtype=np.float64)
    dt_s = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    pick = int(np.argmin(np.abs(times + 0.5 * dt_s - dump_mid)))
    vis = result.vis[pick:pick + 1]
    print(f"visibilities: using integration {pick + 1}/{times.size} "
          f"starting {unix_to_datetime(times[pick]).strftime('%H:%M:%S')} UTC "
          f"(dump at {unix_to_datetime(dump_mid).strftime('%H:%M:%S')} UTC), "
          f"freq {result.freq_mhz[0]:.4f} -> {result.freq_mhz[-1]:.4f} MHz")

    df = mapping.dataframe
    if "functional" in df.columns:
        n_dead = int((df["functional"] == 0).sum())
        if n_dead:
            print(f"note: {n_dead} CSV rows have functional=0 "
                  f"(plotted anyway)")
    vis_autos = autos_from_vis(vis, df["packet_index"].tolist())

    os.makedirs(args.out_dir, exist_ok=True)
    written = plot_snap_autocorrs(
        spec, freq_v, vis_autos, result.freq_mhz, df, args.out_dir,
        voltage_label=f"voltage ({n_used * TSAMP_S:.1f} s)",
        vis_label=f"visibility (1 integration"
                  + (f", {dt_s:.0f} s)" if dt_s else ")"),
    )
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
