#!/usr/bin/env python
"""Dump CASM voltages and read them back.

Everything runs from the offline venv:

    source /home/casm/software/dev/casm_venvs/casm_offline_env/bin/activate

Triggering a dump (run on casm-corr1; works whether or not t2d is up):

    casm-voltage-dump --next 10 --dry-run      # check the plan and disk first
    casm-voltage-dump --next 10                # 10 s starting 5 s from now
    casm-voltage-dump --last 5                 # the 5 s ending 2 s ago
    casm-voltage-dump --streams 1,2 --next 60  # 437.5-468.75 MHz, covers the
                                               # 440-465 MHz live band

The ring buffer holds ~26 s of the past, so --last is capped there. The
writer wants 3x the dump size free per node before it will save anything.
Frequency runs top-down: stream 0 is 468.75-484.375 MHz, stream 5 is
390.625-406.25.

Streams 0-2 land on casm-corr1, streams 3-5 on casm-corr2, both under
/mnt/nvme4/data/casm/cand_dumps/stream_N/. The daemon says OK before the disk
check runs, so always confirm the files exist. To analyse all six streams from
one machine, copy the corr2 ones over:

    scp casm-corr2:/mnt/nvme4/data/casm/cand_dumps/stream_3/<file> stream_3/

Filenames look like 2026-07-29-21:06:34_0166501757706240.000000.dada:
observation start, then byte offset (seconds since start = offset / 2.0625e9),
then file number. Long dumps split into ~10 s files; the reader stitches them.
Several dumps can share a UTC_START, and the reader refuses to stitch across
the gap between two of them, so include the offset in the timestamp prefix you
pass here when more than one dump shares a start time.

Then:

    python voltage_dumps.py /path/to/cand_dumps 2026-07-29-21:06:34 --seconds 1

By default you get the raw SNAP layout; pass --csv for per-antenna order.
--streams 0,1 reads only those sub-bands (contiguous selection), --snaps 0,3
only those SNAP slots. The data is read in gulps of --gulp seconds (default:
one integration time) so the full voltage array never sits in RAM — only the
visibilities accumulate.
"""

import argparse

import numpy as np

from casm_io.voltage.reader import VoltageReader

TSAMP_S = 32.768e-6  # one time sample; 1 s of data is 30518 samples

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("data_dir", help="directory containing the stream_N folders")
parser.add_argument("timestamp", help="filename prefix, e.g. 2026-07-29-21:06:34")
parser.add_argument("--seconds", type=float, default=0.5, help="how much to read")
parser.add_argument("--tint", type=float, default=0.1, help="visibility integration time")
parser.add_argument("--gulp", type=float, default=None,
                    help="seconds of voltages held in RAM at a time "
                         "(default: one --tint integration)")
parser.add_argument("--streams", default=None,
                    help="comma-separated sub-band indices to read, e.g. 0,1 "
                         "(default: all six)")
parser.add_argument("--snaps", default=None,
                    help="comma-separated SNAP slots to read, e.g. 0,3 "
                         "(default: all active)")
parser.add_argument("--csv", default=None,
                    help="antenna layout CSV (snap/adc -> antenna mapping and "
                         "positions); default is the raw SNAP layout")
parser.add_argument("--out", default=None, help="write visibilities to this .npz")
args = parser.parse_args()

subbands = [int(x) for x in args.streams.split(",")] if args.streams else None
snaps = [int(x) for x in args.snaps.split(",")] if args.snaps else None

reader = VoltageReader(args.data_dir, args.timestamp)
print(f"streams found: {reader.subbands_found}  (missing ones read as zeros)")

# Read in gulps of whole integration bins, so no bin straddles a gulp and the
# full voltage array never exists in RAM.
bin_samples = int(round(args.tint / TSAMP_S))
n_bins = int(round(args.seconds / TSAMP_S)) // bin_samples
if n_bins == 0:
    raise SystemExit(f"--tint {args.tint} is longer than --seconds {args.seconds}")
bins_per_gulp = max(int(round((args.gulp or args.tint) / TSAMP_S)) // bin_samples, 1)

vis = None
freq = None
res = None
for b0 in range(0, n_bins, bins_per_gulp):
    nb = min(bins_per_gulp, n_bins - b0)
    first = b0 == 0
    res = reader.read_full_band(
        antenna_csv=args.csv, snaps=snaps, subbands=subbands,
        n_time=nb * bin_samples, time_offset=b0 * bin_samples,
        verbose=first,
    )

    # res.voltages is complex64:
    #   with --csv        (n_time, n_chan, n_ant), antennas in CSV row order
    #                     (res.antenna_df has the rows)
    #   without (default) a dict {snap: (n_time, n_chan, 12)}; stacked below
    #                     into (n_time, n_chan, n_snaps*12) in (snap slot,
    #                     adc) order so cross-SNAP baselines are kept
    # axis 0 is time (one sample every 32.768 us), axis 1 frequency in
    # descending order (res.freq_mhz has the values). Samples are 4+4 bit on
    # disk and come back as integers -8..7 in the complex64.
    if isinstance(res.voltages, dict):
        input_snaps = sorted(res.voltages)
        v = np.concatenate([res.voltages[s] for s in input_snaps], axis=2)
        convention = (f"inputs are (snap, adc) pairs, SNAPs {input_snaps} "
                      f"x 12 ADCs each")
    else:
        v = res.voltages
        convention = "CSV row order"

    nb_read = v.shape[0] // bin_samples
    if v.shape[0] < nb * bin_samples:
        n_bins = b0 + nb_read
        print(f"  dump ends early: {v.shape[0]} samples in this gulp, "
              f"stopping at {n_bins} integrations")
        if nb_read == 0:
            break

    if first:
        freq = res.freq_mhz
        print(f"\ngulp: {v.shape} {v.dtype}, {v.nbytes / 1e9:.2f} GB in RAM "
              f"({n_bins} integrations in {-(-n_bins // bins_per_gulp)} gulps)")
        if res.filled_subbands:
            print(f"zero-filled sub-bands (no data on disk): {res.filled_subbands}")
        print(f"freq: {freq[0]:.4f} -> {freq[-1]:.4f} MHz, "
              f"{abs(freq[1] - freq[0]) * 1e3:.3f} kHz channels")
        # vis[t, f, i, j] = <v_i conj(v_j)> for integration bin t, channel f.
        vis = np.empty((n_bins, v.shape[1], v.shape[2], v.shape[2]),
                       dtype=np.complex64)

    # Visibilities: multiply voltages pairwise and average over an integration
    # time. Averaging the voltages first and multiplying after does not work.
    vb = v[: nb_read * bin_samples].reshape(nb_read, bin_samples,
                                            v.shape[1], v.shape[2])
    vis[b0:b0 + nb_read] = (
        np.einsum("tbfi,tbfj->tfij", vb, np.conj(vb)) / bin_samples
    )
    print(f"  integrations {b0}-{b0 + nb_read - 1} of {n_bins} done")

if vis is None:
    raise SystemExit("no data read")
vis = vis[:n_bins]

print("\ndump header (stream {}):".format(reader.subbands_found[0]))
for key in ("UTC_START", "DUMP_UTC_START", "DUMP_UTC_STOP", "DUMP_BYTES",
            "NCHAN", "NANT", "TSAMP", "RESOLUTION", "STREAM_SUBBAND_ID"):
    if key in res.header:
        print(f"  {key:16s} {res.header[key]}")

# The diagonal is the autocorrelation (real, positive); vis[..., i, j] is the
# visibility on baseline i-j. With --csv, antenna positions for fringe work
# are in res.antenna_df (x, y, z columns, metres, local ENU).
auto = np.diagonal(vis, axis1=2, axis2=3).real
print(f"\nvisibilities: {vis.shape} = (integrations, channels, input, input), "
      f"tint {bin_samples * TSAMP_S:.3f} s")
print(f"autocorrelations: min {auto.min():.3g}, mean {auto.mean():.3g}")

if args.out:
    from casm_io.correlator.writer import write_npz
    write_npz(args.out, vis=vis, freq_mhz=freq.astype(np.float32),
              time_unix=np.arange(n_bins) * bin_samples * TSAMP_S,
              baseline_convention=f"vis[t,f,i,j] = <v_i conj(v_j)>, {convention}",
              filled_subbands=str(res.filled_subbands))
    print(f"wrote {args.out}")
