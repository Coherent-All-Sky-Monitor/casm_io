#!/usr/bin/env python
"""Dump CASM voltages and read them back.

The interactive version of this walkthrough is
examples/voltage_quickstart.ipynb, which triggers dumps from Python via
casm_t2's dump_voltages(past_duration=...).

Quickstart (on casm-corr1, from the offline venv):

    source /home/casm/software/dev/casm_venvs/casm_offline_env/bin/activate
    casm-voltage-dump --last 2 --gather ~/myrun
    python voltage_dumps.py ~/myrun \\
        2026-07-29-21:06:34_0187972821983232 --seconds 1

The second argument is the dump's filename prefix,
``<UTC_START>_<16-digit byte offset>`` as in the example above. --gather is
optional: it waits for the dump files on both nodes, pulls the casm-corr2
streams into your directory (rsync; the local streams are symlinked, so
they break if the originals in cand_dumps are deleted — copy them if the
directory has to stand on its own), and prints the prefix ready-made as
part of the exact ``VoltageReader(data_dir, prefix)`` call to use. Without
it, scp the corr2 stream_N files over yourself and read the prefix off the
filenames. The full prefix pins one dump even when several share a
UTC_START: filenames are ``<UTC_START>_<byte offset>.<file number>.dada``
(seconds since start = offset / 2.0625e9), long dumps split into ~10 s
files, and the reader refuses to stitch across the gap between two dumps
that share a start time. A bare UTC_START (``2026-07-29-21:06:34``) also
works as the prefix when only one dump has it.

On disk each stream_N file is a 4096-byte ASCII header followed by one
byte per complex sample (4+4-bit real/imag, two's complement), ordered
time -> SNAP slot -> channel -> ADC. Files are pre-allocated; the header's
DUMP_BYTES marks where the valid payload ends. The reader handles all of
this — these details only matter if you parse the files yourself.

More trigger forms (the daemon says OK before the disk check runs, so
without --gather always confirm the files exist):

    casm-voltage-dump --next 10 --dry-run      # check the plan and disk first
    casm-voltage-dump --next 10                # 10 s starting 5 s from now
    casm-voltage-dump --streams 1,2 --next 60  # 437.5-468.75 MHz, covers the
                                               # 440-465 MHz live band

The ring buffer holds ~26 s of the past, so --last is capped there. The
writer wants 3x the dump size free per node before it will save anything
(2.06 GB/s per stream). Frequency runs top-down: stream 0 is
468.75-484.375 MHz, stream 5 is 390.625-406.25. Nothing deletes dumps
automatically any more (the T3 janitor is off), so clean up what you are
done with.

Reading: the default is the raw SNAP layout, a dict
``{snap: (n_time, n_chan, 12 adc)}``; pass --csv for one per-antenna array
in CSV row order instead. --streams 0,1 reads only those sub-bands
(contiguous selection), --snaps 0,3 only those SNAP slots. Data is read in
gulps sized to the RAM currently free (override with --gulp seconds), so
the full voltage array never exists in memory.

Forming visibilities is optional and just an example: add --correlate (or
--out file.npz, which implies it) to multiply the inputs pairwise and
average over --tint; only the visibilities then accumulate across gulps.
By default the script only reads, prints what it found, and exits. The
same recipe as a function is casm_io.voltage.correlate().
"""

import argparse

import numpy as np

from casm_io.voltage.reader import VoltageReader
from casm_io.autocorr import psrdada_to_unix
from casm_io._progress import print_progress
from casm_io._time import format_time_span, unix_to_datetime

TSAMP_S = 32.768e-6  # one time sample; 1 s of data is 30518 samples
LOCAL_TZ = "America/Los_Angeles"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("data_dir", help="directory containing the stream_N folders")
parser.add_argument("timestamp", help="filename prefix, e.g. 2026-07-29-21:06:34")
parser.add_argument("--seconds", type=float, default=0.5, help="how much to read")
parser.add_argument("--tint", type=float, default=0.1, help="visibility integration time")
parser.add_argument("--gulp", type=float, default=None,
                    help="seconds of voltages held in RAM at a time "
                         "(default: sized from the RAM currently available)")
parser.add_argument("--streams", default=None,
                    help="comma-separated sub-band indices to read, e.g. 0,1 "
                         "(default: all six)")
parser.add_argument("--snaps", default=None,
                    help="comma-separated SNAP slots to read, e.g. 0,3 "
                         "(default: all active)")
parser.add_argument("--csv", default=None,
                    help="antenna layout CSV (snap/adc -> antenna mapping and "
                         "positions); default is the raw SNAP layout")
parser.add_argument("--correlate", action="store_true",
                    help="also form visibilities (implied by --out)")
parser.add_argument("--out", default=None, help="write visibilities to this .npz")
args = parser.parse_args()
correlate = args.correlate or args.out is not None

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
if args.gulp is not None:
    gulp_samples = int(round(args.gulp / TSAMP_S))
else:
    from casm_io.autocorr import default_gulp_samples
    cfg = reader._cfg
    n_inputs = len(snaps or cfg["active_snaps"]) * cfg["n_adc_per_snap"]
    n_chan = (len(subbands) if subbands else cfg["n_subbands"]) \
        * cfg["n_chan_per_subband"]
    gulp_samples = default_gulp_samples(n_chan, n_inputs)
bins_per_gulp = max(gulp_samples // bin_samples, 1)

vis = None
freq = None
res = None
power_sum = 0.0
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
        dump_start = psrdada_to_unix(
            res.header.get("DUMP_UTC_START", res.header.get("UTC_START", ""))
        )
        if dump_start is not None:
            span = n_bins * bin_samples * TSAMP_S
            print(f"\nReading (UTC): "
                  f"{format_time_span(dump_start, dump_start + span)}")
            print(f"Reading (PT):  "
                  f"{format_time_span(dump_start, dump_start + span, LOCAL_TZ)}")
        print(f"\ngulp: {v.shape} {v.dtype}, {v.nbytes / 1e9:.2f} GB in RAM "
              f"({n_bins} integrations in {-(-n_bins // bins_per_gulp)} gulps)")
        if res.filled_subbands:
            print(f"zero-filled sub-bands (no data on disk): {res.filled_subbands}")
        print(f"freq: {freq[0]:.4f} -> {freq[-1]:.4f} MHz, "
              f"{abs(freq[1] - freq[0]) * 1e3:.3f} kHz channels")
        if correlate:
            # vis[t, f, i, j] = <v_i conj(v_j)> for bin t and channel f.
            vis = np.empty((n_bins, v.shape[1], v.shape[2], v.shape[2]),
                           dtype=np.complex64)

    if correlate:
        # Visibilities: multiply voltages pairwise and average over an
        # integration time. Averaging the voltages first and multiplying
        # after does not work.
        vb = v[: nb_read * bin_samples].reshape(nb_read, bin_samples,
                                                v.shape[1], v.shape[2])
        vis[b0:b0 + nb_read] = (
            np.einsum("tbfi,tbfj->tfij", vb, np.conj(vb)) / bin_samples
        )
    power_sum += float(np.sum(np.abs(v[: nb_read * bin_samples]) ** 2))
    suffix = "integrations"
    if dump_start is not None:
        now = unix_to_datetime(
            dump_start + (b0 + nb_read) * bin_samples * TSAMP_S, LOCAL_TZ,
        )
        suffix = f"integrations, at {now.strftime('%H:%M:%S.%f')[:-3]} PT"
    print_progress(b0 + nb_read, n_bins,
                   prefix="Correlating" if correlate else "Reading",
                   suffix=suffix)
if n_bins and (b0 + nb_read) < n_bins:
    print()  # close the progress bar line after a short dump

if res is None or freq is None:
    raise SystemExit("no data read")

print("\ndump header (stream {}):".format(reader.subbands_found[0]))
for key in ("UTC_START", "DUMP_UTC_START", "DUMP_UTC_STOP", "DUMP_BYTES",
            "NCHAN", "NANT", "TSAMP", "RESOLUTION", "STREAM_SUBBAND_ID"):
    if key in res.header:
        print(f"  {key:16s} {res.header[key]}")

n_samples = n_bins * bin_samples
print(f"\nread {n_samples} samples ({n_samples * TSAMP_S:.2f} s), "
      f"mean |v|^2 = {power_sum / max(n_samples * v.shape[1] * v.shape[2], 1):.3f}")

if correlate:
    vis = vis[:n_bins]
    # The diagonal is the autocorrelation (real, positive); vis[..., i, j] is
    # the visibility on baseline i-j. With --csv, antenna positions for
    # fringe work are in res.antenna_df (x, y, z columns, metres, local ENU).
    auto = np.diagonal(vis, axis1=2, axis2=3).real
    print(f"visibilities: {vis.shape} = (integrations, channels, input, input), "
          f"tint {bin_samples * TSAMP_S:.3f} s")
    print(f"autocorrelations: min {auto.min():.3g}, mean {auto.mean():.3g}")

if args.out:
    from casm_io.correlator.writer import write_npz
    write_npz(args.out, vis=vis, freq_mhz=freq.astype(np.float32),
              time_unix=np.arange(n_bins) * bin_samples * TSAMP_S,
              baseline_convention=f"vis[t,f,i,j] = <v_i conj(v_j)>, {convention}",
              filled_subbands=str(res.filled_subbands))
    print(f"wrote {args.out}")
