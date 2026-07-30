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

The ring buffer holds ~26 s of the past, so --last is capped there. --next is
capped by what survives on disk: the T3 janitor keeps only ~150 GB per stream,
about 72 s at the full 2.06 GB/s rate, and deletes dumps oldest-first past
that — so a dump longer than ~72 s eats itself, and anything you want to keep
has to be moved off the nodes promptly. The writer also wants 3x the dump size
free per node before it will save anything. Frequency runs top-down: stream 0
is 468.75-484.375 MHz, stream 5 is 390.625-406.25.

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
parser.add_argument("--csv", default="/home/casm/software/dev/antenna_layouts/current",
                    help="antenna layout CSV (snap/adc -> antenna mapping and positions)")
parser.add_argument("--out", default=None, help="write visibilities to this .npz")
args = parser.parse_args()

reader = VoltageReader(args.data_dir, args.timestamp)
print(f"streams found: {reader.subbands_found}  (missing ones read as zeros)")

n_time = int(round(args.seconds / TSAMP_S))
res = reader.read_full_band(antenna_csv=args.csv, n_time=n_time)

# res.voltages is complex64 with shape (n_time, 3072, n_ant):
#   axis 0  time, one sample every 32.768 us
#   axis 1  frequency, 3072 channels, descending (res.freq_mhz has the values)
#   axis 2  antennas, in CSV row order (res.antenna_df has the rows)
# Without antenna_csv= you get the raw layout instead: a dict {snap: array of
# shape (n_time, 3072, 12)} where 12 is the ADC inputs on that SNAP board.
# Samples are 4+4 bit on disk and come back as integers -8..7 in the complex64.
v = res.voltages
freq = res.freq_mhz
print(f"voltages: {v.shape} {v.dtype}, {v.nbytes / 1e9:.1f} GB in memory")
if res.filled_subbands:
    print(f"zero-filled sub-bands (no data on disk): {res.filled_subbands}")
print(f"freq: {freq[0]:.4f} -> {freq[-1]:.4f} MHz, "
      f"{abs(freq[1] - freq[0]) * 1e3:.3f} kHz channels")

print("\ndump header (stream {}):".format(reader.subbands_found[0]))
for key in ("UTC_START", "DUMP_UTC_START", "DUMP_UTC_STOP", "DUMP_BYTES",
            "NCHAN", "NANT", "TSAMP", "RESOLUTION", "STREAM_SUBBAND_ID"):
    if key in res.header:
        print(f"  {key:16s} {res.header[key]}")

# Visibilities: multiply voltages pairwise and average over an integration
# time. Averaging the voltages first and multiplying after does not work.
bin_samples = int(round(args.tint / TSAMP_S))
n_bins = v.shape[0] // bin_samples
if n_bins == 0:
    raise SystemExit(f"--tint {args.tint} is longer than the data read")
vb = v[: n_bins * bin_samples].reshape(n_bins, bin_samples, v.shape[1], v.shape[2])
vis = np.einsum("tbfi,tbfj->tfij", vb, np.conj(vb)) / bin_samples
vis = vis.astype(np.complex64)

# vis[t, f, i, j] = <v_i conj(v_j)> for integration bin t and channel f.
# The diagonal is the autocorrelation (real, positive); vis[..., i, j] is the
# visibility on baseline i-j. Antenna positions for fringe work are in
# res.antenna_df (x, y, z columns, metres, local ENU).
auto = np.diagonal(vis, axis1=2, axis2=3).real
print(f"\nvisibilities: {vis.shape} = (integrations, channels, ant, ant), "
      f"tint {bin_samples * TSAMP_S:.3f} s")
print(f"autocorrelations: min {auto.min():.3g}, mean {auto.mean():.3g}")

if args.out:
    from casm_io.correlator.writer import write_npz
    write_npz(args.out, vis=vis, freq_mhz=freq.astype(np.float32),
              time_unix=np.arange(n_bins) * bin_samples * TSAMP_S,
              baseline_convention="vis[t,f,i,j] = <v_i conj(v_j)>, CSV row order",
              filled_subbands=str(res.filled_subbands))
    print(f"wrote {args.out}")
