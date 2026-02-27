#!/usr/bin/env python
"""
Filterbank inspection script.

Reads a .fil file, prints header summary, and generates diagnostic plots.
Supports dedispersion and time-range selection.

Usage:
    python examples/inspect_filterbank.py --input beam.fil
    python examples/inspect_filterbank.py --input beam.fil --dm 125.0 --time-range 15.5 17.5
    python examples/inspect_filterbank.py --input beam.fil --dm 125.0 --no-dedisperse
"""

import argparse
import os
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect a SIGPROC filterbank file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True, help="Path to .fil file"
    )
    parser.add_argument(
        "--dm", type=float, default=0.0, help="Dispersion measure (pc/cm^3, default: 0.0)"
    )
    parser.add_argument(
        "--time-range", type=float, nargs=2, metavar=("START", "END"),
        help="Time range in seconds (e.g., --time-range 15.5 17.5)"
    )
    parser.add_argument(
        "--no-dedisperse", action="store_true",
        help="Show raw dynamic spectrum even when dm > 0"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PNG path (auto-generated from input filename if omitted)"
    )
    parser.add_argument(
        "--scale", type=str, default="linear", choices=["linear", "db"],
        help="Intensity scale: 'linear' or 'db' (default: linear)"
    )
    return parser.parse_args()


def auto_output_path(input_path, dm, dedisperse, time_range):
    """Generate output filename from input path and options."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    parts = [base]
    if dm > 0:
        parts.append(f"DM{dm:.1f}")
        if not dedisperse:
            parts.append("raw")
    if time_range is not None:
        parts.append(f"t{time_range[0]:.1f}-{time_range[1]:.1f}")
    return "_".join(parts) + ".png"


def print_header_summary(header, backend):
    """Print a concise header summary."""
    nchans = header.get("nchans", "?")
    nbits = header.get("nbits", "?")
    fch1 = header.get("fch1", "?")
    foff = header.get("foff", "?")
    tsamp = header.get("tsamp", "?")
    nsamples = header.get("_nsamples", "?")
    source = header.get("source_name", "unknown")

    print(f"  Source:     {source}")
    print(f"  Channels:   {nchans}")
    print(f"  Bits:       {nbits}")
    print(f"  fch1:       {fch1} MHz")
    print(f"  foff:       {foff} MHz")
    print(f"  tsamp:      {tsamp} s")
    print(f"  Samples:    {nsamples}")
    if isinstance(tsamp, (int, float)) and isinstance(nsamples, (int, float)):
        print(f"  Duration:   {nsamples * tsamp:.2f} s")
    print(f"  Backend:    {backend}")


def main():
    args = parse_args()

    from casm_io.filterbank import read_filterbank
    from casm_io.filterbank.plotting import (
        plot_dynamic_spectrum,
        plot_dedispersed_waterfall,
    )

    # Read file
    print(f"Reading {args.input} ...")
    result = read_filterbank(args.input)

    header = result["header"]
    data = result["data"]
    freq_mhz = result["freq_mhz"]
    backend = result["backend_used"]

    print("Header summary:")
    print_header_summary(header, backend)
    print(f"  Data shape: {data.shape}")
    print(f"  Freq range: {freq_mhz[0]:.3f} - {freq_mhz[-1]:.3f} MHz")

    # Determine output path
    dedisperse = args.dm > 0 and not args.no_dedisperse
    output_path = args.output or auto_output_path(
        args.input, args.dm, dedisperse, args.time_range
    )

    time_range = tuple(args.time_range) if args.time_range else None

    # Plot
    if dedisperse:
        print(f"Plotting 3-panel waterfall (DM={args.dm}, scale={args.scale}) ...")
        plot_dedispersed_waterfall(
            data, header, dm=args.dm,
            time_range=time_range,
            output_path=output_path,
        )
    else:
        dm_for_plot = args.dm if (args.dm > 0 and not args.no_dedisperse) else None
        print(f"Plotting dynamic spectrum (scale={args.scale}) ...")
        plot_dynamic_spectrum(
            data, header,
            scale=args.scale,
            dm=dm_for_plot,
            time_range=time_range,
            output_path=output_path,
        )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
