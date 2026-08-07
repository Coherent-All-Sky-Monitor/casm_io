"""Optional RFI mitigation for CASM filterbanks — the validated filtool pass.

Wraps pulsarx filtool (apptainer) with the settings validated on B0329
(2026-08-04/05 factorial runs): kadaneF 8 4 time-domain zaps + zdot zero-DM
removal, zapthre -1, baseline 8, float32 output. Cleaning is deliberately
separate from casm-beamdump-to-fil: conversion is always lossless; this step
is a choice (zdot is mandatory before trusting any fold/search number, but
costs ~6% SNR on clean data — see casm-wiki detections.md fold recipe).

Only run on `your`-/casm_io-written fils (casm-beamdump-to-fil output is fine);
sigpyproc-written headers make filtool emit an fchannel table that crashes
dspsr downstream (casm-wiki b0329-factorial-2026-08.md).

CLI:
    casm-fil-clean input.fil --out-base myname [--workdir DIR]
        [--source-name PSRJ0332+5434 --ra 03:32:59.4096 --dec 54:34:43.329]
    -> DIR/myname_01.fil
"""
from __future__ import annotations

import argparse
import os
import subprocess

PULSARX_SIF = "/home/casm/software/vishnu/apptainer_images/pulsarx_latest.sif"
BINDS = "/mnt/nvme5,/mnt/nvme3,/mnt/nvme4,/home/casm"


def clean_filterbank(
    fil: str,
    out_base: str,
    workdir: str | None = None,
    source_name: str = "PSRJ0332+5434",
    ra: str = "03:32:59.4096",
    dec: str = "54:34:43.329",
    threads: int = 8,
    sif: str = PULSARX_SIF,
) -> str:
    """Run the validated filtool pass; returns the cleaned fil path."""
    workdir = workdir or os.getcwd()
    from .header import read_sigproc_header
    hdr, _ = read_sigproc_header(fil)
    if hdr.get("tsamp", 0) > 0.1:
        raise ValueError(
            f"tsamp = {hdr['tsamp']:.3f} s — filtool core-dumps on slow-sampled "
            f"(monitoring/downsamp-4096) data; cleaning is for search-rate fils")
    cmd = ["apptainer", "exec", "--bind", BINDS, sif,
           "filtool",
           "--telescope", "OVRO",
           "--source_name", source_name,
           "--ra", ra, "--dec", dec,
           "--nbits", "32", "--mean", "0", "--std", "1",
           "--zapthre", "-1",
           "--baseline", "8",
           "-z", "kadaneF", "8", "4", "-z", "zdot",
           "-t", str(threads),
           "-o", out_base, "-f", os.path.abspath(fil)]
    subprocess.run(cmd, cwd=workdir, check=True)
    out = os.path.join(workdir, f"{out_base}_01.fil")
    if not os.path.exists(out):
        raise RuntimeError(f"filtool completed but {out} is missing")
    print(f"cleaned: {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validated CASM RFI pass (filtool kadaneF 8 4 + zdot); "
                    "optional, separate from conversion")
    p.add_argument("fil", help="input .fil (from casm-beamdump-to-fil)")
    p.add_argument("--out-base", required=True,
                   help="output basename -> <workdir>/<out-base>_01.fil")
    p.add_argument("--workdir", default=None, help="default: cwd")
    p.add_argument("--source-name", default="PSRJ0332+5434")
    p.add_argument("--ra", default="03:32:59.4096")
    p.add_argument("--dec", default="54:34:43.329")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    clean_filterbank(a.fil, a.out_base, workdir=a.workdir,
                     source_name=a.source_name, ra=a.ra, dec=a.dec,
                     threads=a.threads)


if __name__ == "__main__":
    main()
