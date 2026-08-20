"""Streaming beam-dump (.dat) -> per-beam SIGPROC filterbank conversion.

Converts casm_beam_dump output (4096-byte ASCII DADA header + float32 payload of
shape [ndump, nbeam, nchan, samps_per_dump]) into one .fil per dumped beam,
without loading the dump into RAM. Handles incoherent-beam dumps
(beam_Incoherent_*, nbeam=1) and REMOTE dumps: beam blocks served by
casm-corr2 land on casm-corr2's own disk, so a `host:/path/glob` source is
streamed over ssh (BatchMode; nothing is copied to local disk first).

Replaces the per-session copies of
BEAM_TESTS/casm_combine_time_contigous_multi_beam_dump_to_sigproc.py (whose
in-RAM concatenate does not scale to multi-100-GB dumps). Output is identical:
time-major (ndump*samps, nchan) float32, nbits=32.

RFI mitigation is deliberately NOT part of this tool — cleaning (filtool
kadaneF/zdot, clfd) is a separate, optional step; see the casm-wiki fold recipe.

CLI:
    casm-beamdump-to-fil '/mnt/nvme4/data/casm/beam_dumps/beam_0-63_<UTC>.dat.*' \\
        --out-dir /mnt/nvme5/fil --prefix aug05_ --name 8=b08_matched_ja
    casm-beamdump-to-fil 'casm-corr2:/mnt/nvme4/data/casm/beam_dumps/beam_384-447_<UTC>.dat.*' \\
        --out-dir /mnt/nvme5/fil --prefix aug07_

Beams without an explicit --name get "b<NN>". Keep output basenames short:
filtool crashes when the rawdatafile header reaches 80 characters, so paths
longer than 78 characters are rejected here.
"""
from __future__ import annotations

import argparse
import glob as _glob
import os
import subprocess

import numpy as np

from .header import write_sigproc_header

HDR_SIZE = 4096
MAX_RAWDATAFILE = 78  # filtool fixed buffer: crash at >= 80 chars
SSH = ["ssh", "-o", "BatchMode=yes"]


# ----------------------------------------------------------------------------
# local/remote source abstraction
# ----------------------------------------------------------------------------
def _split_host(spec: str) -> tuple[str | None, str]:
    """'host:/path' -> (host, path); plain path -> (None, path)."""
    if ":" in spec and not spec.startswith("/") and "/" in spec.split(":", 1)[1]:
        host, path = spec.split(":", 1)
        if "/" not in host:
            return host, path
    return None, spec


def _list_files(host: str | None, pattern: str) -> list[str]:
    if host is None:
        files = _glob.glob(pattern)
    else:
        r = subprocess.run(SSH + [host, f"ls -1 {pattern}"],
                           capture_output=True, text=True)
        files = [ln for ln in r.stdout.split("\n") if ln.strip()]
    return sorted(files, key=lambda f: int(f.rsplit(".", 1)[-1]))


def _file_size(host: str | None, path: str) -> int:
    if host is None:
        return os.path.getsize(path)
    r = subprocess.run(SSH + [host, f"stat -c %s {path}"],
                       capture_output=True, text=True, check=True)
    return int(r.stdout.strip())


def _read_header_bytes(host: str | None, path: str) -> bytes:
    if host is None:
        return open(path, "rb").read(HDR_SIZE)
    r = subprocess.run(SSH + [host, f"dd if={path} bs={HDR_SIZE} count=1 2>/dev/null"],
                       capture_output=True, check=True)
    return r.stdout


class _Stream:
    """Sequential reader over a dump file, local or ssh-piped."""

    def __init__(self, host: str | None, path: str):
        self.proc = None
        if host is None:
            self.f = open(path, "rb")
        else:
            self.proc = subprocess.Popen(SSH + [host, f"cat {path}"],
                                         stdout=subprocess.PIPE)
            self.f = self.proc.stdout

    def read_exact(self, n: int) -> bytes:
        chunks, got = [], 0
        while got < n:
            b = self.f.read(n - got)
            if not b:
                break
            chunks.append(b)
            got += len(b)
        return b"".join(chunks)

    def close(self):
        self.f.close()
        if self.proc is not None:
            self.proc.wait()


# ----------------------------------------------------------------------------
# header parsing
# ----------------------------------------------------------------------------
def parse_dump_header(raw: bytes) -> dict:
    out: dict[str, str] = {}
    for line in raw.decode(errors="ignore").split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def dump_start_mjd(hdr: dict) -> float:
    """UTC_START (+ PICOSECONDS offset) -> MJD."""
    from astropy.time import Time, TimeDelta

    utc = hdr["UTC_START"]
    isot = utc[::-1].replace("-", "T", 1)[::-1]  # last date dash -> T
    t = Time(isot, scale="utc", format="isot")
    t = t + TimeDelta(float(hdr.get("PICOSECONDS", 0)) * 1e-12, format="sec")
    return float(t.mjd)


# ----------------------------------------------------------------------------
# conversion
# ----------------------------------------------------------------------------
def convert_beam_dump(
    dump_glob: str,
    out_dir: str,
    names: dict[int, str] | None = None,
    prefix: str = "",
    source_name: str = "B0329+54",
    src_raj: float = 33259.4096,
    src_dej: float = 543443.329,
    max_dumps: int | None = None,
    verbose: bool = True,
) -> list[str]:
    """Stream-convert a beam dump to one .fil per beam. Returns output paths.

    dump_glob may be a local glob or 'host:/glob' (streamed over ssh).
    """
    host, pattern = _split_host(dump_glob)
    files = _list_files(host, pattern)
    if not files:
        raise FileNotFoundError(f"no files match {dump_glob!r}")
    hdr = parse_dump_header(_read_header_bytes(host, files[0]))

    incoh = "beam_Incoherent" in os.path.basename(files[0])
    if incoh:
        beam_list = [0]
        ninner = int(hdr["INCOH_BEAM_DUMP_SAMPS_PER_DUMP"])
        downsamp = float(hdr["INCOH_BEAM_DUMP_DOWNSAMP"])
    else:
        # BEAM_DUMP_LIST is the GLOBAL user request; each block file only
        # stores the listed beams inside its own [FIRST_BEAM, LAST_BEAM]
        # range (matches ldunn's casm_beam_dump_to_sigproc.py reference).
        requested = [int(x) for x in hdr["BEAM_DUMP_LIST"].split(",")]
        first = int(hdr["FIRST_BEAM"])
        last = int(hdr["LAST_BEAM"])
        beam_list = [b for b in requested if first <= b <= last]
        if not beam_list:
            raise ValueError(f"no requested beams {requested} fall in this "
                             f"file's block {first}-{last}")
        ninner = int(hdr["BEAM_DUMP_SAMPS_PER_DUMP"])
        downsamp = float(hdr["BEAM_DUMP_DOWNSAMP"])
    nchan = int(hdr["NCHAN"])
    nbeam = len(beam_list)
    tsamp = float(hdr["TSAMP"]) * downsamp * 1e-6
    tstart = dump_start_mjd(hdr)
    fch1 = float(hdr["FREQ_START"])
    foff = float(hdr["CHANBW"])
    dump_bytes = nbeam * nchan * ninner * 4
    if verbose:
        where = f"{host}:" if host else ""
        print(f"{len(files)} files ({where}) | beams {beam_list} | nchan {nchan} | "
              f"tsamp {tsamp*1e3:.4f} ms | tstart MJD {tstart:.8f} | "
              f"fch1 {fch1} foff {foff}")

    os.makedirs(out_dir, exist_ok=True)
    names = names or {}
    if prefix and not prefix.endswith(("_", "-", ".")):
        prefix = prefix + "_"      # aug07 -> aug07_b70.fil
    outnames, handles = [], []
    for b in beam_list:
        base = names.get(b, "IB" if incoh else f"b{b:02d}")
        name = os.path.join(out_dir, f"{prefix}{base}.fil")
        if len(name) > MAX_RAWDATAFILE:
            raise ValueError(f"output path {name!r} is {len(name)} chars; filtool "
                             f"crashes at rawdatafile >= 80 — shorten it")
        with open(name, "wb") as f:
            write_sigproc_header(f, {
                "rawdatafile": name, "source_name": source_name,
                "src_raj": src_raj, "src_dej": src_dej,
                "machine_id": 0, "telescope_id": 0, "data_type": 1,
                "nchans": nchan, "fch1": fch1, "foff": foff,
                "tsamp": tsamp, "tstart": tstart,
                # each output .fil holds ONE beam: nbeams must be 1 here, not
                # the block beam count, or readers compute nsamples nbeam-x small
                "nbeams": 1, "ibeam": b, "nbits": 32, "nifs": 1,
            })
        outnames.append(name)
        handles.append(open(name, "ab"))

    total = 0
    chunk = max(1, (1 << 30) // dump_bytes)  # ~1 GB reads
    try:
        for fn in files:
            ndump = (_file_size(host, fn) - HDR_SIZE) // dump_bytes
            st = _Stream(host, fn)
            st.read_exact(HDR_SIZE)
            done = 0
            while done < ndump:
                n = min(chunk, ndump - done)
                buf = st.read_exact(n * dump_bytes)
                if len(buf) < n * dump_bytes:  # truncated / still being written
                    n = len(buf) // dump_bytes
                    if n == 0:
                        break
                    buf = buf[:n * dump_bytes]
                d = np.frombuffer(buf, dtype=np.float32).reshape(n, nbeam, nchan, ninner)
                for i in range(nbeam):
                    blk = np.ascontiguousarray(d[:, i].transpose(0, 2, 1))
                    blk.reshape(-1, nchan).tofile(handles[i])
                done += n
                total += n
                if max_dumps is not None and total >= max_dumps:
                    st.close()
                    raise StopIteration
            st.close()
            if verbose:
                print(f"  {os.path.basename(fn)}: {done} dumps (cumulative {total})",
                      flush=True)
    except StopIteration:
        pass
    finally:
        for h in handles:
            h.close()
    if verbose:
        mins = total * ninner * tsamp / 60
        print(f"DONE: {total} dumps = {mins:.1f} min per beam")
        for n in outnames:
            print(f"  {n}  {os.path.getsize(n)/1e9:.2f} GB")
    return outnames


def main() -> None:
    p = argparse.ArgumentParser(
        description="Streaming beam-dump -> per-beam filterbank conversion "
                    "(local or host:/path source; see module docstring)")
    p.add_argument("dump_glob", help="glob for the .dat.* files, or 'host:/glob' (quote it)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", default="", help="prepended to every output basename")
    p.add_argument("--name", action="append", default=[], metavar="BEAM=NAME",
                   help="output name for a beam number (repeatable)")
    p.add_argument("--source-name", default="B0329+54")
    p.add_argument("--max-dumps", type=int, default=None,
                   help="stop after N dumps (testing)")
    a = p.parse_args()
    names = {}
    for spec in a.name:
        k, v = spec.split("=", 1)
        names[int(k)] = v
    convert_beam_dump(a.dump_glob, a.out_dir, names=names, prefix=a.prefix,
                      source_name=a.source_name, max_dumps=a.max_dumps)


if __name__ == "__main__":
    main()
