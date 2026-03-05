"""
Correlator visibility file reader.

Reads raw .dat visibility files from the CASM correlator.
Files are named like: {base_str}.{index} where index is 0, 1, 2, ...
Each file contains ntime_per_file integrations of nchan x n_baselines x 2 (re/im) int32s.
"""

import gc
import glob
import os
import re
import warnings
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from . import baselines
from .formats import VisibilityFormat
from .header import get_header_offset, format_from_header
from .._progress import print_progress
from .._results import VisibilityResult
from .._time import format_time_span


def extract_file_index(path: str) -> int:
    """Extract numeric index from file path (e.g., 'file.123' -> 123)."""
    m = re.search(r"\.(\d+)$", path)
    return int(m.group(1)) if m else -1


def discover_files(data_dir: str, base_str: str) -> dict[int, str]:
    """
    Find all .{index} files for an observation.

    Parameters
    ----------
    data_dir : str
        Directory containing visibility files.
    base_str : str
        Observation identifier (UTC timestamp like '2026-01-27-20:38:33').

    Returns
    -------
    dict
        Mapping of file index to full path.
    """
    pattern = os.path.join(data_dir, f"{base_str}.*")
    files = glob.glob(pattern)
    idx_to_path = {}
    for f in files:
        idx = extract_file_index(f)
        if idx >= 0:
            idx_to_path[idx] = f
    return idx_to_path


class VisibilityReader:
    """
    Correlator visibility reader with file discovery on init.

    Parameters
    ----------
    data_dir : str
        Directory containing visibility .dat files.
    base_str : str
        Observation identifier (UTC timestamp).
    fmt : VisibilityFormat, optional
        Format configuration (from load_format). If None, auto-detected
        from file header (new files with 4096-byte header). Required for
        old files without headers.
    """

    def __init__(self, data_dir: str, base_str: str, fmt: VisibilityFormat | None = None):
        self._data_dir = data_dir
        self._base_str = base_str
        self._fmt_explicit = fmt is not None

        self._idx_to_path = discover_files(data_dir, base_str)
        if not self._idx_to_path:
            raise RuntimeError(f"No files found for {base_str} in {data_dir}")

        # Auto-detect format from header if fmt not provided
        if fmt is None:
            first_path = self._idx_to_path[min(self._idx_to_path)]
            offset, header = get_header_offset(first_path)
            if header is not None:
                fmt = format_from_header(header)
            else:
                raise ValueError(
                    "No header found in data files and no fmt provided. "
                    "Pass a VisibilityFormat via load_format() for old files."
                )

        self._fmt = fmt

        self._available_idxs = sorted(self._idx_to_path.keys())
        self._max_idx = self._available_idxs[-1]

        # Parse base time
        self._t0_unix = (
            datetime.strptime(base_str, "%Y-%m-%d-%H:%M:%S")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )

    @property
    def n_files(self) -> int:
        """Number of available .dat files."""
        return len(self._available_idxs)

    @property
    def available_indices(self) -> list[int]:
        """Sorted list of available file indices."""
        return list(self._available_idxs)

    @property
    def missing_indices(self) -> list[int]:
        """Indices missing from the contiguous range [0, max_idx]."""
        full = set(range(self._max_idx + 1))
        return sorted(full - set(self._available_idxs))

    @property
    def time_span(self) -> tuple[float, float]:
        """(start_unix, end_unix) of available data."""
        file_dur = self._fmt.file_duration_s
        start = self._t0_unix
        end = self._t0_unix + (self._max_idx + 1) * file_dur
        return (start, end)

    def time_span_str(self, tz: str = "UTC") -> str:
        """Human-readable time span in the given timezone.

        Parameters
        ----------
        tz : str
            Timezone name (e.g. 'UTC', 'America/Los_Angeles').

        Returns
        -------
        str
            Formatted string like '2026-01-28 12:00:00 UTC -> 2026-01-28 22:00:00 UTC'.
        """
        start, end = self.time_span
        return format_time_span(start, end, tz)

    def read(
        self,
        time_start: str | datetime | None = None,
        time_end: str | datetime | None = None,
        time_tz: str = "UTC",
        nfiles: int | None = None,
        skip_nfiles: int = 0,
        ref: int | None = None,
        targets: list[int] | None = None,
        freq_order: str = "descending",
        verbose: bool = True,
    ) -> dict:
        """
        Read visibility data from .dat files.

        Parameters
        ----------
        time_start : str or datetime, optional
            Start time for slicing. None = from beginning.
        time_end : str or datetime, optional
            End time for slicing. None = to end.
        time_tz : str
            Timezone for time_start/end (default 'UTC').
        nfiles : int, optional
            Number of files to read from the start. Mutually exclusive
            with time_end.
        skip_nfiles : int
            Number of files to skip before reading. Requires nfiles.
        ref : int, optional
            Reference input index for baseline extraction.
        targets : list of int, optional
            Target input indices.
        freq_order : str
            'descending' (default, native) or 'ascending'.
        verbose : bool
            Print progress messages.

        Returns
        -------
        dict with keys:
            vis : np.ndarray
                (T, F, n_baselines) or (T, F, n_targets) complex64.
            freq_mhz : np.ndarray
                Frequency axis in MHz.
            time_unix : np.ndarray
                Unix timestamps per integration.
            metadata : dict
                Format info, files used, missing_files list, etc.
        """
        fmt = self._fmt

        if nfiles is not None and time_end is not None:
            raise ValueError("nfiles and time_end are mutually exclusive")
        if skip_nfiles and nfiles is None:
            raise ValueError("skip_nfiles requires nfiles (use time_start/time_end for time-based slicing)")

        tz_obj = ZoneInfo(time_tz) if time_tz != "UTC" else timezone.utc

        def parse_dt(spec):
            if spec is None:
                return None
            if isinstance(spec, str):
                dt = datetime.fromisoformat(spec)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=tz_obj)
                return dt
            return spec.replace(tzinfo=tz_obj) if spec.tzinfo is None else spec

        utc_start_dt = parse_dt(time_start)
        utc_end_dt = parse_dt(time_end)
        if utc_start_dt:
            utc_start_dt = utc_start_dt.astimezone(timezone.utc)
        if utc_end_dt:
            utc_end_dt = utc_end_dt.astimezone(timezone.utc)

        t0_unix = self._t0_unix
        file_dur_s = fmt.file_duration_s
        data_start_unix = t0_unix
        data_end_unix = t0_unix + (self._max_idx + 1) * file_dur_s

        _LOCAL_TZ = "America/Los_Angeles"

        if verbose:
            print(f"Data span (UTC): {format_time_span(data_start_unix, data_end_unix)}")
            print(f"Data span (PT):  {format_time_span(data_start_unix, data_end_unix, _LOCAL_TZ)}")
            print(f"Files available: {self.n_files} (indices 0-{self._max_idx})")

            # Timezone echo: show local and UTC times when non-UTC timezone used
            if time_tz != "UTC" and (time_start is not None or time_end is not None):
                local_start = parse_dt(time_start) if time_start else None
                local_end = parse_dt(time_end) if time_end else None
                local_parts = []
                utc_parts = []
                if local_start:
                    local_parts.append(local_start.strftime("%Y-%m-%d %H:%M:%S"))
                    utc_parts.append(utc_start_dt.strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    local_parts.append("(start)")
                    utc_parts.append("(start)")
                if local_end:
                    local_parts.append(local_end.strftime("%Y-%m-%d %H:%M:%S"))
                    utc_parts.append(utc_end_dt.strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    local_parts.append("(end)")
                    utc_parts.append("(end)")
                print(f"Requested ({time_tz}): {local_parts[0]} -> {local_parts[1]}")
                print(f"Requested (UTC):{' ' * (len(time_tz) - 2)} {utc_parts[0]} -> {utc_parts[1]}")

        # Determine time range
        utc_start_unix = utc_start_dt.timestamp() if utc_start_dt else data_start_unix
        utc_end_unix = utc_end_dt.timestamp() if utc_end_dt else data_end_unix

        # Handle nfiles parameter
        if nfiles is not None:
            # Start from first file (or from time_start if given)
            start_file_idx = 0
            if utc_start_dt:
                delta = utc_start_unix - t0_unix
                start_file_idx = int(delta / file_dur_s)
            start_file_idx += skip_nfiles
            needed_file_idxs = list(range(start_file_idx, start_file_idx + nfiles))
            utc_end_unix = t0_unix + (start_file_idx + nfiles) * file_dur_s
        else:
            if utc_start_dt is None and utc_end_dt is None:
                # No time slicing — just use all available files
                needed_file_idxs = self._available_idxs
            else:
                if utc_start_unix < data_start_unix or utc_end_unix > data_end_unix:
                    raise ValueError(
                        f"Requested time range outside available data.\n"
                        f"Requested: {utc_start_unix} -> {utc_end_unix}\n"
                        f"Available: {data_start_unix} -> {data_end_unix}"
                    )

                # Convert to integration indices
                # Subtract small epsilon before ceil to prevent float noise
                # (e.g., 352.0000000003 → 352 instead of 353)
                delta_start = utc_start_unix - t0_unix
                delta_end = utc_end_unix - t0_unix

                k_start = int(np.ceil(delta_start / fmt.dt_raw_s - 1e-6))
                k_stop = int(np.ceil(delta_end / fmt.dt_raw_s - 1e-6))
                if k_stop <= k_start:
                    raise ValueError("Requested window contains zero integrations")

                file_start_idx = k_start // fmt.ntime_per_file
                file_stop_idx_excl = (k_stop - 1) // fmt.ntime_per_file + 1
                needed_file_idxs = list(range(file_start_idx, file_stop_idx_excl))

        # Check for missing files — warn + zero-fill
        missing = [i for i in needed_file_idxs if i not in self._idx_to_path]
        if missing and nfiles is None:
            raise RuntimeError(f"Missing required .dat files for indices: {missing}")
        if missing and nfiles is not None:
            warnings.warn(
                f"Missing {len(missing)} files in requested range: indices {missing}. "
                f"These will be zero-filled.",
                stacklevel=2,
            )

        if verbose:
            if nfiles is not None:
                nfiles_start_unix = t0_unix + needed_file_idxs[0] * file_dur_s
                skip_msg = f" (skipped {skip_nfiles})" if skip_nfiles else ""
                print(f"Reading {nfiles} files starting from index {needed_file_idxs[0]}{skip_msg}")
                print(f"Time range (UTC): {format_time_span(nfiles_start_unix, utc_end_unix)}")
                print(f"Time range (PT):  {format_time_span(nfiles_start_unix, utc_end_unix, _LOCAL_TZ)}")
            else:
                # Compute integration indices for verbose output
                delta_start = utc_start_unix - t0_unix
                delta_end = utc_end_unix - t0_unix
                k_start_v = int(np.ceil(delta_start / fmt.dt_raw_s - 1e-6))
                k_stop_v = int(np.ceil(delta_end / fmt.dt_raw_s - 1e-6))
                print(f"Reading integrations {k_start_v}-{k_stop_v} from files "
                      f"{needed_file_idxs[0]}-{needed_file_idxs[-1]}")

        # Build frequency axis
        freq_mhz = fmt.get_frequency_axis(order=freq_order)

        # Baseline extraction setup
        nbaseline = fmt.n_baselines
        bl_idx = None
        bl_conj = None
        extract_specific = ref is not None

        if extract_specific:
            if targets is None:
                targets = [i for i in range(fmt.nsig) if i != ref]
            bl_idx, bl_conj = baselines.build_baseline_plan(ref, targets, fmt.nsig)
            n_output = len(targets)
        else:
            n_output = nbaseline

        # Determine integration slicing for non-nfiles mode
        if nfiles is None:
            delta_start = utc_start_unix - t0_unix
            delta_end = utc_end_unix - t0_unix
            k_start = int(np.ceil(delta_start / fmt.dt_raw_s - 1e-6))
            k_stop = int(np.ceil(delta_end / fmt.dt_raw_s - 1e-6))
        else:
            # For nfiles mode, read all integrations from requested files
            k_start = needed_file_idxs[0] * fmt.ntime_per_file
            k_stop = (needed_file_idxs[-1] + 1) * fmt.ntime_per_file

        # Read files
        vis_chunks = []
        time_chunks = []
        kept_files = []
        file_headers = {}

        denom = fmt.nchan * nbaseline * 2  # int32 count per integration

        for file_num, file_idx in enumerate(needed_file_idxs):
            if verbose:
                print_progress(file_num + 1, len(needed_file_idxs), prefix="Reading files")
            k0 = file_idx * fmt.ntime_per_file
            k1 = (file_idx + 1) * fmt.ntime_per_file
            s0 = max(k_start, k0) - k0
            s1 = min(k_stop, k1) - k0
            if s1 <= s0:
                continue

            if file_idx not in self._idx_to_path:
                # Zero-fill missing file
                ntime_fill = s1 - s0
                if verbose:
                    print(f"  File .{file_idx} MISSING — zero-filling {ntime_fill} integrations")
                v = np.zeros((ntime_fill, fmt.nchan, n_output), dtype=np.complex64)
                t_file_start = t0_unix + file_idx * file_dur_s
                t_local = t_file_start + fmt.dt_raw_s * np.arange(
                    fmt.ntime_per_file, dtype=np.float64
                )
                t_sel = t_local[s0:s1]
                vis_chunks.append(v)
                time_chunks.append(t_sel)
                kept_files.append(f"MISSING.{file_idx}")
                continue

            fpath = self._idx_to_path[file_idx]

            offset, file_header = get_header_offset(fpath)
            if file_header is not None:
                file_headers[file_idx] = file_header
                # Cross-validate header vs format when fmt was explicitly passed
                if self._fmt_explicit:
                    h_nchan = int(file_header.get("NCHAN", 0))
                    h_nbl = int(file_header.get("NBASELINE", 0))
                    if h_nchan and h_nchan != fmt.nchan:
                        warnings.warn(
                            f"File .{file_idx} header NCHAN={h_nchan} != fmt.nchan={fmt.nchan}",
                            stacklevel=2,
                        )
                    if h_nbl and h_nbl != fmt.n_baselines:
                        warnings.warn(
                            f"File .{file_idx} header NBASELINE={h_nbl} != fmt.n_baselines={fmt.n_baselines}",
                            stacklevel=2,
                        )

            raw = np.fromfile(fpath, dtype=np.int32, offset=offset)
            if raw.size % denom != 0:
                raise ValueError(
                    f"File {fpath} has {raw.size} int32s, not divisible by {denom}"
                )

            ntime = raw.size // denom
            if ntime == 0:
                raise ValueError(
                    f"File {fpath} too small for even one integration"
                )
            if ntime != fmt.ntime_per_file and verbose:
                print(f"  {os.path.basename(fpath)}: ntime={ntime} "
                      f"(expected {fmt.ntime_per_file})")

            # Cap slice to actual integrations available
            s1 = min(s1, ntime)

            xcorrs = raw.reshape(ntime, fmt.nchan, nbaseline, 2)
            del raw
            xcorrs = xcorrs[s0:s1, :, :, :]

            if extract_specific:
                xsel = xcorrs[:, :, bl_idx, :]
                del xcorrs
                v = xsel[..., 0].astype(np.float32) + 1j * xsel[..., 1].astype(np.float32)
                del xsel
                if np.any(bl_conj):
                    v[:, :, bl_conj] = np.conj(v[:, :, bl_conj])
            else:
                v = xcorrs[..., 0].astype(np.float32) + 1j * xcorrs[..., 1].astype(np.float32)
                del xcorrs

            # Apply frequency ordering
            if freq_order == "ascending":
                v = v[:, ::-1, :]

            # Build times
            t_file_start = t0_unix + file_idx * file_dur_s
            t_local = t_file_start + fmt.dt_raw_s * np.arange(
                ntime, dtype=np.float64
            )
            t_sel = t_local[s0:s1]

            vis_chunks.append(v.astype(np.complex64))
            time_chunks.append(t_sel)
            kept_files.append(os.path.basename(fpath))
            gc.collect()

        if not vis_chunks:
            raise RuntimeError("No data collected after slicing")

        vis = np.concatenate(vis_chunks, axis=0)
        time_unix = np.concatenate(time_chunks)
        del vis_chunks, time_chunks
        gc.collect()

        if verbose:
            print(f"Output shape: {vis.shape} ({vis.dtype})")
            print(f"Frequency: {freq_mhz[0]:.3f} -> {freq_mhz[-1]:.3f} MHz ({freq_order})")
            print(f"Time samples: {len(time_unix)}")

        metadata = dict(
            format_name=fmt.name,
            nsig=fmt.nsig,
            dt_raw_s=fmt.dt_raw_s,
            nchan=fmt.nchan,
            base_str=self._base_str,
            data_dir=self._data_dir,
            files=kept_files,
            freq_order=freq_order,
            missing_files=missing,
            file_headers=file_headers,
        )

        if extract_specific:
            metadata["ref"] = ref
            metadata["targets"] = targets
            metadata["baseline_convention"] = "V(ref,target) with conjugation when ref>target"

        return VisibilityResult(
            vis=vis,
            freq_mhz=freq_mhz,
            time_unix=time_unix,
            metadata=metadata,
        )
