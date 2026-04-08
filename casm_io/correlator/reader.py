"""
Correlator visibility file reader.

Reads raw .dat visibility files from the CASM correlator.
Files are named like: {base_str}.dat.{index} where index is 0, 1, 2, ...
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


# Regex to extract base_str from filenames like "2026-03-27-07:56:53.dat.0"
_BASE_STR_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2})")


def _parse_time(spec, tz_obj):
    """Parse a time specification to a timezone-aware datetime."""
    if spec is None:
        return None
    if isinstance(spec, str):
        dt = datetime.fromisoformat(spec)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz_obj)
        return dt
    return spec.replace(tzinfo=tz_obj) if spec.tzinfo is None else spec


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m"
    elif seconds < 86400:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m" if m else f"{h}h"
    else:
        d = int(seconds // 86400)
        h = int((seconds % 86400) // 3600)
        return f"{d}d {h}h" if h else f"{d}d"


def _resolve_channels(
    fmt: VisibilityFormat,
    channels: tuple[int, int] | None,
    freq_range_mhz: tuple[float, float] | None,
) -> tuple[int, int] | None:
    """
    Resolve channel slicing parameters to (ch_start, ch_end) in native order.

    Returns None if no slicing requested.
    """
    if channels is not None and freq_range_mhz is not None:
        raise ValueError("channels and freq_range_mhz are mutually exclusive")
    if channels is not None:
        ch_start, ch_end = channels
        if ch_start < 0 or ch_end > fmt.nchan or ch_start >= ch_end:
            raise ValueError(
                f"Invalid channel range ({ch_start}, {ch_end}). "
                f"Must be 0 <= start < end <= {fmt.nchan}"
            )
        return ch_start, ch_end
    if freq_range_mhz is not None:
        return fmt.freq_range_to_channels(freq_range_mhz[0], freq_range_mhz[1])
    return None


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


def discover_observations(
    data_dir: str,
    fmt: VisibilityFormat | None = None,
    verbose: bool = False,
) -> list[dict]:
    """
    Scan a data directory and return metadata for each observation.

    Parameters
    ----------
    data_dir : str
        Directory containing visibility .dat files.
    fmt : VisibilityFormat, optional
        Format for headerless files. If None, format is auto-detected
        from file headers. Observations without headers and no fmt
        are skipped with a warning.

    Returns
    -------
    list of dict
        Each dict has keys: 'base_str', 'n_files', 'time_start',
        'time_end', 'fmt', 'data_dir'. Sorted by time_start.
    """
    # Find all unique base_strs
    try:
        all_files = os.listdir(data_dir)
    except FileNotFoundError:
        return []

    base_strs = set()
    for fname in all_files:
        m = _BASE_STR_RE.match(fname)
        if m:
            base_strs.add(m.group(1))

    observations = []
    for bs in sorted(base_strs):
        idx_to_path = discover_files(data_dir, bs)
        if not idx_to_path:
            continue

        # Determine format
        obs_fmt = fmt
        if obs_fmt is None:
            first_path = idx_to_path[min(idx_to_path)]
            offset, header = get_header_offset(first_path)
            if header is not None:
                try:
                    obs_fmt = format_from_header(header)
                except (KeyError, ValueError) as e:
                    if verbose:
                        print(f"  Skipping {bs}: failed to parse header ({e})")
                    continue
            else:
                if verbose:
                    print(f"  Skipping {bs}: no header (pre-March 4 file)")
                continue

        max_idx = max(idx_to_path.keys())
        t0_unix = (
            datetime.strptime(bs, "%Y-%m-%d-%H:%M:%S")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        t_end_unix = t0_unix + (max_idx + 1) * obs_fmt.file_duration_s

        observations.append(dict(
            base_str=bs,
            n_files=len(idx_to_path),
            time_start=t0_unix,
            time_end=t_end_unix,
            fmt=obs_fmt,
            data_dir=data_dir,
        ))

    observations.sort(key=lambda o: o["time_start"])
    return observations


def read_visibilities(
    time_start: str | datetime,
    time_end: str | datetime,
    time_tz: str = "UTC",
    data_root: str = "/mnt",
    data_dir: str | None = None,
    fmt: VisibilityFormat | None = None,
    ref: int | None = None,
    targets: list[int] | None = None,
    freq_order: str = "descending",
    channels: tuple[int, int] | None = None,
    freq_range_mhz: tuple[float, float] | None = None,
    verbose: bool = True,
) -> VisibilityResult:
    """
    Read visibility data by time range with automatic data discovery.

    Scans data directories to find observations overlapping the requested
    time range, reads and stitches them together. Warns about gaps.

    Parameters
    ----------
    time_start : str or datetime
        Start of desired time range (ISO format or datetime).
    time_end : str or datetime
        End of desired time range.
    time_tz : str
        Timezone for time_start/time_end (default 'UTC').
        Supports any timezone (e.g. 'America/Los_Angeles').
    data_root : str
        Base path to scan for visibilities_* subdirectories.
        Ignored if data_dir is provided.
    data_dir : str, optional
        Explicit data directory. If given, bypasses auto-scan of data_root.
    fmt : VisibilityFormat, optional
        Format for headerless files. If None, auto-detected from headers.
    ref : int, optional
        Reference input index for baseline extraction.
    targets : list of int, optional
        Target input indices.
    freq_order : str
        'descending' (default, native) or 'ascending'.
    channels : tuple of (int, int), optional
        (ch_start, ch_end) channel range in native descending order.
        Exclusive end. Mutually exclusive with freq_range_mhz.
    freq_range_mhz : tuple of (float, float), optional
        (freq_lo, freq_hi) frequency range in MHz.
        Mutually exclusive with channels.
    verbose : bool
        Print progress and diagnostic messages.

    Returns
    -------
    VisibilityResult
        Concatenated result. metadata['observations'] lists base_strs used.
        metadata['gaps'] lists any data gaps within the requested range.
    """
    _LOCAL_TZ = "America/Los_Angeles"
    tz_obj = ZoneInfo(time_tz) if time_tz != "UTC" else timezone.utc

    # Parse time range
    utc_start_dt = _parse_time(time_start, tz_obj)
    utc_end_dt = _parse_time(time_end, tz_obj)
    if utc_start_dt is None or utc_end_dt is None:
        raise ValueError("Both time_start and time_end are required")
    utc_start_dt = utc_start_dt.astimezone(timezone.utc)
    utc_end_dt = utc_end_dt.astimezone(timezone.utc)
    req_start_unix = utc_start_dt.timestamp()
    req_end_unix = utc_end_dt.timestamp()

    if req_start_unix >= req_end_unix:
        raise ValueError("time_start must be before time_end")

    if verbose:
        print(f"Requested (UTC): {format_time_span(req_start_unix, req_end_unix)}")
        print(f"Requested (PT):  {format_time_span(req_start_unix, req_end_unix, _LOCAL_TZ)}")
        if time_tz != "UTC":
            print(f"Requested ({time_tz}): "
                  f"{utc_start_dt.astimezone(tz_obj).strftime('%Y-%m-%d %H:%M:%S')} -> "
                  f"{utc_end_dt.astimezone(tz_obj).strftime('%Y-%m-%d %H:%M:%S')}")

    # Discover data directories
    if data_dir is not None:
        data_dirs = [data_dir]
        if verbose:
            print(f"Using data directory: {data_dir}")
    else:
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"Data root not found: {data_root}")
        data_dirs = sorted(glob.glob(os.path.join(data_root, "**/visibilities_*"), recursive=True))
        # Only keep actual directories
        data_dirs = [d for d in data_dirs if os.path.isdir(d)]
        if not data_dirs:
            raise FileNotFoundError(
                f"No visibilities_* directories found under {data_root}"
            )
        if verbose:
            print(f"Scanning {data_root} ...")
            for d in data_dirs:
                print(f"  Found: {os.path.basename(d)}/")

    # Discover all observations across all data dirs
    all_observations = []
    for ddir in data_dirs:
        obs = discover_observations(ddir, fmt=fmt)
        all_observations.extend(obs)
        if verbose and obs:
            print(f"  {os.path.basename(ddir)}/: {len(obs)} observations "
                  f"({obs[0]['base_str']} -> {obs[-1]['base_str']})")

    if not all_observations:
        raise RuntimeError(
            f"No observations found in {data_dirs}. "
            f"Check that data files exist and have valid headers."
        )

    all_observations.sort(key=lambda o: o["time_start"])

    # Filter observations overlapping [req_start, req_end]
    matching = [
        o for o in all_observations
        if o["time_end"] > req_start_unix and o["time_start"] < req_end_unix
    ]

    if not matching:
        # Build helpful error message with nearest observations
        nearest_before = [o for o in all_observations if o["time_end"] <= req_start_unix]
        nearest_after = [o for o in all_observations if o["time_start"] >= req_end_unix]
        msg = (
            f"No observations found overlapping requested time range.\n"
            f"Requested: {format_time_span(req_start_unix, req_end_unix)}\n"
        )
        if nearest_before:
            nb = nearest_before[-1]
            msg += f"Nearest before: {nb['base_str']} (ends {format_time_span(nb['time_start'], nb['time_end'])})\n"
        if nearest_after:
            na = nearest_after[0]
            msg += f"Nearest after:  {na['base_str']} (starts {format_time_span(na['time_start'], na['time_end'])})\n"
        raise ValueError(msg)

    # Verify consistent frequency setup across matching observations
    ref_fmt = matching[0]["fmt"]
    for obs in matching[1:]:
        ofmt = obs["fmt"]
        if (abs(ofmt.freq_top_mhz - ref_fmt.freq_top_mhz) > 0.001 or
                abs(ofmt.chan_bw_mhz - ref_fmt.chan_bw_mhz) > 0.001 or
                ofmt.nchan != ref_fmt.nchan):
            raise ValueError(
                f"Frequency mismatch between observations.\n"
                f"  {matching[0]['base_str']}: {ref_fmt.freq_top_mhz:.3f} MHz, "
                f"{ref_fmt.nchan} ch, {ref_fmt.chan_bw_mhz:.6f} MHz/ch\n"
                f"  {obs['base_str']}: {ofmt.freq_top_mhz:.3f} MHz, "
                f"{ofmt.nchan} ch, {ofmt.chan_bw_mhz:.6f} MHz/ch\n"
                f"Cannot stitch observations with different frequency setups.\n"
                f"Read them separately with VisibilityReader."
            )

    if verbose:
        print(f"\nMatching observations: {len(matching)}")
        for i, obs in enumerate(matching, 1):
            print(f"  [{i}] {obs['base_str']}  ({obs['n_files']} files)")
            print(f"      UTC: {format_time_span(obs['time_start'], obs['time_end'])}")
            print(f"      PT:  {format_time_span(obs['time_start'], obs['time_end'], _LOCAL_TZ)}")

    # Detect gaps between matching observations and with request boundaries
    gaps = []
    # Gap at start: requested start is before first observation
    if matching[0]["time_start"] > req_start_unix:
        gap_start = req_start_unix
        gap_end = matching[0]["time_start"]
        gaps.append(dict(
            start_unix=gap_start, end_unix=gap_end,
            duration_s=gap_end - gap_start,
        ))

    # Gaps between consecutive observations
    for i in range(len(matching) - 1):
        gap_start = matching[i]["time_end"]
        gap_end = matching[i + 1]["time_start"]
        if gap_end > gap_start + 1.0:  # >1s gap
            gaps.append(dict(
                start_unix=gap_start, end_unix=gap_end,
                duration_s=gap_end - gap_start,
            ))

    # Gap at end: requested end is after last observation
    if matching[-1]["time_end"] < req_end_unix:
        gap_start = matching[-1]["time_end"]
        gap_end = req_end_unix
        gaps.append(dict(
            start_unix=gap_start, end_unix=gap_end,
            duration_s=gap_end - gap_start,
        ))

    if verbose and gaps:
        print(f"\nWARNING: {len(gaps)} gap(s) in requested time range:")
        for g in gaps:
            print(f"  No data (UTC): {format_time_span(g['start_unix'], g['end_unix'])} "
                  f"(duration: {_format_duration(g['duration_s'])})")
            print(f"  No data (PT):  {format_time_span(g['start_unix'], g['end_unix'], _LOCAL_TZ)}")

    # Read each matching observation
    vis_chunks = []
    time_chunks = []
    all_metadata_files = []
    all_file_headers = {}

    for obs in matching:
        # Clip time range to this observation's bounds
        clip_start = max(req_start_unix, obs["time_start"])
        clip_end = min(req_end_unix, obs["time_end"])

        clip_start_dt = datetime.fromtimestamp(clip_start, tz=timezone.utc)
        clip_end_dt = datetime.fromtimestamp(clip_end, tz=timezone.utc)

        if verbose:
            print(f"\nReading {obs['base_str']} "
                  f"({format_time_span(clip_start, clip_end)}) ...")

        reader = VisibilityReader(obs["data_dir"], obs["base_str"], fmt=obs["fmt"])
        result = reader.read(
            time_start=clip_start_dt,
            time_end=clip_end_dt,
            time_tz="UTC",
            ref=ref,
            targets=targets,
            freq_order=freq_order,
            channels=channels,
            freq_range_mhz=freq_range_mhz,
            verbose=verbose,
        )

        vis_chunks.append(result.vis)
        time_chunks.append(result.time_unix)
        all_metadata_files.extend(result.metadata.get("files", []))
        all_file_headers.update(result.metadata.get("file_headers", {}))

    # Concatenate
    vis = np.concatenate(vis_chunks, axis=0)
    time_unix = np.concatenate(time_chunks)
    freq_mhz = result.freq_mhz  # same across all observations (verified above)
    del vis_chunks, time_chunks
    gc.collect()

    if verbose:
        print(f"\nFinal output shape: {vis.shape} ({vis.dtype})")
        print(f"Frequency: {freq_mhz[0]:.3f} -> {freq_mhz[-1]:.3f} MHz ({freq_order})")
        print(f"Time samples: {len(time_unix)}")
        print(f"Time span (UTC): {format_time_span(time_unix[0], time_unix[-1])}")
        print(f"Time span (PT):  {format_time_span(time_unix[0], time_unix[-1], _LOCAL_TZ)}")

    metadata = dict(
        format_name=ref_fmt.name,
        nsig=ref_fmt.nsig,
        dt_raw_s=ref_fmt.dt_raw_s,
        nchan=ref_fmt.nchan,
        freq_order=freq_order,
        observations=[o["base_str"] for o in matching],
        data_dirs=list(set(o["data_dir"] for o in matching)),
        files=all_metadata_files,
        file_headers=all_file_headers,
        gaps=gaps,
        missing_files=[],
    )
    if channels is not None:
        metadata["channels"] = channels
    if freq_range_mhz is not None:
        metadata["freq_range_mhz"] = freq_range_mhz
    if ref is not None:
        metadata["ref"] = ref
        metadata["targets"] = targets
        metadata["baseline_convention"] = "V(ref,target) with conjugation when ref>target"

    return VisibilityResult(
        vis=vis,
        freq_mhz=freq_mhz,
        time_unix=time_unix,
        metadata=metadata,
    )


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

        # Read header from first file (used for auto-detect and freq override)
        first_path = self._idx_to_path[min(self._idx_to_path)]
        offset, header = get_header_offset(first_path)

        # Auto-detect format from header if fmt not provided; otherwise
        # check header FREQ_START against the explicit format config and
        # override if they disagree (handles band-shift scenarios).
        if fmt is None:
            if header is not None:
                fmt = format_from_header(header)
            else:
                raise ValueError(
                    "No header found in data files and no fmt provided. "
                    "Pass a VisibilityFormat via load_format() for old files."
                )
        elif header is not None and "FREQ_START" in header:
            # Override freq_top_mhz from file header when it differs from
            # the JSON config value (e.g., after the March 27 2026 band shift).
            header_freq_top = float(header["FREQ_START"])
            if abs(header_freq_top - fmt.freq_top_mhz) > 0.001:
                warnings.warn(
                    f"File header FREQ_START={header_freq_top:.4f} MHz differs "
                    f"from format config freq_top_mhz={fmt.freq_top_mhz:.4f} MHz. "
                    f"Using file header value.",
                    stacklevel=2,
                )
                fmt = VisibilityFormat(
                    name=fmt.name,
                    nsig=fmt.nsig,
                    dt_raw_s=fmt.dt_raw_s,
                    ntime_per_file=fmt.ntime_per_file,
                    nchan=fmt.nchan,
                    chan_bw_mhz=fmt.chan_bw_mhz,
                    freq_top_mhz=header_freq_top,
                    freq_bottom_mhz=header_freq_top - fmt.nchan * fmt.chan_bw_mhz,
                    native_order=fmt.native_order,
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
        channels: tuple[int, int] | None = None,
        freq_range_mhz: tuple[float, float] | None = None,
        verbose: bool = True,
    ) -> VisibilityResult:
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
        channels : tuple of (int, int), optional
            (ch_start, ch_end) channel range in native descending order.
            Exclusive end. Mutually exclusive with freq_range_mhz.
        freq_range_mhz : tuple of (float, float), optional
            (freq_lo, freq_hi) frequency range in MHz.
            Mutually exclusive with channels.
        verbose : bool
            Print progress messages.

        Returns
        -------
        VisibilityResult with keys:
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

        # Resolve channel slicing
        ch_slice = _resolve_channels(fmt, channels, freq_range_mhz)
        use_memmap = ch_slice is not None
        if ch_slice is not None:
            ch_start, ch_end = ch_slice
            nchan_read = ch_end - ch_start
        else:
            ch_start, ch_end = 0, fmt.nchan
            nchan_read = fmt.nchan

        tz_obj = ZoneInfo(time_tz) if time_tz != "UTC" else timezone.utc

        utc_start_dt = _parse_time(time_start, tz_obj)
        utc_end_dt = _parse_time(time_end, tz_obj)
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

            if ch_slice is not None:
                freq_full = fmt.get_frequency_axis(order="descending")
                print(f"Channel slicing: [{ch_start}:{ch_end}] "
                      f"({freq_full[ch_start]:.3f} -> {freq_full[ch_end - 1]:.3f} MHz, "
                      f"{nchan_read} channels)")

            # Timezone echo: show local and UTC times when non-UTC timezone used
            if time_tz != "UTC" and (time_start is not None or time_end is not None):
                local_start = _parse_time(time_start, tz_obj) if time_start else None
                local_end = _parse_time(time_end, tz_obj) if time_end else None
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
                if utc_start_unix < data_start_unix - 1.0 or utc_end_unix > data_end_unix + 1.0:
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

        # Build frequency axis (with channel slicing)
        freq_mhz_full = fmt.get_frequency_axis(order="descending")
        freq_mhz_sliced = freq_mhz_full[ch_start:ch_end]
        if freq_order == "ascending":
            freq_mhz = freq_mhz_sliced[::-1].copy()
        else:
            freq_mhz = freq_mhz_sliced

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
                v = np.zeros((ntime_fill, nchan_read, n_output), dtype=np.complex64)
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

            if use_memmap:
                # Memory-mapped read: only load requested channels
                file_size = os.path.getsize(fpath)
                data_bytes = file_size - offset
                ntime = data_bytes // (fmt.nchan * nbaseline * 2 * 4)
                if ntime == 0:
                    raise ValueError(f"File {fpath} too small for even one integration")

                mm = np.memmap(
                    fpath, dtype=np.int32, mode='r', offset=offset,
                    shape=(ntime, fmt.nchan, nbaseline, 2),
                )
                # Cap slice to actual integrations available
                s1_capped = min(s1, ntime)
                xcorrs = np.array(mm[s0:s1_capped, ch_start:ch_end, :, :])
                del mm
            else:
                # Full read path (unchanged from original)
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

                # Cap slice to actual integrations available
                s1_capped = min(s1, ntime)

                xcorrs = raw.reshape(ntime, fmt.nchan, nbaseline, 2)
                del raw
                xcorrs = xcorrs[s0:s1_capped, :, :, :]

            if ntime != fmt.ntime_per_file and verbose:
                print(f"  {os.path.basename(fpath)}: ntime={ntime} "
                      f"(expected {fmt.ntime_per_file})")

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
            t_sel = t_local[s0:s1_capped]

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

        if ch_slice is not None:
            metadata["channels"] = (ch_start, ch_end)
        if freq_range_mhz is not None:
            metadata["freq_range_mhz"] = freq_range_mhz
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
