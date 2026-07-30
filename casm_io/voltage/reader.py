"""
Voltage dump DADA file reader.

Supports:
- Single subband files (1024 channels)
- Full 3-subband dumps (3072 channels, stitched)
- Triggered dumps written to per-stream directories (stream_0 ... stream_5,
  512 channels each, one dump possibly split over several files)
- Optional per-antenna extraction via antenna mapping CSV

Stream layout notes (verified against March 2026 dumps in cand_dumps/)
---------------------------------------------------------------------
Channel order: the same as the legacy layout — descending. Every stream
header carries BW -15.625 and CHANBW -0.030517578125 (both negative), and
the trusted FREQ field falls with stream index (460.9375, 445.3125,
429.6875 MHz for streams 0, 1, 2). So channel 0 of stream 0 is the top of
the band and stream_N runs downwards in frequency, exactly like
chan0_1023 / chan1024_2047 / chan2048_3071. The bandpass confirms it: the
last channels of stream_1 and the first channels of stream_2 form one
continuous narrowband RFI line, and the stream_0/stream_1 join is smooth
to 0.2% of the per-channel scatter.

Equivalently, in ascending raw PFB channels (freq = pfb_freq_bottom_mhz +
raw * chan_bw_mhz, 4096 channels from 375.0 MHz), channel c of stream i is
raw channel raw_chan_offset + n_chan_total - (i * n_chan_per_subband + c).
The frequency axis is built from the stream index and the config, never
from the header.

FREQ_START / START_CHANNEL quirks: FREQ_START is off-grid for stream 0
(468.45 where the grid says 468.75 — the other streams are exact), so it
is not usable. START_CHANNEL is a descending channel index counted from
raw channel n_chan_pfb (500.0 MHz), i.e. 1024 higher than the legacy
in-band channel index; read_subband cross-checks it and warns when it
disagrees with the config, which it does by exactly one sub-band width
(15.625 MHz) for data taken before the 2026-03-27 band shift.

Sizing: RESOLUTION (bytes per time sample) varies between epochs — 67584
for 11 SNAP slots, 264192 for the 43-slot March 4 configuration — so it is
read per file. FILE_SIZE is a pre-allocated ~10 s span; DUMP_BYTES gives
the real payload and everything past it is zeros.
"""

import json
import os
import re

import numpy as np
import pandas as pd

from .header import parse_dada_header, HEADER_SIZE
from .unpack import unpack_4bit
from .._progress import print_progress
from .._results import SubbandResult, FullBandResult

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _load_dada_config(name: str = "dada_format.json") -> dict:
    """Load a DADA format config by file name."""
    config_path = os.path.join(_CONFIGS_DIR, name)
    with open(config_path) as f:
        return json.load(f)


def _select_dada_config(data_dir: str) -> dict:
    """Pick the format config matching the sub-band directories on disk.

    Triggered dumps are written to stream_0 ... stream_5; older dumps use
    chan0_1023 / chan1024_2047 / chan2048_3071.
    """
    stream_cfg = _load_dada_config("dada_format_stream.json")
    for sub_dir in stream_cfg["subband_dirs"]:
        if os.path.isdir(os.path.join(data_dir, sub_dir)):
            return stream_cfg
    return _load_dada_config()


def _int_or_none(value) -> int | None:
    """Parse a header value as int, returning None if it isn't one."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _dump_sort_key(path: str) -> tuple:
    """Sort key for the files of one dump.

    Files are named <UTC_START>_<16-digit OBS_OFFSET>.<6-digit number>.dada,
    where OBS_OFFSET is bytes since UTC_START, so sorting on it puts the
    files in time order. Names that don't match sort by name.
    """
    name = os.path.basename(path)
    m = re.search(r"_(\d+)\.(\d+)\.dada$", name)
    if m:
        return (int(m.group(1)), int(m.group(2)), name)
    return (-1, -1, name)


def _file_time_layout(filename: str, cfg: dict, header: dict) -> tuple[int, int]:
    """Bytes per time sample and number of valid time samples in one file.

    RESOLUTION is read per file because it changed between epochs. The valid
    payload is DUMP_BYTES when present (the file is pre-allocated longer and
    zero-padded past it), otherwise the file size less the header, floored to
    a whole number of RESOLUTION blocks.
    """
    n_chan_sub = cfg["n_chan_per_subband"]
    n_adc = cfg["n_adc_per_snap"]
    bytes_per_time = cfg["n_snap_slots"] * n_chan_sub * n_adc

    resolution = _int_or_none(header.get("RESOLUTION"))
    if resolution and resolution % (n_chan_sub * n_adc) == 0:
        bytes_per_time = resolution

    payload = os.path.getsize(filename) - HEADER_SIZE
    dump_bytes = _int_or_none(header.get("DUMP_BYTES"))
    if dump_bytes is not None:
        payload = min(payload, dump_bytes)

    return bytes_per_time, max(payload, 0) // bytes_per_time


def _make_freq_axis_subband(
    subband_index: int, n_chan: int, freq_top_mhz: float,
    chan_bw_mhz: float, order: str,
) -> np.ndarray:
    """Build frequency axis for one subband.

    Sub-band 0 is the top of the band for both layouts, so channel c of
    sub-band i sits at freq_top_mhz - (i * n_chan + c) * chan_bw_mhz.
    """
    start_chan = subband_index * n_chan
    freqs_desc = (
        freq_top_mhz
        - chan_bw_mhz * np.arange(start_chan, start_chan + n_chan, dtype=np.float64)
    )
    if order == "ascending":
        return freqs_desc[::-1].copy()
    return freqs_desc


def _guess_subband_index(filename: str) -> int:
    """Guess subband index (0=highest freq) from filename/path."""
    path_lower = filename.lower()
    if "chan0_1023" in path_lower:
        return 0
    elif "chan1024_2047" in path_lower:
        return 1
    elif "chan2048_3071" in path_lower:
        return 2
    m = re.search(r"stream_(\d+)", path_lower)
    if m:
        return int(m.group(1))
    return 0  # default


class VoltageReader:
    """
    Voltage dump DADA reader with config and subband discovery on init.

    Parameters
    ----------
    data_dir : str
        Base directory containing subband subdirectories — either
        chan0_1023/, chan1024_2047/, chan2048_3071/ (legacy) or
        stream_0/ ... stream_5/ (triggered dumps).
    timestamp : str
        UTC timestamp string (e.g. '2026-02-17-21:10:43'). Matched as a
        filename prefix, so it can be extended with the OBS_OFFSET field
        ('2026-03-16-00:30:11_0000044756219904') to pick one dump when a
        stream directory holds several dumps with the same UTC_START.
    config : str or dict, optional
        Format config to use, as a path to a JSON file or an already
        loaded dict. Default: chosen from the subdirectories in data_dir.
    """

    def __init__(self, data_dir: str, timestamp: str, config: str | dict | None = None):
        self._data_dir = data_dir
        self._timestamp = timestamp
        if config is None:
            self._cfg = _select_dada_config(data_dir)
        elif isinstance(config, dict):
            self._cfg = config
        else:
            with open(config) as f:
                self._cfg = json.load(f)
        self._stream_layout = self._cfg.get("layout") == "stream"

        # Discover which subbands are available. The legacy layout has one
        # file per subband; a stream dump can be split over several files.
        self._subband_files = {}
        for i, sub_dir in enumerate(self._cfg["subband_dirs"]):
            sub_path = os.path.join(data_dir, sub_dir)
            if os.path.isdir(sub_path):
                matches = [
                    os.path.join(sub_path, f) for f in os.listdir(sub_path)
                    if f.startswith(timestamp) and f.endswith(".dada")
                ]
                if matches:
                    if self._stream_layout:
                        self._subband_files[i] = sorted(matches, key=_dump_sort_key)
                    else:
                        self._subband_files[i] = sorted(matches)[0]

    @property
    def subbands_found(self) -> list[int]:
        """List of discovered subband indices."""
        return sorted(self._subband_files.keys())

    def read_subband(
        self,
        index: int,
        n_time: int | None = None,
        snaps: list[int] | None = None,
        freq_order: str = "descending",
        trust_header: bool = False,
        verbose: bool = True,
        allow_gaps: bool = False,
    ) -> dict:
        """
        Read a single DADA subband file.

        Parameters
        ----------
        index : int
            Subband index (0-2 legacy, 0-5 stream layout).
        n_time : int, optional
            Number of time samples to read from the start of the dump.
            None = all. For the stream layout this counts across the files
            the dump is split into.
        snaps : list of int, optional
            SNAP slot indices to extract. Default: active_snaps from config.
        freq_order : str
            'descending' (default, native) or 'ascending'.
        trust_header : bool
            If True, trust all header fields.
        verbose : bool
            Print progress.
        allow_gaps : bool
            Stream layout only. If the files matched by the timestamp prefix
            do not run end to end in OBS_OFFSET, the read raises by default —
            they are usually separate dumps. Set True to stitch across the
            gap (or overlap) with a warning instead.

        Returns
        -------
        dict with keys:
            voltages : dict
                {snap_id: (n_time, n_chan_per_sub, n_adc) complex64}
            header : dict
                Parsed DADA header.
            freq_mhz : np.ndarray
                Frequency axis for this subband.
        """
        if index not in self._subband_files:
            available = self.subbands_found
            raise FileNotFoundError(
                f"Subband {index} not found. Available: {available}"
            )

        cfg = self._cfg
        if snaps is None:
            snaps = cfg["active_snaps"]

        if self._stream_layout:
            voltages, header = self._read_stream_subband(
                index, n_time, snaps, verbose, allow_gaps,
            )
            if freq_order == "ascending":
                for s in snaps:
                    voltages[s] = voltages[s][:, ::-1, :]
            freq_mhz = _make_freq_axis_subband(
                index, cfg["n_chan_per_subband"], cfg["freq_top_mhz"],
                cfg["chan_bw_mhz"], freq_order,
            )
            return SubbandResult(voltages=voltages, header=header, freq_mhz=freq_mhz)

        filename = self._subband_files[index]
        n_snaps = cfg["n_snap_slots"]
        n_adc = cfg["n_adc_per_snap"]
        n_chan_sub = cfg["n_chan_per_subband"]

        header = parse_dada_header(filename)
        file_size = os.path.getsize(filename)
        data_size = file_size - HEADER_SIZE
        bytes_per_time = n_snaps * n_chan_sub * n_adc
        n_time_total = data_size // bytes_per_time

        if n_time is None:
            n_time = n_time_total
        else:
            n_time = min(n_time, n_time_total)

        if verbose:
            print(f"Reading {filename}")
            print(f"  n_time={n_time} / {n_time_total}, SNAPs={snaps}")

        n_bytes = n_time * bytes_per_time
        with open(filename, "rb") as f:
            f.seek(HEADER_SIZE)
            raw = np.frombuffer(f.read(n_bytes), dtype=np.uint8)

        raw = raw.reshape(n_time, n_snaps, n_chan_sub, n_adc)

        voltages = {}
        for i, s in enumerate(snaps):
            if verbose:
                print_progress(i + 1, len(snaps), prefix="Unpacking SNAPs")
            voltages[s] = unpack_4bit(raw[:, s, :, :])
            if freq_order == "ascending":
                voltages[s] = voltages[s][:, ::-1, :]

        freq_mhz = _make_freq_axis_subband(
            index, n_chan_sub, cfg["freq_top_mhz"],
            cfg["chan_bw_mhz"], freq_order,
        )

        return SubbandResult(voltages=voltages, header=header, freq_mhz=freq_mhz)

    def _check_stream_header(self, index: int, filename: str, header: dict) -> None:
        """Print any disagreement between a stream header and the config.

        STREAM_SUBBAND_ID should match the stream_N directory and NCHAN the
        config. START_CHANNEL counts channels down from raw channel
        n_chan_pfb, so it implies a top frequency for the sub-band; it is one
        sub-band width low for data taken before the 2026-03-27 band shift.
        """
        cfg = self._cfg
        subband_id = _int_or_none(header.get("STREAM_SUBBAND_ID"))
        if subband_id is not None and subband_id != index:
            print(f"  WARNING: {filename} has STREAM_SUBBAND_ID {subband_id}, "
                  f"expected {index}")

        n_chan_hdr = _int_or_none(header.get("NCHAN"))
        if n_chan_hdr is not None and n_chan_hdr != cfg["n_chan_per_subband"]:
            print(f"  WARNING: {filename} has NCHAN {n_chan_hdr}, config says "
                  f"{cfg['n_chan_per_subband']}")

        start_chan = _int_or_none(header.get("START_CHANNEL"))
        if start_chan is None:
            return
        chan_bw = cfg["chan_bw_mhz"]
        raw_chan = cfg["n_chan_pfb"] - start_chan
        hdr_top = cfg["pfb_freq_bottom_mhz"] + raw_chan * chan_bw
        cfg_top = cfg["freq_top_mhz"] - index * cfg["n_chan_per_subband"] * chan_bw
        if abs(hdr_top - cfg_top) > 0.5 * chan_bw:
            print(f"  NOTE: START_CHANNEL {start_chan} implies a sub-band top of "
                  f"{hdr_top:.4f} MHz, config gives {cfg_top:.4f} MHz "
                  f"({(cfg_top - hdr_top) / chan_bw:.0f} channels apart) — "
                  f"expected for data taken before the 2026-03-27 band shift. "
                  f"Frequencies come from the config.")

    def _check_stream_alignment(self) -> None:
        """Check that every stream present starts at the same OBS_OFFSET.

        The streams of one dump are byte-identical in OBS_OFFSET, so a
        mismatch means the timestamp prefix picked up different dumps in
        different stream directories — likely when one node's disk was too
        full to write and a later trigger filled the gap.
        """
        offsets = {}
        for index, files in self._subband_files.items():
            header = parse_dada_header(files[0])
            offset = _int_or_none(header.get("OBS_OFFSET"))
            if offset is not None:
                offsets[index] = offset

        if len(set(offsets.values())) > 1:
            detail = ", ".join(
                f"{self._cfg['subband_dirs'][i]}: OBS_OFFSET {offsets[i]}"
                for i in sorted(offsets)
            )
            raise ValueError(
                f"Streams for {self._timestamp} do not start at the same time "
                f"({detail}). The streams of one dump share an OBS_OFFSET, so "
                f"these are different dumps — pass the full "
                f"UTC_START_OBSOFFSET filename prefix to select one."
            )

    def _read_stream_subband(
        self, index: int, n_time: int | None, snaps: list[int], verbose: bool,
        allow_gaps: bool = False,
    ) -> tuple[dict, dict]:
        """Read one stream sub-band, stitching the files of the dump in time.

        Files are taken in OBS_OFFSET order and treated as one timeline, so
        n_time counts from the start of the dump. OBS_OFFSET is bytes since
        UTC_START, so the files of one dump run end to end; a nonzero gap (or
        overlap) means the prefix matched more than one dump, which raises
        unless allow_gaps=True. Returns (voltages, header) with the header
        from the first file.
        """
        cfg = self._cfg
        n_chan_sub = cfg["n_chan_per_subband"]
        n_adc = cfg["n_adc_per_snap"]
        files = self._subband_files[index]

        layouts = []
        prev_end = None
        for filename in files:
            header = parse_dada_header(filename)
            self._check_stream_header(index, filename, header)
            bytes_per_time, n_time_file = _file_time_layout(filename, cfg, header)
            offset = _int_or_none(header.get("OBS_OFFSET"))

            if prev_end is not None and offset is not None:
                gap_bytes = offset - prev_end
                if gap_bytes != 0:
                    rate = float(header.get("BYTES_PER_SECOND", 0) or 0)
                    gap_s = gap_bytes / rate if rate else float("nan")
                    kind = "gap" if gap_bytes > 0 else "overlap"
                    msg = (f"{os.path.basename(filename)} is not continuous with "
                           f"the previous file: {gap_bytes} bytes ({gap_s:.3f} s) "
                           f"{kind}")
                    if allow_gaps:
                        print(f"  WARNING: {msg} — stitching anyway")
                    else:
                        raise ValueError(
                            f"{msg}. Files sharing a UTC_START can be separate "
                            f"dumps: pass the full UTC_START_OBSOFFSET filename "
                            f"prefix (e.g. "
                            f"'{os.path.basename(files[0]).split('.')[0]}') to "
                            f"select a single dump, or allow_gaps=True to stitch "
                            f"anyway."
                        )

            if offset is not None:
                prev_end = offset + n_time_file * bytes_per_time
            layouts.append((filename, header, bytes_per_time, n_time_file))

        n_time_total = sum(lay[3] for lay in layouts)
        if n_time is None:
            n_time = n_time_total
        else:
            n_time = min(n_time, n_time_total)

        voltages = {
            s: np.empty((n_time, n_chan_sub, n_adc), dtype=np.complex64)
            for s in snaps
        }

        t0 = 0
        for filename, header, bytes_per_time, n_time_file in layouts:
            take = min(n_time_file, n_time - t0)
            if take <= 0:
                break
            n_snaps_file = bytes_per_time // (n_chan_sub * n_adc)
            bad = [s for s in snaps if s >= n_snaps_file]
            if bad:
                raise ValueError(
                    f"{filename} has {n_snaps_file} SNAP slots "
                    f"(RESOLUTION {bytes_per_time}); requested SNAPs {bad}"
                )
            if verbose:
                print(f"Reading {filename}")
                print(f"  n_time={take} / {n_time_file}, SNAPs={snaps}")

            with open(filename, "rb") as f:
                f.seek(HEADER_SIZE)
                raw = np.frombuffer(f.read(take * bytes_per_time), dtype=np.uint8)
            raw = raw.reshape(take, n_snaps_file, n_chan_sub, n_adc)

            for s in snaps:
                voltages[s][t0:t0 + take] = unpack_4bit(raw[:, s, :, :])
            t0 += take

        if verbose:
            print(f"  Stream {index}: n_time={n_time} / {n_time_total} "
                  f"from {len(layouts)} file(s)")

        return voltages, layouts[0][1]

    def read_full_band(
        self,
        antenna_csv: str | None = None,
        n_time: int | None = None,
        snaps: list[int] | None = None,
        freq_order: str = "descending",
        trust_header: bool = False,
        verbose: bool = True,
        allow_gaps: bool = False,
    ) -> dict:
        """
        Read all subbands and stitch into full band.

        The legacy layout needs all 3 subbands. A stream dump can be missing
        streams (only the streams around the trigger are written), and those
        sub-bands are zero-filled with a warning (their indices come back in
        `filled_subbands`); only a dump with no data at all raises. The
        streams that are present must start at the same OBS_OFFSET, otherwise
        they are not the same dump and the read raises.

        Parameters
        ----------
        antenna_csv : str, optional
            Path to antenna mapping CSV. If provided, extracts per-antenna
            voltages as (n_time, 3072, n_ant).
        n_time : int, optional
            Time samples to read per subband. None = all.
        snaps : list of int, optional
            SNAP indices to extract. Default: active_snaps from config.
        freq_order : str
            'descending' (default, native) or 'ascending'.
        trust_header : bool
            Trust all DADA header fields.
        verbose : bool
            Print progress.
        allow_gaps : bool
            Stitch across a discontinuity in OBS_OFFSET within a stream
            instead of raising. See read_subband().

        Returns
        -------
        dict with keys:
            voltages : np.ndarray or dict
                If antenna_csv: (n_time, 3072, n_ant) complex64.
                If not: {snap_id: (n_time, 3072, n_adc) complex64}.
            header : dict
                Header from first subband file.
            freq_mhz : np.ndarray
                Full 3072-channel frequency axis.
            utc_start : str
                UTC_START from header.
            antenna_df : pd.DataFrame or None
                Antenna mapping if csv was provided.
            filled_subbands : list of int
                Sub-band indices that were zero-filled because no data was
                found for them.
        """
        cfg = self._cfg
        subband_dirs = cfg["subband_dirs"]

        if snaps is None:
            snaps = cfg["active_snaps"]

        n_subbands = cfg["n_subbands"]
        n_chan_sub = cfg["n_chan_per_subband"]
        n_adc = cfg["n_adc_per_snap"]

        if self._stream_layout:
            # Only the streams around the trigger are written
            if not self._subband_files:
                raise FileNotFoundError(
                    f"No stream data for {self._timestamp} in {self._data_dir}"
                )
            self._check_stream_alignment()
        else:
            # Require all 3 subbands
            for i in range(n_subbands):
                if i not in self._subband_files:
                    raise FileNotFoundError(
                        f"Subband directory not found for index {i} "
                        f"({subband_dirs[i]})"
                    )

        # Read each subband
        sub_voltages = [None] * n_subbands
        first_header = None
        utc_start = "unknown"

        for i in range(n_subbands):
            if verbose:
                print_progress(i + 1, n_subbands, prefix="Reading subbands")
            if i not in self._subband_files:
                continue
            result = self.read_subband(
                i, n_time=n_time, snaps=snaps,
                freq_order=freq_order, trust_header=trust_header,
                verbose=verbose, allow_gaps=allow_gaps,
            )
            if first_header is None:
                first_header = result["header"]
                utc_start = first_header.get("UTC_START", "unknown")
            sub_voltages[i] = result["voltages"]

        # Zero-fill the sub-bands with no data (stream layout only — the
        # legacy check above already required all 3) and trim the rest to the
        # shortest one so they stitch
        n_time_read = min(
            v[snaps[0]].shape[0] for v in sub_voltages if v is not None
        )
        filled_subbands = []
        for i in range(n_subbands):
            if sub_voltages[i] is None:
                filled_subbands.append(i)
                print(f"  WARNING: no data for sub-band {i} "
                      f"({subband_dirs[i]}) — zero-filling {n_time_read} samples")
                sub_voltages[i] = {
                    s: np.zeros((n_time_read, n_chan_sub, n_adc), dtype=np.complex64)
                    for s in snaps
                }
            else:
                lost = sub_voltages[i][snaps[0]].shape[0] - n_time_read
                if lost > 0:
                    print(f"  WARNING: sub-band {i} ({subband_dirs[i]}) is "
                          f"{lost} samples longer than the shortest sub-band — "
                          f"trimming all to {n_time_read} samples")
                for s in snaps:
                    sub_voltages[i][s] = sub_voltages[i][s][:n_time_read]

        # Stitch along frequency axis. Each sub-band is already reversed
        # internally for ascending order; the sub-band ORDER also flips, so
        # the lowest one comes first.
        order = sub_voltages[::-1] if freq_order == "ascending" else sub_voltages
        voltages_stitched = {}
        for s in snaps:
            voltages_stitched[s] = np.concatenate(
                [sub[s] for sub in order], axis=1
            )

            if verbose:
                print(f"\nSNAP {s} stitched: shape={voltages_stitched[s].shape}, "
                      f"mean_power={np.mean(np.abs(voltages_stitched[s]) ** 2):.3f}")

        # Build full frequency axis
        n_chan_total = cfg["n_chan_total"]
        freq_top = cfg["freq_top_mhz"]
        chan_bw = cfg["chan_bw_mhz"]
        freq_desc = freq_top - chan_bw * np.arange(n_chan_total, dtype=np.float64)
        if freq_order == "ascending":
            freq_mhz = freq_desc[::-1].copy()
        else:
            freq_mhz = freq_desc

        if verbose:
            print(f"\nFrequency axis: {freq_mhz[0]:.3f} – {freq_mhz[-1]:.3f} MHz "
                  f"({n_chan_total} channels, {freq_order})")
            print(f"UTC_START: {utc_start}")

        # Extract per-antenna voltages if CSV provided
        antenna_df = None
        if antenna_csv is not None:
            antenna_df = pd.read_csv(antenna_csv)
            # Support legacy column names
            rename = {}
            if "antenna" in antenna_df.columns and "antenna_id" not in antenna_df.columns:
                rename["antenna"] = "antenna_id"
            if "snap" in antenna_df.columns and "snap_id" not in antenna_df.columns:
                rename["snap"] = "snap_id"
            if rename:
                antenna_df = antenna_df.rename(columns=rename)

            n_ant = len(antenna_df)
            s0 = list(voltages_stitched.keys())[0]
            nt, nc, _ = voltages_stitched[s0].shape

            # Bounds-check every row before indexing: a negative adc would
            # otherwise wrap round and hand back a different antenna
            n_snap_max = max(cfg["n_snap_slots"], max(snaps) + 1)
            bad_rows = []
            for i, (_, row) in enumerate(antenna_df.iterrows()):
                snap = int(row["snap_id"])
                adc = int(row["adc"])
                if not 0 <= snap < n_snap_max or not 0 <= adc < n_adc:
                    bad_rows.append(f"row {i} (antenna_id "
                                    f"{row.get('antenna_id', '?')}): "
                                    f"snap_id {snap}, adc {adc}")
            if bad_rows:
                raise ValueError(
                    f"{antenna_csv} has rows outside the hardware layout "
                    f"(snap_id must be 0-{n_snap_max - 1}, adc must be "
                    f"0-{n_adc - 1}): " + "; ".join(bad_rows)
                )

            ant_voltages = np.zeros((nt, nc, n_ant), dtype=np.complex64)
            for i, (_, row) in enumerate(antenna_df.iterrows()):
                if verbose:
                    print_progress(i + 1, n_ant, prefix="Extracting antennas")
                snap = int(row["snap_id"])
                adc = int(row["adc"])
                if snap in voltages_stitched:
                    ant_voltages[:, :, i] = voltages_stitched[snap][:, :, adc]
                else:
                    if verbose:
                        print(f"  Ant {int(row['antenna_id']):2d}: "
                              f"SNAP {snap} NOT LOADED — zeros")

            return FullBandResult(
                voltages=ant_voltages,
                header=first_header,
                freq_mhz=freq_mhz,
                utc_start=utc_start,
                antenna_df=antenna_df,
                filled_subbands=filled_subbands,
            )

        return FullBandResult(
            voltages=voltages_stitched,
            header=first_header,
            freq_mhz=freq_mhz,
            utc_start=utc_start,
            antenna_df=None,
            filled_subbands=filled_subbands,
        )
