"""Shared test fixtures for casm_io tests."""

import os
import struct
import tempfile

import numpy as np
import pandas as pd
import pytest

from casm_io.correlator.formats import VisibilityFormat


# ---------------------------------------------------------------------------
# Correlator fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_format():
    """A small VisibilityFormat for fast tests: 4 inputs, 8 channels, 2 time steps."""
    return VisibilityFormat(
        name="test_tiny",
        nsig=4,
        dt_raw_s=1.0,
        ntime_per_file=2,
        nchan=8,
        chan_bw_mhz=0.030517578125,
        freq_top_mhz=468.75,
        freq_bottom_mhz=468.75 - 8 * 0.030517578125,
        native_order="descending",
    )


@pytest.fixture
def tiny_dat_file(tmp_path, tiny_format):
    """Write a synthetic .dat file with known values and return (path, expected_data)."""
    fmt = tiny_format
    nbl = fmt.n_baselines  # 4*5/2 = 10
    ntime = fmt.ntime_per_file  # 2

    # Create known complex data: vis[t, f, bl] = (t+1) + (f+1)*1j for simplicity
    # Stored as int32 pairs (re, im)
    data = np.zeros((ntime, fmt.nchan, nbl, 2), dtype=np.int32)
    for t in range(ntime):
        for f in range(fmt.nchan):
            for bl in range(nbl):
                data[t, f, bl, 0] = t * 100 + f * 10 + bl  # real
                data[t, f, bl, 1] = t * 100 + f * 10 + bl + 1  # imag

    fpath = tmp_path / "2026-01-01-00:00:00.0"
    data.tofile(str(fpath))
    return str(fpath), data


@pytest.fixture
def antenna_csv_legacy(tmp_path):
    """CSV with legacy column names (antenna, snap, packet_idx)."""
    df = pd.DataFrame({
        "antenna": [1, 2, 3, 4],
        "snap": [0, 0, 1, 1],
        "adc": [0, 1, 0, 1],
        "packet_idx": [10, 20, 30, 40],
    })
    path = tmp_path / "antenna_legacy.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def antenna_csv_standard(tmp_path):
    """CSV with standard column names and functional column."""
    df = pd.DataFrame({
        "antenna_id": [1, 2, 3, 4],
        "snap_id": [0, 0, 1, 1],
        "adc": [0, 1, 0, 1],
        "packet_index": [10, 20, 30, 40],
        "functional": [1, 1, 0, 1],
    })
    path = tmp_path / "antenna_standard.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def synthetic_dat_with_header(tmp_path, tiny_format):
    """Write a synthetic .dat file with 4096-byte ASCII header + binary data.

    Returns (filepath, header_dict, expected_data, tiny_format).
    """
    fmt = tiny_format
    nbl = fmt.n_baselines  # 10
    ntime = fmt.ntime_per_file  # 2

    # Build header matching tiny_format
    tsamp_us = fmt.dt_raw_s * 1e6  # 1.0s -> 1e6 microseconds
    header_lines = [
        "HDR_SIZE 4096",
        f"NCHAN {fmt.nchan}",
        f"NBASELINE {nbl}",
        f"CORR_DUMP_DUMPS_PER_FILE {ntime}",
        f"TSAMP {tsamp_us}",
        f"FREQ_START {fmt.freq_top_mhz}",
        f"CHANBW -{fmt.chan_bw_mhz}",
        "UTC_START 2026-03-05-08:02:39",
    ]
    header_text = "\n".join(header_lines) + "\n"
    header_bytes = header_text.encode("ascii")
    header_padded = header_bytes.ljust(4096, b"\x00")

    # Create known data (same pattern as tiny_dat_file)
    data = np.zeros((ntime, fmt.nchan, nbl, 2), dtype=np.int32)
    for t in range(ntime):
        for f in range(fmt.nchan):
            for bl in range(nbl):
                data[t, f, bl, 0] = t * 100 + f * 10 + bl
                data[t, f, bl, 1] = t * 100 + f * 10 + bl + 1

    fpath = tmp_path / "2026-03-05-08:02:39.0"
    with open(fpath, "wb") as fobj:
        fobj.write(header_padded)
        data.tofile(fobj)

    header_dict = {}
    for line in header_lines:
        parts = line.split(None, 1)
        header_dict[parts[0]] = parts[1]

    return str(fpath), header_dict, data, fmt


# ---------------------------------------------------------------------------
# Voltage fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dada_file(tmp_path):
    """Write a synthetic .dada file: 4096-byte header + known byte pattern.

    Returns (filepath, header_dict, raw_bytes_array, config).
    """
    n_snaps = 11
    n_adc = 12
    n_chan = 1024
    n_time = 5

    # Build header
    header_lines = [
        "HDR_SIZE 4096",
        "UTC_START 2026-02-17-21:10:43",
        "TSAMP 32.768",
        "NCHAN 1024",
        "NBIT 4",
        "NDIM 2",
        "NANT 66",
        "NPOL 2",
        "FILE_SIZE 999999999",
        "SOURCE J0000-0000",
        "ENCODING TWOSCOMPLEMENT",
        "BW -31.25",
        "RESOLUTION 135168",
        "UDP_NANT 6",
        "STREAM_SUBBAND_ID 0",
        "FREQ 437.5",
    ]
    header_text = "\n".join(header_lines) + "\n"
    header_bytes = header_text.encode("ascii")
    header_padded = header_bytes.ljust(4096, b"\x00")

    # Data: each byte encodes (real_4bit, imag_4bit)
    # Use a reproducible pattern
    rng = np.random.RandomState(42)
    raw = rng.randint(0, 256, size=(n_time, n_snaps, n_chan, n_adc), dtype=np.uint8)

    fpath = tmp_path / "chan0_1023" / "2026-02-17-21:10:43_CHAN0_1023_0000.dada"
    fpath.parent.mkdir(parents=True)
    with open(fpath, "wb") as f:
        f.write(header_padded)
        raw.tofile(f)

    header_dict = {}
    for line in header_lines:
        parts = line.split(None, 1)
        header_dict[parts[0]] = parts[1]

    return str(fpath), header_dict, raw, {
        "n_snaps": n_snaps, "n_adc": n_adc, "n_chan": n_chan, "n_time": n_time,
    }


def _stream_header_lines(subband_id, obs_offset, dump_bytes, resolution,
                         start_channel, n_chan=512):
    """DADA header lines for one triggered-dump stream file."""
    return [
        "HDR_SIZE 4096",
        "UTC_START 2026-03-16-00:30:11",
        "TSAMP 32.768",
        f"NCHAN {n_chan}",
        "NBIT 4",
        "NDIM 2",
        "NANT 66",
        "NPOL 2",
        "ENCODING TWOSCOMPLEMENT",
        "BW -15.625",
        "CHANBW -0.030517578125",
        "BYTES_PER_SECOND 2062500000.0",
        f"RESOLUTION {resolution}",
        f"START_CHANNEL {start_channel}",
        f"END_CHANNEL {start_channel + n_chan - 1}",
        "UDP_NANT 6",
        f"STREAM_SUBBAND_ID {subband_id}",
        "FREQ 437.5",
        f"OBS_OFFSET {obs_offset}",
        "DUMP_UTC_START 2026-03-16-00:30:32.700",
        "DUMP_UTC_STOP 2026-03-16-00:30:32.800",
        f"DUMP_BYTES {dump_bytes}",
        "FILE_NUMBER 0",
        "FILE_SIZE 20608862208",
    ]


def _write_stream_file(path, header_lines, raw):
    """Write a 4096-byte ASCII header followed by raw bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header_text = "\n".join(header_lines) + "\n"
    with open(path, "wb") as f:
        f.write(header_text.encode("ascii").ljust(4096, b"\x00"))
        raw.tofile(f)


def _stream_ramp(t_start, n_time, subband_id, n_snaps, n_chan, n_adc):
    """Raw bytes encoding a time ramp: sample t unpacks to t + 1j*subband_id.

    The upper nibble (real part) counts time from the start of the dump and
    the lower nibble (imaginary part) is the stream index, so a stitched read
    shows immediately if files or sub-bands landed in the wrong order.
    """
    times = np.arange(t_start, t_start + n_time, dtype=np.uint8)
    values = ((times & 0x07) << 4) | (subband_id & 0x07)
    return np.broadcast_to(
        values[:, None, None, None], (n_time, n_snaps, n_chan, n_adc)
    ).astype(np.uint8)


@pytest.fixture
def synthetic_stream_dump(tmp_path):
    """Write a synthetic triggered dump in the stream_N layout.

    Streams 0, 1 and 3 are present (2, 4 and 5 are missing, to exercise
    zero-filling); stream 0 is split over two files with a continuous
    OBS_OFFSET, and stream 1 has 3 samples of junk past DUMP_BYTES.

    Returns (data_dir, timestamp, config) where config holds the shape
    parameters used to build the files.
    """
    n_snaps = 11
    n_adc = 12
    n_chan = 512
    n_time = 7            # valid samples per stream, after DUMP_BYTES
    n_time_split = 4      # samples in the first file of stream 0
    resolution = n_snaps * n_chan * n_adc      # 67584 bytes per time sample
    timestamp = "2026-03-16-00:30:11"

    def name(obs_offset, filenum):
        return f"{timestamp}_{obs_offset:016d}.{filenum:06d}.dada"

    # Stream 0: split over two files, second one continues the ramp
    for filenum, (t_start, n_t) in enumerate(
        [(0, n_time_split), (n_time_split, n_time - n_time_split)]
    ):
        obs_offset = t_start * resolution
        _write_stream_file(
            tmp_path / "stream_0" / name(obs_offset, filenum),
            _stream_header_lines(0, obs_offset, n_t * resolution, resolution, 512),
            _stream_ramp(t_start, n_t, 0, n_snaps, n_chan, n_adc),
        )

    # Stream 1: one file, written into a longer pre-allocated span — the
    # 3 samples past DUMP_BYTES are junk and must not be read
    raw = np.concatenate([
        _stream_ramp(0, n_time, 1, n_snaps, n_chan, n_adc),
        np.full((3, n_snaps, n_chan, n_adc), 0xEE, dtype=np.uint8),
    ])
    _write_stream_file(
        tmp_path / "stream_1" / name(0, 0),
        _stream_header_lines(1, 0, n_time * resolution, resolution, 1024),
        raw,
    )

    # Stream 3: one file, no DUMP_BYTES padding
    _write_stream_file(
        tmp_path / "stream_3" / name(0, 0),
        _stream_header_lines(3, 0, n_time * resolution, resolution, 2048),
        _stream_ramp(0, n_time, 3, n_snaps, n_chan, n_adc),
    )

    return str(tmp_path), timestamp, {
        "n_snaps": n_snaps, "n_adc": n_adc, "n_chan": n_chan,
        "n_time": n_time, "n_time_split": n_time_split,
        "resolution": resolution, "streams": [0, 1, 3],
    }


@pytest.fixture
def synthetic_stream_dump_43slots(tmp_path):
    """A stream file from the March 4 epoch: 43 SNAP slots, not 11.

    RESOLUTION has to be read per file, so a reader that assumes the config's
    11 slots reshapes this file wrongly.

    Returns (data_dir, timestamp, config).
    """
    n_snaps = 43
    n_adc = 12
    n_chan = 512
    n_time = 2
    resolution = n_snaps * n_chan * n_adc      # 264192
    timestamp = "2026-03-04-23:16:08"

    raw = _stream_ramp(0, n_time, 0, n_snaps, n_chan, n_adc)
    _write_stream_file(
        tmp_path / "stream_0" / f"{timestamp}_{0:016d}.{0:06d}.dada",
        _stream_header_lines(0, 0, n_time * resolution, resolution, 512),
        raw,
    )

    return str(tmp_path), timestamp, {
        "n_snaps": n_snaps, "n_adc": n_adc, "n_chan": n_chan,
        "n_time": n_time, "resolution": resolution,
    }


@pytest.fixture
def synthetic_stream_dump_two_dumps(tmp_path):
    """Two separate dumps in stream_0 sharing one UTC_START.

    Both files match the bare timestamp prefix, but their OBS_OFFSETs are
    84 s apart, so they are not one dump split over two files. The second
    dump's time ramp starts at 4 so the two are told apart in the data.

    Returns (data_dir, timestamp, prefixes, config) where prefixes are the
    full UTC_START_OBSOFFSET names of the two dumps.
    """
    n_snaps = 11
    n_adc = 12
    n_chan = 512
    n_time = 3
    resolution = n_snaps * n_chan * n_adc
    timestamp = "2026-03-16-00:30:11"
    bytes_per_second = 2062500000.0
    gap_s = 84.0

    offsets = [0, int(gap_s * bytes_per_second)]
    for offset, t_start in zip(offsets, [0, 4]):
        _write_stream_file(
            tmp_path / "stream_0" / f"{timestamp}_{offset:016d}.{0:06d}.dada",
            _stream_header_lines(0, offset, n_time * resolution, resolution, 512),
            _stream_ramp(t_start, n_time, 0, n_snaps, n_chan, n_adc),
        )

    prefixes = [f"{timestamp}_{offset:016d}" for offset in offsets]
    return str(tmp_path), timestamp, prefixes, {
        "n_snaps": n_snaps, "n_adc": n_adc, "n_chan": n_chan,
        "n_time": n_time, "resolution": resolution,
        "gap_s": gap_s, "t_starts": [0, 4],
    }


@pytest.fixture
def synthetic_stream_dump_misaligned(tmp_path):
    """Streams 0 and 1 whose first files start at different OBS_OFFSETs.

    A healthy dump writes the same OBS_OFFSET into every stream, so this is
    two different dumps that happen to share a UTC_START.

    Returns (data_dir, timestamp, offsets, config).
    """
    n_snaps = 11
    n_adc = 12
    n_chan = 512
    n_time = 2
    resolution = n_snaps * n_chan * n_adc
    timestamp = "2026-03-16-00:30:11"

    offsets = {0: 0, 1: 5 * resolution}
    for index, offset in offsets.items():
        _write_stream_file(
            tmp_path / f"stream_{index}"
            / f"{timestamp}_{offset:016d}.{0:06d}.dada",
            _stream_header_lines(
                index, offset, n_time * resolution, resolution,
                512 * (index + 1),
            ),
            _stream_ramp(0, n_time, index, n_snaps, n_chan, n_adc),
        )

    return str(tmp_path), timestamp, offsets, {
        "n_snaps": n_snaps, "n_adc": n_adc, "n_chan": n_chan,
        "n_time": n_time, "resolution": resolution,
    }


@pytest.fixture
def synthetic_stream_dump_all_streams(tmp_path):
    """A complete triggered dump: all six streams, one file each.

    Returns (data_dir, timestamp, config).
    """
    n_snaps = 11
    n_adc = 12
    n_chan = 512
    n_time = 2
    resolution = n_snaps * n_chan * n_adc
    timestamp = "2026-03-16-00:33:14"

    for index in range(6):
        _write_stream_file(
            tmp_path / f"stream_{index}" / f"{timestamp}_{0:016d}.{0:06d}.dada",
            _stream_header_lines(
                index, 0, n_time * resolution, resolution, 512 * (index + 1),
            ),
            _stream_ramp(0, n_time, index, n_snaps, n_chan, n_adc),
        )

    return str(tmp_path), timestamp, {
        "n_snaps": n_snaps, "n_adc": n_adc, "n_chan": n_chan,
        "n_time": n_time, "resolution": resolution,
    }


# ---------------------------------------------------------------------------
# Filterbank fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_filterbank_header():
    """A minimal SIGPROC header dict."""
    return {
        "telescope_id": 20,
        "machine_id": 0,
        "data_type": 1,
        "source_name": "TEST_SOURCE",
        "nchans": 64,
        "nifs": 1,
        "nbits": 8,
        "tsamp": 0.001,
        "tstart": 60000.0,
        "fch1": 468.75,
        "foff": -0.030517578125,
    }


@pytest.fixture
def synthetic_filterbank_data():
    """Synthetic filterbank data (100 samples, 64 channels)."""
    rng = np.random.RandomState(123)
    return rng.randint(0, 256, size=(100, 64), dtype=np.uint8)


@pytest.fixture
def synthetic_multibeam_filterbank_header():
    """A minimal SIGPROC header dict for a multibeam file."""
    return {
        "telescope_id": 20,
        "machine_id": 0,
        "data_type": 1,
        "source_name": "TEST_MULTIBEAM",
        "nchans": 64,
        "nifs": 1,
        "nbits": 8,
        "nbeams": 4,
        "ibeam": 0,
        "tsamp": 0.001,
        "tstart": 60000.0,
        "fch1": 468.75,
        "foff": -0.030517578125,
    }


@pytest.fixture
def synthetic_multibeam_filterbank_data():
    """Synthetic multibeam filterbank data (4 beams, 100 samples, 64 channels).

    Each beam has a distinct pattern for verification:
    beam N has values offset by N * 50.
    """
    rng = np.random.RandomState(789)
    data = np.zeros((4, 100, 64), dtype=np.uint8)
    for beam in range(4):
        data[beam, :, :] = rng.randint(
            beam * 50, beam * 50 + 50, size=(100, 64), dtype=np.uint8
        )
    return data
