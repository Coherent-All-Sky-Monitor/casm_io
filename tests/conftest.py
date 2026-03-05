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
