"""Tests for filterbank read/write round-trips."""

import numpy as np
import pytest

from casm_io.filterbank.reader import read_filterbank
from casm_io.filterbank.writer import write_filterbank
from casm_io.filterbank.header import write_sigproc_header


class TestRoundTrip8bit:
    def test_write_read_uint8(self, tmp_path, synthetic_filterbank_header, synthetic_filterbank_data):
        header = synthetic_filterbank_header
        data = synthetic_filterbank_data  # (100, 64) uint8
        fpath = str(tmp_path / "test8.fil")

        info = write_filterbank(fpath, data, header, nbits=8)
        assert info["backend_used"] in ("sigpyproc", "standalone")

        result = read_filterbank(fpath)
        assert result["backend_used"] in ("sigpyproc", "standalone")
        assert result["data"].shape == data.shape
        np.testing.assert_array_equal(result["data"].astype(np.uint8), data)


class TestRoundTrip32bit:
    def test_write_read_float32(self, tmp_path, synthetic_filterbank_header):
        header = synthetic_filterbank_header.copy()
        rng = np.random.RandomState(456)
        data = rng.randn(50, 64).astype(np.float32)
        fpath = str(tmp_path / "test32.fil")

        info = write_filterbank(fpath, data, header, nbits=32)
        assert info["backend_used"] in ("sigpyproc", "standalone")

        result = read_filterbank(fpath)
        np.testing.assert_allclose(
            result["data"].astype(np.float32), data, atol=1e-5
        )


class TestBackendUsed:
    def test_backend_field_present(self, tmp_path, synthetic_filterbank_header, synthetic_filterbank_data):
        fpath = str(tmp_path / "test_backend.fil")
        write_info = write_filterbank(fpath, synthetic_filterbank_data, synthetic_filterbank_header, nbits=8)
        assert "backend_used" in write_info

        read_info = read_filterbank(fpath)
        assert "backend_used" in read_info
        assert read_info["backend_used"] in ("sigpyproc", "standalone")


class TestDefaultTelescopeId:
    def test_default_telescope_id(self, tmp_path):
        """When telescope_id is not specified, default to 20 (OVRO)."""
        header = {
            "nchans": 8,
            "nifs": 1,
            "nbits": 8,
            "tsamp": 0.001,
            "tstart": 60000.0,
            "fch1": 400.0,
            "foff": -1.0,
        }
        data = np.zeros((10, 8), dtype=np.uint8)
        fpath = str(tmp_path / "test_default.fil")

        write_filterbank(fpath, data, header, nbits=8)
        result = read_filterbank(fpath)
        assert result["header"].get("telescope_id") == 20


class TestFreqTimeAxes:
    def test_freq_axis_from_read(self, tmp_path, synthetic_filterbank_header, synthetic_filterbank_data):
        fpath = str(tmp_path / "test_axes.fil")
        write_filterbank(fpath, synthetic_filterbank_data, synthetic_filterbank_header, nbits=8)

        result = read_filterbank(fpath)
        freq = result["freq_mhz"]
        time = result["time_s"]

        assert freq.shape == (64,)
        assert abs(freq[0] - 468.75) < 1e-4
        assert time.shape == (100,)
        assert abs(time[0]) < 1e-10
