"""Tests for correlator reader and NPZ writer round-trips."""

import os

import numpy as np
import pytest

from casm_io.correlator.formats import VisibilityFormat
from casm_io.correlator.reader import read_visibilities, discover_files
from casm_io.correlator.writer import write_npz, read_npz


class TestSyntheticDatRoundTrip:
    """Write a synthetic .dat, read it back, verify values."""

    def test_read_full_file(self, tmp_path, tiny_format):
        """Read all integrations from a synthetic .dat file."""
        fmt = tiny_format
        nbl = fmt.n_baselines  # 10
        ntime = fmt.ntime_per_file  # 2

        # Create data with known pattern
        data = np.zeros((ntime, fmt.nchan, nbl, 2), dtype=np.int32)
        for t in range(ntime):
            for f in range(fmt.nchan):
                for bl in range(nbl):
                    data[t, f, bl, 0] = t * 1000 + f * 10 + bl
                    data[t, f, bl, 1] = -(t * 1000 + f * 10 + bl)

        base_str = "2026-01-01-00:00:00"
        fpath = tmp_path / f"{base_str}.0"
        data.tofile(str(fpath))

        result = read_visibilities(
            str(tmp_path), base_str, fmt, verbose=False,
        )
        vis = result["vis"]
        assert vis.shape == (ntime, fmt.nchan, nbl)
        assert vis.dtype == np.complex64

        # Check a specific value
        expected_re = 0 * 1000 + 3 * 10 + 5  # t=0, f=3, bl=5
        expected_im = -expected_re
        np.testing.assert_allclose(
            vis[0, 3, 5], expected_re + 1j * expected_im, atol=1e-3
        )

    def test_read_partial_last_file(self, tmp_path, tiny_format):
        """A partial last file (fewer integrations) should be read without error."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        base_str = "2026-01-01-00:00:00"

        # File 0: full (2 integrations)
        data0 = np.ones((2, fmt.nchan, nbl, 2), dtype=np.int32)
        (tmp_path / f"{base_str}.0").write_bytes(data0.tobytes())

        # File 1: partial (1 integration instead of 2)
        data1 = np.ones((1, fmt.nchan, nbl, 2), dtype=np.int32) * 2
        (tmp_path / f"{base_str}.1").write_bytes(data1.tobytes())

        result = read_visibilities(
            str(tmp_path), base_str, fmt, verbose=False,
        )
        # Should get 3 integrations total (2 + 1)
        assert result["vis"].shape[0] == 3

    def test_ascending_reverses_freq(self, tmp_path, tiny_format):
        fmt = tiny_format
        nbl = fmt.n_baselines
        base_str = "2026-01-01-00:00:00"

        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        # Mark first freq channel with value 100, last with 200
        data[:, 0, :, 0] = 100
        data[:, -1, :, 0] = 200
        (tmp_path / f"{base_str}.0").write_bytes(data.tobytes())

        desc = read_visibilities(str(tmp_path), base_str, fmt, freq_order="descending", verbose=False)
        asc = read_visibilities(str(tmp_path), base_str, fmt, freq_order="ascending", verbose=False)

        # Ascending should reverse the frequency axis
        np.testing.assert_array_equal(asc["freq_mhz"], desc["freq_mhz"][::-1])


class TestDiscoverFiles:
    def test_discover(self, tmp_path):
        base = "2026-01-01-00:00:00"
        for i in range(5):
            (tmp_path / f"{base}.{i}").touch()
        result = discover_files(str(tmp_path), base)
        assert set(result.keys()) == {0, 1, 2, 3, 4}


class TestNpzRoundTrip:
    def test_basic_round_trip(self, tmp_path):
        vis = np.random.randn(10, 64, 5).astype(np.complex64)
        freq = np.linspace(375, 468, 64)
        time = np.arange(10) * 137.0 + 1700000000.0

        path = str(tmp_path / "test.npz")
        write_npz(path, vis, freq, time, ref=5, targets=[0, 1, 2, 3, 4])

        loaded = read_npz(path)
        np.testing.assert_array_almost_equal(loaded["vis"], vis, decimal=5)
        np.testing.assert_array_almost_equal(loaded["freq_mhz"], freq, decimal=3)
        np.testing.assert_array_almost_equal(loaded["time_unix"], time, decimal=3)
        assert loaded["ref"] == 5
        np.testing.assert_array_equal(loaded["targets"], [0, 1, 2, 3, 4])

    def test_old_key_names(self, tmp_path):
        """read_npz handles legacy key names (vis_ref, ref_adc, target_adcs)."""
        vis = np.random.randn(5, 32, 3).astype(np.complex64)
        freq = np.linspace(375, 468, 32).astype(np.float32)
        time = np.arange(5, dtype=np.float64) * 100.0 + 1700000000.0

        path = str(tmp_path / "legacy.npz")
        np.savez(
            path,
            vis_ref=vis,
            freq_mhz=freq,
            time_unix=time,
            ref_adc=np.int64(7),
            target_adcs=np.array([1, 2, 3], dtype=np.int64),
        )

        loaded = read_npz(path)
        np.testing.assert_array_almost_equal(loaded["vis"], vis, decimal=5)
        assert loaded["ref"] == 7
        np.testing.assert_array_equal(loaded["targets"], [1, 2, 3])
