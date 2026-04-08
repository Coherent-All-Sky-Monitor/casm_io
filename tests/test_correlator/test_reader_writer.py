"""Tests for correlator reader and NPZ writer round-trips."""

import os
import warnings

import numpy as np
import pytest

from casm_io.correlator.formats import VisibilityFormat
from casm_io.correlator.reader import VisibilityReader, discover_files, discover_observations, read_visibilities
from casm_io.correlator.writer import write_npz, read_npz


class TestVisibilityReaderInit:
    """Test VisibilityReader initialization and properties."""

    def test_init_discovers_files(self, tmp_path, tiny_format):
        base_str = "2026-01-01-00:00:00"
        fmt = tiny_format
        nbl = fmt.n_baselines
        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        for i in range(3):
            (tmp_path / f"{base_str}.{i}").write_bytes(data.tobytes())

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        assert reader.n_files == 3
        assert reader.available_indices == [0, 1, 2]
        assert reader.missing_indices == []

    def test_missing_indices(self, tmp_path, tiny_format):
        base_str = "2026-01-01-00:00:00"
        fmt = tiny_format
        nbl = fmt.n_baselines
        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        # Create files 0 and 2 but not 1
        (tmp_path / f"{base_str}.0").write_bytes(data.tobytes())
        (tmp_path / f"{base_str}.2").write_bytes(data.tobytes())

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        assert reader.n_files == 2
        assert reader.missing_indices == [1]

    def test_no_files_raises(self, tmp_path, tiny_format):
        with pytest.raises(RuntimeError, match="No files found"):
            VisibilityReader(str(tmp_path), "nonexistent", tiny_format)

    def test_time_span(self, tmp_path, tiny_format):
        base_str = "2026-01-01-00:00:00"
        fmt = tiny_format
        nbl = fmt.n_baselines
        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        (tmp_path / f"{base_str}.0").write_bytes(data.tobytes())

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        start, end = reader.time_span
        assert end - start == pytest.approx(fmt.file_duration_s)


class TestVisibilityReaderRead:
    """Test VisibilityReader.read() with synthetic data."""

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

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        result = reader.read(verbose=False)
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

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        result = reader.read(verbose=False)
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

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        desc = reader.read(freq_order="descending", verbose=False)
        asc = reader.read(freq_order="ascending", verbose=False)

        # Ascending should reverse the frequency axis
        np.testing.assert_array_equal(asc["freq_mhz"], desc["freq_mhz"][::-1])


class TestNfiles:
    """Test nfiles parameter and zero-fill behavior."""

    def test_nfiles_basic(self, tmp_path, tiny_format):
        fmt = tiny_format
        nbl = fmt.n_baselines
        base_str = "2026-01-01-00:00:00"

        for i in range(5):
            data = np.ones((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32) * (i + 1)
            (tmp_path / f"{base_str}.{i}").write_bytes(data.tobytes())

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        result = reader.read(nfiles=3, verbose=False)
        # Should read 3 files x 2 integrations = 6 integrations
        assert result["vis"].shape[0] == 3 * fmt.ntime_per_file

    def test_nfiles_time_end_mutual_exclusion(self, tmp_path, tiny_format):
        fmt = tiny_format
        nbl = fmt.n_baselines
        base_str = "2026-01-01-00:00:00"
        data = np.ones((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        (tmp_path / f"{base_str}.0").write_bytes(data.tobytes())

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        with pytest.raises(ValueError, match="mutually exclusive"):
            reader.read(nfiles=1, time_end="2026-01-01-00:00:10", verbose=False)

    def test_nfiles_zero_fill_missing(self, tmp_path, tiny_format):
        """Missing files in nfiles range should be zero-filled with warning."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        base_str = "2026-01-01-00:00:00"

        # Create files 0 and 2 but not 1
        data = np.ones((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32) * 5
        (tmp_path / f"{base_str}.0").write_bytes(data.tobytes())
        (tmp_path / f"{base_str}.2").write_bytes(data.tobytes())

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = reader.read(nfiles=3, verbose=False)
            assert len(w) == 1
            assert "Missing" in str(w[0].message)

        # Should still get all integrations
        assert result["vis"].shape[0] == 3 * fmt.ntime_per_file
        # File 1 (indices 2:4) should be zeros
        file1_vis = result["vis"][fmt.ntime_per_file:2*fmt.ntime_per_file]
        assert np.all(file1_vis == 0)
        # Files 0 and 2 should be non-zero
        assert np.any(result["vis"][:fmt.ntime_per_file] != 0)

    def test_metadata_includes_missing(self, tmp_path, tiny_format):
        fmt = tiny_format
        nbl = fmt.n_baselines
        base_str = "2026-01-01-00:00:00"
        data = np.ones((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        (tmp_path / f"{base_str}.0").write_bytes(data.tobytes())
        (tmp_path / f"{base_str}.2").write_bytes(data.tobytes())

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = reader.read(nfiles=3, verbose=False)

        assert "missing_files" in result["metadata"]
        assert 1 in result["metadata"]["missing_files"]


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
class TestChannelSlicing:
    """Tests for channels and freq_range_mhz parameters."""

    def test_channels_tuple(self, tmp_path, tiny_format):
        """Reading with channels=(2, 5) returns 3 frequency channels."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        base_str = "2026-01-01-00:00:00"

        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        # Mark each freq channel with its index
        for f in range(fmt.nchan):
            data[:, f, :, 0] = f * 10
        (tmp_path / f"{base_str}.0").write_bytes(data.tobytes())

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        result = reader.read(channels=(2, 5), verbose=False)

        assert result.vis.shape == (fmt.ntime_per_file, 3, nbl)
        # Channel 2 should have re=20, channel 3=30, channel 4=40
        np.testing.assert_allclose(result.vis[0, 0, 0].real, 20.0, atol=1)
        np.testing.assert_allclose(result.vis[0, 1, 0].real, 30.0, atol=1)
        np.testing.assert_allclose(result.vis[0, 2, 0].real, 40.0, atol=1)

    def test_channels_freq_axis_sliced(self, tmp_path, tiny_format):
        """Frequency axis should match the sliced channels."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        base_str = "2026-01-01-00:00:00"

        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        (tmp_path / f"{base_str}.0").write_bytes(data.tobytes())

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        full = reader.read(verbose=False)
        sliced = reader.read(channels=(2, 5), verbose=False)

        np.testing.assert_array_almost_equal(
            sliced.freq_mhz, full.freq_mhz[2:5]
        )

    def test_channels_with_ascending(self, tmp_path, tiny_format):
        """Channel slicing + freq_order=ascending should flip correctly."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        base_str = "2026-01-01-00:00:00"

        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        for f in range(fmt.nchan):
            data[:, f, :, 0] = f * 10
        (tmp_path / f"{base_str}.0").write_bytes(data.tobytes())

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        desc = reader.read(channels=(2, 5), freq_order="descending", verbose=False)
        asc = reader.read(channels=(2, 5), freq_order="ascending", verbose=False)

        np.testing.assert_array_almost_equal(asc.freq_mhz, desc.freq_mhz[::-1])

    def test_channels_and_freq_range_exclusive(self, tmp_path, tiny_format):
        """Passing both channels and freq_range_mhz raises ValueError."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        base_str = "2026-01-01-00:00:00"

        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        (tmp_path / f"{base_str}.0").write_bytes(data.tobytes())

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        with pytest.raises(ValueError, match="mutually exclusive"):
            reader.read(channels=(0, 4), freq_range_mhz=(460, 465), verbose=False)

    def test_channels_full_range_matches_no_slice(self, tmp_path, tiny_format):
        """channels=(0, nchan) should produce same result as no slicing."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        base_str = "2026-01-01-00:00:00"

        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        for f in range(fmt.nchan):
            data[:, f, :, 0] = f
        (tmp_path / f"{base_str}.0").write_bytes(data.tobytes())

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        full = reader.read(verbose=False)
        sliced = reader.read(channels=(0, fmt.nchan), verbose=False)

        np.testing.assert_array_equal(full.vis, sliced.vis)
        np.testing.assert_array_equal(full.freq_mhz, sliced.freq_mhz)

    def test_channels_with_ref_targets(self, tmp_path, tiny_format):
        """Channel slicing works together with baseline extraction."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        base_str = "2026-01-01-00:00:00"

        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        (tmp_path / f"{base_str}.0").write_bytes(data.tobytes())

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        result = reader.read(channels=(1, 4), ref=0, targets=[1, 2], verbose=False)

        assert result.vis.shape == (fmt.ntime_per_file, 3, 2)

    def test_channels_metadata(self, tmp_path, tiny_format):
        """Metadata should include channel slicing info."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        base_str = "2026-01-01-00:00:00"

        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        (tmp_path / f"{base_str}.0").write_bytes(data.tobytes())

        reader = VisibilityReader(str(tmp_path), base_str, fmt)
        result = reader.read(channels=(2, 6), verbose=False)

        assert result.metadata["channels"] == (2, 6)


class TestDiscoverObservations:
    """Tests for discover_observations()."""

    def test_discovers_multiple_obs(self, tmp_path, tiny_format):
        """Discover multiple observations in a directory."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)

        for bs in ["2026-01-01-00:00:00", "2026-01-01-06:00:00", "2026-01-02-00:00:00"]:
            (tmp_path / f"{bs}.0").write_bytes(data.tobytes())

        obs = discover_observations(str(tmp_path), fmt=fmt)
        assert len(obs) == 3
        assert obs[0]["base_str"] == "2026-01-01-00:00:00"
        assert obs[-1]["base_str"] == "2026-01-02-00:00:00"

    def test_obs_time_spans(self, tmp_path, tiny_format):
        """Time spans should be computed correctly from file count."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)

        bs = "2026-01-01-00:00:00"
        for i in range(3):
            (tmp_path / f"{bs}.{i}").write_bytes(data.tobytes())

        obs = discover_observations(str(tmp_path), fmt=fmt)
        assert len(obs) == 1
        assert obs[0]["n_files"] == 3
        expected_dur = 3 * fmt.file_duration_s
        assert abs(obs[0]["time_end"] - obs[0]["time_start"] - expected_dur) < 0.01

    def test_empty_dir(self, tmp_path):
        """Empty directory returns empty list."""
        obs = discover_observations(str(tmp_path))
        assert obs == []

    def test_nonexistent_dir(self):
        """Non-existent directory returns empty list."""
        obs = discover_observations("/nonexistent/path/12345")
        assert obs == []


class TestReadVisibilities:
    """Tests for read_visibilities() top-level function."""

    def test_single_obs(self, tmp_path, tiny_format):
        """Read from a single observation."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        data = np.ones((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)

        bs = "2026-01-01-00:00:00"
        for i in range(3):
            (tmp_path / f"{bs}.{i}").write_bytes(data.tobytes())

        result = read_visibilities(
            "2026-01-01 00:00:00", "2026-01-01 00:00:10",
            data_dir=str(tmp_path), fmt=fmt, verbose=False,
        )
        assert result.vis.shape[0] > 0
        assert result.vis.shape[2] == nbl

    def test_no_overlap_raises(self, tmp_path, tiny_format):
        """Time range with no matching observations raises ValueError."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        (tmp_path / "2026-01-01-00:00:00.0").write_bytes(data.tobytes())

        with pytest.raises(ValueError, match="No observations found"):
            read_visibilities(
                "2026-06-01 00:00:00", "2026-06-01 01:00:00",
                data_dir=str(tmp_path), fmt=fmt, verbose=False,
            )

    def test_gap_detection(self, tmp_path, tiny_format):
        """Gaps between observations should be flagged in metadata."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        data = np.ones((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)

        # Two observations with a gap between them
        for i in range(2):
            (tmp_path / f"2026-01-01-00:00:00.{i}").write_bytes(data.tobytes())
        for i in range(2):
            (tmp_path / f"2026-01-01-06:00:00.{i}").write_bytes(data.tobytes())

        result = read_visibilities(
            "2026-01-01 00:00:00", "2026-01-01 07:00:00",
            data_dir=str(tmp_path), fmt=fmt, verbose=False,
        )
        assert len(result.metadata["gaps"]) > 0
        assert result.metadata["gaps"][0]["duration_s"] > 0

    def test_with_channel_slicing(self, tmp_path, tiny_format):
        """read_visibilities with channel slicing."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)

        bs = "2026-01-01-00:00:00"
        for i in range(2):
            (tmp_path / f"{bs}.{i}").write_bytes(data.tobytes())

        result = read_visibilities(
            "2026-01-01 00:00:00", "2026-01-01 00:00:10",
            data_dir=str(tmp_path), fmt=fmt, channels=(1, 4), verbose=False,
        )
        assert result.vis.shape[1] == 3  # 3 channels

    def test_metadata_observations_list(self, tmp_path, tiny_format):
        """Metadata should list which observations were used."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        data = np.ones((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        bs = "2026-01-01-00:00:00"
        (tmp_path / f"{bs}.0").write_bytes(data.tobytes())

        result = read_visibilities(
            "2026-01-01 00:00:00", "2026-01-01 00:00:05",
            data_dir=str(tmp_path), fmt=fmt, verbose=False,
        )
        assert bs in result.metadata["observations"]