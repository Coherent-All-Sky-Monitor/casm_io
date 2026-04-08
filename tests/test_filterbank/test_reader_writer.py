"""Tests for filterbank read/write round-trips."""

import numpy as np
import pytest

from casm_io.filterbank.reader import FilterbankFile
from casm_io.filterbank.writer import write_filterbank
from casm_io.filterbank.header import write_sigproc_header


class TestRoundTrip8bit:
    def test_write_read_uint8(self, tmp_path, synthetic_filterbank_header, synthetic_filterbank_data):
        header = synthetic_filterbank_header
        data = synthetic_filterbank_data  # (100, 64) uint8
        fpath = str(tmp_path / "test8.fil")

        info = write_filterbank(fpath, data, header, nbits=8)
        assert info["backend_used"] in ("sigpyproc", "standalone")

        fb = FilterbankFile(fpath)
        assert fb.backend_used in ("sigpyproc", "standalone")
        assert fb.data.shape == data.shape
        np.testing.assert_array_equal(fb.data.astype(np.uint8), data)


class TestRoundTrip32bit:
    def test_write_read_float32(self, tmp_path, synthetic_filterbank_header):
        header = synthetic_filterbank_header.copy()
        rng = np.random.RandomState(456)
        data = rng.randn(50, 64).astype(np.float32)
        fpath = str(tmp_path / "test32.fil")

        info = write_filterbank(fpath, data, header, nbits=32)
        assert info["backend_used"] in ("sigpyproc", "standalone")

        fb = FilterbankFile(fpath)
        np.testing.assert_allclose(
            fb.data.astype(np.float32), data, atol=1e-5
        )


class TestBackendUsed:
    def test_backend_field_present(self, tmp_path, synthetic_filterbank_header, synthetic_filterbank_data):
        fpath = str(tmp_path / "test_backend.fil")
        write_info = write_filterbank(fpath, synthetic_filterbank_data, synthetic_filterbank_header, nbits=8)
        assert "backend_used" in write_info

        fb = FilterbankFile(fpath)
        assert fb.backend_used in ("sigpyproc", "standalone")


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
        fb = FilterbankFile(fpath)
        assert fb.header.get("telescope_id") == 20


class TestFreqTimeAxes:
    def test_freq_axis_from_read(self, tmp_path, synthetic_filterbank_header, synthetic_filterbank_data):
        fpath = str(tmp_path / "test_axes.fil")
        write_filterbank(fpath, synthetic_filterbank_data, synthetic_filterbank_header, nbits=8)

        fb = FilterbankFile(fpath)
        freq = fb.freq_mhz
        time = fb.time_s

        assert freq.shape == (64,)
        assert abs(freq[0] - 468.75) < 1e-4
        assert time.shape == (100,)
        assert abs(time[0]) < 1e-10


class TestLazyLoading:
    def test_data_not_loaded_on_init(self, tmp_path, synthetic_filterbank_header, synthetic_filterbank_data):
        fpath = str(tmp_path / "test_lazy.fil")
        write_filterbank(fpath, synthetic_filterbank_data, synthetic_filterbank_header, nbits=8)

        fb = FilterbankFile(fpath)
        assert fb._data is None  # data not loaded yet
        assert fb.nchans == 64
        assert fb.nsamples == 100
        # Now access data
        _ = fb.data
        assert fb._data is not None

    def test_properties_without_data_load(self, tmp_path, synthetic_filterbank_header, synthetic_filterbank_data):
        fpath = str(tmp_path / "test_props.fil")
        write_filterbank(fpath, synthetic_filterbank_data, synthetic_filterbank_header, nbits=8)

        fb = FilterbankFile(fpath)
        # These should work without loading data
        assert fb.nchans == 64
        assert fb.nsamples == 100
        assert fb.freq_mhz.shape == (64,)
        assert fb.time_s.shape == (100,)
        assert fb._data is None  # still not loaded


class TestMultibeamRoundTrip:
    def test_write_read_multibeam_beam0(
        self, tmp_path, synthetic_multibeam_filterbank_header,
        synthetic_multibeam_filterbank_data,
    ):
        header = synthetic_multibeam_filterbank_header
        data = synthetic_multibeam_filterbank_data  # (4, 100, 64) uint8
        fpath = str(tmp_path / "test_multibeam.fil")

        write_filterbank(fpath, data, header, nbits=8)

        fb = FilterbankFile(fpath, beam=0)
        assert fb.nbeams == 4
        assert fb.nsamples == 100
        assert fb.data.shape == (100, 64)
        np.testing.assert_array_equal(fb.data.astype(np.uint8), data[0, :, :])

    def test_write_read_multibeam_each_beam(
        self, tmp_path, synthetic_multibeam_filterbank_header,
        synthetic_multibeam_filterbank_data,
    ):
        header = synthetic_multibeam_filterbank_header
        data = synthetic_multibeam_filterbank_data
        fpath = str(tmp_path / "test_multibeam.fil")

        write_filterbank(fpath, data, header, nbits=8)

        for beam_idx in range(4):
            fb = FilterbankFile(fpath, beam=beam_idx, verbose=False)
            np.testing.assert_array_equal(
                fb.data.astype(np.uint8), data[beam_idx, :, :]
            )

    def test_multibeam_default_beam(
        self, tmp_path, synthetic_multibeam_filterbank_header,
        synthetic_multibeam_filterbank_data,
    ):
        """beam=None on a multibeam file should default to beam 0."""
        header = synthetic_multibeam_filterbank_header
        data = synthetic_multibeam_filterbank_data
        fpath = str(tmp_path / "test_multibeam.fil")

        write_filterbank(fpath, data, header, nbits=8)

        fb = FilterbankFile(fpath)
        assert fb.nbeams == 4
        assert fb.data.shape == (100, 64)
        np.testing.assert_array_equal(fb.data.astype(np.uint8), data[0, :, :])

    def test_multibeam_invalid_beam_raises(
        self, tmp_path, synthetic_multibeam_filterbank_header,
        synthetic_multibeam_filterbank_data,
    ):
        header = synthetic_multibeam_filterbank_header
        data = synthetic_multibeam_filterbank_data
        fpath = str(tmp_path / "test_multibeam.fil")

        write_filterbank(fpath, data, header, nbits=8)

        with pytest.raises(ValueError, match="beam=5 out of range"):
            FilterbankFile(fpath, beam=5)

        with pytest.raises(ValueError, match="beam=-1 out of range"):
            FilterbankFile(fpath, beam=-1)

    def test_multibeam_float32(
        self, tmp_path, synthetic_multibeam_filterbank_header,
    ):
        header = synthetic_multibeam_filterbank_header.copy()
        rng = np.random.RandomState(999)
        data = rng.randn(4, 50, 64).astype(np.float32)
        fpath = str(tmp_path / "test_multibeam32.fil")

        write_filterbank(fpath, data, header, nbits=32)

        for beam_idx in range(4):
            fb = FilterbankFile(fpath, beam=beam_idx, verbose=False)
            np.testing.assert_allclose(
                fb.data.astype(np.float32), data[beam_idx, :, :], atol=1e-5
            )


class TestSingleBeamBackwardCompat:
    def test_beam_param_ignored_single_beam(
        self, tmp_path, synthetic_filterbank_header, synthetic_filterbank_data,
    ):
        """beam parameter is ignored for single-beam files."""
        fpath = str(tmp_path / "test_single.fil")
        write_filterbank(fpath, synthetic_filterbank_data, synthetic_filterbank_header, nbits=8)

        fb = FilterbankFile(fpath, beam=0)
        assert fb.nbeams == 1
        assert fb.data.shape == synthetic_filterbank_data.shape
        np.testing.assert_array_equal(fb.data.astype(np.uint8), synthetic_filterbank_data)


class TestMultibeamProperties:
    def test_nbeams_property(
        self, tmp_path, synthetic_multibeam_filterbank_header,
        synthetic_multibeam_filterbank_data,
    ):
        fpath = str(tmp_path / "test_multibeam.fil")
        write_filterbank(
            fpath, synthetic_multibeam_filterbank_data,
            synthetic_multibeam_filterbank_header, nbits=8,
        )

        fb = FilterbankFile(fpath, beam=0, verbose=False)
        assert fb.nbeams == 4
        assert fb.nsamples == 100
        assert fb.nchans == 64

    def test_nbeams_default_single(
        self, tmp_path, synthetic_filterbank_header, synthetic_filterbank_data,
    ):
        fpath = str(tmp_path / "test_single.fil")
        write_filterbank(fpath, synthetic_filterbank_data, synthetic_filterbank_header, nbits=8)

        fb = FilterbankFile(fpath, verbose=False)
        assert fb.nbeams == 1
