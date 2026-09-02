"""Tests for split_filterbank and requantize_filterbank utilities."""

import numpy as np
import pytest

from casm_io.filterbank.header import read_sigproc_header
from casm_io.filterbank.reader import FilterbankFile
from casm_io.filterbank.writer import write_filterbank
from casm_io.filterbank.split import split_filterbank
from casm_io.filterbank.requantize import requantize_filterbank


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fb_uint8(tmp_path):
    """Write a synthetic 8-bit filterbank (200 samples, 32 channels).

    Channel k has mean ~(50 + 2*k) so channels are distinguishable.
    """
    nsamples, nchans = 200, 32
    rng = np.random.RandomState(42)
    data = np.zeros((nsamples, nchans), dtype=np.uint8)
    for k in range(nchans):
        mean_val = 50 + 2 * k
        data[:, k] = np.clip(
            rng.normal(mean_val, 5, size=nsamples).round(), 0, 255
        ).astype(np.uint8)

    header = {
        "telescope_id": 20,
        "machine_id": 0,
        "data_type": 1,
        "source_name": "SPLIT_TEST",
        "nchans": nchans,
        "nifs": 1,
        "nbits": 8,
        "tsamp": 0.001,
        "tstart": 60000.0,
        "fch1": 468.75,
        "foff": -0.030517578125,
    }
    fpath = str(tmp_path / "input_8bit.fil")
    write_filterbank(fpath, data, header, nbits=8)
    return fpath, data, header


@pytest.fixture
def fb_float32(tmp_path):
    """Write a synthetic 32-bit filterbank (300 samples, 16 channels).

    Includes one dead channel (channel 0 = constant 0.0).
    """
    nsamples, nchans = 300, 16
    rng = np.random.RandomState(99)
    data = rng.randn(nsamples, nchans).astype(np.float32) * 10.0
    # Give channels distinct means
    for k in range(nchans):
        data[:, k] += 100.0 + 5.0 * k
    # Dead channel
    data[:, 0] = 0.0

    header = {
        "telescope_id": 20,
        "machine_id": 0,
        "data_type": 1,
        "source_name": "REQUANT_TEST",
        "nchans": nchans,
        "nifs": 1,
        "nbits": 32,
        "tsamp": 0.002,
        "tstart": 60000.0,
        "fch1": 468.75,
        "foff": -0.5,
    }
    fpath = str(tmp_path / "input_32bit.fil")
    write_filterbank(fpath, data, header, nbits=32)
    return fpath, data, header


# ---------------------------------------------------------------------------
# split_filterbank tests
# ---------------------------------------------------------------------------

class TestSplitFilterbank:
    def test_correct_nsamples(self, tmp_path, fb_uint8):
        """Splitting extracts the requested number of samples."""
        in_path, data, _ = fb_uint8
        out_path = str(tmp_path / "split_out.fil")
        result = split_filterbank(in_path, out_path,
                                  start_sample=50, nsamples=80, verbose=False)
        assert result["nsamples_written"] == 80

        fb = FilterbankFile(out_path, verbose=False)
        assert fb.nsamples == 80
        assert fb.data.shape == (80, 32)
        np.testing.assert_array_equal(
            fb.data.astype(np.uint8), data[50:130, :]
        )

    def test_tstart_adjustment(self, tmp_path, fb_uint8):
        """tstart is shifted forward by start_sample * tsamp / 86400."""
        in_path, _, header = fb_uint8
        out_path = str(tmp_path / "split_tstart.fil")
        start = 100
        result = split_filterbank(in_path, out_path,
                                  start_sample=start, nsamples=10, verbose=False)
        expected_tstart = header["tstart"] + start * header["tsamp"] / 86400.0
        assert abs(result["header"]["tstart"] - expected_tstart) < 1e-12

        fb = FilterbankFile(out_path, verbose=False)
        assert abs(fb.header["tstart"] - expected_tstart) < 1e-9

    def test_out_of_range_start_raises(self, tmp_path, fb_uint8):
        """start_sample beyond file length raises ValueError."""
        in_path, _, _ = fb_uint8
        out_path = str(tmp_path / "bad.fil")
        with pytest.raises(ValueError, match="start_sample.*out of range"):
            split_filterbank(in_path, out_path,
                             start_sample=9999, nsamples=1, verbose=False)

    def test_out_of_range_nsamples_raises(self, tmp_path, fb_uint8):
        """Requesting more samples than available raises ValueError."""
        in_path, _, _ = fb_uint8
        out_path = str(tmp_path / "bad2.fil")
        with pytest.raises(ValueError, match="only.*available"):
            split_filterbank(in_path, out_path,
                             start_sample=190, nsamples=20, verbose=False)

    def test_preserves_nbits(self, tmp_path, fb_uint8):
        """Output file preserves the original nbits."""
        in_path, _, _ = fb_uint8
        out_path = str(tmp_path / "split_nbits.fil")
        result = split_filterbank(in_path, out_path,
                                  start_sample=0, nsamples=10, verbose=False)
        assert result["header"]["nbits"] == 8

        fb = FilterbankFile(out_path, verbose=False)
        assert fb.header["nbits"] == 8

    def test_default_nsamples_reads_to_end(self, tmp_path, fb_uint8):
        """nsamples=None reads from start_sample to end of file."""
        in_path, data, _ = fb_uint8
        out_path = str(tmp_path / "split_toend.fil")
        result = split_filterbank(in_path, out_path,
                                  start_sample=150, verbose=False)
        assert result["nsamples_written"] == 50

        fb = FilterbankFile(out_path, verbose=False)
        np.testing.assert_array_equal(
            fb.data.astype(np.uint8), data[150:, :]
        )


# ---------------------------------------------------------------------------
# requantize_filterbank tests
# ---------------------------------------------------------------------------

class TestRequantizeFilterbank:
    def test_output_dtype_uint8(self, tmp_path, fb_float32):
        """Requantized 8-bit output records nbits=8 and data fits uint8."""
        in_path, _, _ = fb_float32
        out_path = str(tmp_path / "requant8.fil")
        requantize_filterbank(in_path, out_path, target_nbits=8, verbose=False)

        fb = FilterbankFile(out_path, verbose=False)
        assert fb.header["nbits"] == 8
        # Reader may return float32 in memory; verify values are valid uint8
        raw = fb.data.astype(np.uint8)
        assert raw.min() >= 0
        assert raw.max() <= 255

    def test_per_channel_median_near_128(self, tmp_path, fb_float32):
        """Alive channels should have per-channel median near 128."""
        in_path, data, _ = fb_float32
        out_path = str(tmp_path / "requant_median.fil")
        requantize_filterbank(in_path, out_path,
                              target_nbits=8, verbose=False)

        fb = FilterbankFile(out_path, verbose=False)
        out_data = fb.data.astype(np.float64)
        # Channels 1..15 are alive (channel 0 is dead)
        alive_medians = np.median(out_data[:, 1:], axis=0)
        # Median should be near 128 (127.5 rounded); allow tolerance of 2
        np.testing.assert_allclose(alive_medians, 128, atol=2)

    def test_dead_channels_zero(self, tmp_path, fb_float32):
        """Dead channels (zero std) should be all zeros in output."""
        in_path, _, _ = fb_float32
        out_path = str(tmp_path / "requant_dead.fil")
        requantize_filterbank(in_path, out_path, target_nbits=8, verbose=False)

        fb = FilterbankFile(out_path, verbose=False)
        dead_col = fb.data[:, 0]
        assert np.all(dead_col == 0)

    def test_no_sigpyproc_problematic_keys(self, tmp_path, fb_float32):
        """On-disk header should not contain keys that break sigpyproc.

        We read the raw SIGPROC header to avoid derived keys that
        sigpyproc's reader adds in memory (bandwidth, ftop, etc.).
        """
        in_path, _, _ = fb_float32
        out_path = str(tmp_path / "requant_header.fil")
        requantize_filterbank(in_path, out_path, target_nbits=8, verbose=False)

        raw_header, _ = read_sigproc_header(out_path)
        bad_keys = {"accel", "bandwidth", "fbottom", "ftop",
                    "obs_time", "period", "tobs"}
        present = bad_keys & set(raw_header.keys())
        assert len(present) == 0, f"Unexpected keys in on-disk header: {present}"

    def test_requantize_uint8_to_uint8(self, tmp_path, fb_uint8):
        """Requantizing 8-bit -> 8-bit still works."""
        in_path, _, _ = fb_uint8
        out_path = str(tmp_path / "requant_8to8.fil")
        requantize_filterbank(in_path, out_path, target_nbits=8, verbose=False)

        fb = FilterbankFile(out_path, verbose=False)
        assert fb.header["nbits"] == 8
        assert fb.nsamples == 200


# ---------------------------------------------------------------------------
# Round-trip: split then requantize
# ---------------------------------------------------------------------------

class TestSplitThenRequantize:
    def test_roundtrip_readable(self, tmp_path, fb_float32):
        """Split a 32-bit file, requantize to 8-bit, verify readable."""
        in_path, data, _ = fb_float32
        split_path = str(tmp_path / "rt_split.fil")
        requant_path = str(tmp_path / "rt_requant.fil")

        # Split: take samples 50..249 (200 samples)
        split_result = split_filterbank(
            in_path, split_path,
            start_sample=50, nsamples=200, verbose=False,
        )
        assert split_result["nsamples_written"] == 200

        # Requantize the split output
        requantize_filterbank(
            split_path, requant_path,
            target_nbits=8, verbose=False,
        )

        # Verify the final file is fully readable
        fb = FilterbankFile(requant_path, verbose=False)
        assert fb.nsamples == 200
        assert fb.nchans == 16
        assert fb.header["nbits"] == 8
        assert fb.data.shape == (200, 16)

        # Cast to uint8 for value checks (reader may use float32 in memory)
        out_data = fb.data.astype(np.uint8)

        # Dead channel (channel 0) should be zero after requantization
        assert np.all(out_data[:, 0] == 0)

        # Alive channels should have reasonable values (not all 0 or 255)
        alive_data = out_data[:, 1:]
        assert alive_data.min() < 50
        assert alive_data.max() > 200

    def test_roundtrip_preserves_metadata(self, tmp_path, fb_uint8):
        """Split then requantize preserves key metadata fields."""
        in_path, _, header = fb_uint8
        split_path = str(tmp_path / "rt2_split.fil")
        requant_path = str(tmp_path / "rt2_requant.fil")

        split_filterbank(in_path, split_path,
                         start_sample=0, nsamples=50, verbose=False)
        requantize_filterbank(split_path, requant_path,
                              target_nbits=8, verbose=False)

        fb = FilterbankFile(requant_path, verbose=False)
        assert fb.header["nchans"] == header["nchans"]
        assert abs(fb.header["fch1"] - header["fch1"]) < 1e-6
        assert abs(fb.header["foff"] - header["foff"]) < 1e-12
        assert abs(fb.header["tsamp"] - header["tsamp"]) < 1e-12
