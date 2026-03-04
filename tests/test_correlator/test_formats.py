"""Tests for correlator format loading and frequency axis."""

import numpy as np
import pytest

from casm_io.correlator.formats import load_format, VisibilityFormat


class TestLoadFormat:
    def test_load_post_jan27(self):
        fmt = load_format("layout_64ant")
        assert fmt.nsig == 128
        assert fmt.nchan == 3072
        assert fmt.ntime_per_file == 32
        assert abs(fmt.dt_raw_s - 137.438953472) < 1e-6

    def test_load_pre_jan27(self):
        fmt = load_format("layout_32ant")
        assert fmt.nsig == 64
        assert abs(fmt.dt_raw_s - 34.359738368) < 1e-6

    def test_nonexistent_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            load_format("does_not_exist")

    def test_n_baselines(self):
        fmt = load_format("layout_64ant")
        assert fmt.n_baselines == 128 * 129 // 2  # 8256

        fmt = load_format("layout_32ant")
        assert fmt.n_baselines == 64 * 65 // 2  # 2080


class TestFrequencyAxis:
    def test_descending_starts_at_top(self):
        fmt = load_format("layout_64ant")
        freq = fmt.get_frequency_axis(order="descending")
        assert freq.shape == (3072,)
        assert abs(freq[0] - 468.75) < 1e-6
        assert freq[0] > freq[-1]
        assert abs(freq[-1] - (468.75 - 3071 * 0.030517578125)) < 1e-4

    def test_ascending_is_reverse(self):
        fmt = load_format("layout_64ant")
        desc = fmt.get_frequency_axis(order="descending")
        asc = fmt.get_frequency_axis(order="ascending")
        np.testing.assert_array_almost_equal(asc, desc[::-1])

    def test_ascending_starts_near_375(self):
        fmt = load_format("layout_64ant")
        freq = fmt.get_frequency_axis(order="ascending")
        # First channel should be near 375 MHz
        assert freq[0] < 376.0
        assert freq[-1] > 468.0

    def test_channel_spacing(self):
        fmt = load_format("layout_64ant")
        freq = fmt.get_frequency_axis(order="descending")
        diffs = np.diff(freq)
        np.testing.assert_allclose(diffs, -fmt.chan_bw_mhz, atol=1e-10)
