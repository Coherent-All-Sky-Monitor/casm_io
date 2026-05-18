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
        assert abs(freq[0] - fmt.freq_top_mhz) < 1e-6
        assert freq[0] > freq[-1]
        # Channel 0 is at freq_top_mhz, channel 3071 is freq_top - 3071 * chan_bw.
        assert abs(freq[-1] - (fmt.freq_top_mhz - 3071 * fmt.chan_bw_mhz)) < 1e-4

    def test_ascending_is_reverse(self):
        fmt = load_format("layout_64ant")
        desc = fmt.get_frequency_axis(order="descending")
        asc = fmt.get_frequency_axis(order="ascending")
        np.testing.assert_array_almost_equal(asc, desc[::-1])

    def test_ascending_first_at_band_bottom(self):
        fmt = load_format("layout_64ant")
        freq = fmt.get_frequency_axis(order="ascending")
        # First (lowest) channel should be just above freq_bottom_mhz.
        assert freq[0] < fmt.freq_bottom_mhz + 1.0
        # Last (highest) channel should be at freq_top_mhz.
        assert abs(freq[-1] - fmt.freq_top_mhz) < 1e-4

    def test_channel_spacing(self):
        fmt = load_format("layout_64ant")
        freq = fmt.get_frequency_axis(order="descending")
        diffs = np.diff(freq)
        np.testing.assert_allclose(diffs, -fmt.chan_bw_mhz, atol=1e-10)


class TestFreqToChannel:
    """Tests for freq_to_channel and freq_range_to_channels methods."""

    def test_freq_to_channel_top(self):
        fmt = load_format("layout_64ant")
        assert fmt.freq_to_channel(fmt.freq_top_mhz) == 0

    def test_freq_to_channel_bottom(self):
        fmt = load_format("layout_64ant")
        # Bottom channel center = freq_top - 3071 * chan_bw, i.e. channel 3071.
        bottom = fmt.freq_top_mhz - 3071 * fmt.chan_bw_mhz
        assert fmt.freq_to_channel(bottom) == 3071

    def test_freq_to_channel_mid(self):
        fmt = load_format("layout_64ant")
        freq_500 = fmt.freq_top_mhz - 500 * fmt.chan_bw_mhz
        assert fmt.freq_to_channel(freq_500) == 500

    def test_freq_to_channel_out_of_band_raises(self):
        fmt = load_format("layout_64ant")
        with pytest.raises(ValueError, match="above band top"):
            fmt.freq_to_channel(500.0)
        with pytest.raises(ValueError, match="below band bottom"):
            fmt.freq_to_channel(300.0)

    def test_freq_range_to_channels(self):
        fmt = load_format("layout_64ant")
        ch_start, ch_end = fmt.freq_range_to_channels(450.0, 460.0)
        # 460 MHz -> ch ~286, 450 MHz -> ch ~614
        assert ch_start < ch_end
        freq = fmt.get_frequency_axis(order="descending")
        assert freq[ch_start] <= 460.0 + fmt.chan_bw_mhz
        assert freq[ch_end - 1] >= 450.0 - fmt.chan_bw_mhz

    def test_freq_range_invalid_order_raises(self):
        fmt = load_format("layout_64ant")
        with pytest.raises(ValueError, match="must be less than"):
            fmt.freq_range_to_channels(460.0, 450.0)

    def test_freq_range_round_trip(self):
        """freq_range_to_channels -> get_frequency_axis slice should be consistent."""
        fmt = load_format("layout_64ant")
        ch_start, ch_end = fmt.freq_range_to_channels(420.0, 425.0)
        freq_slice = fmt.get_frequency_axis(order="descending")[ch_start:ch_end]
        # All frequencies in slice should be within [420, 425] +/- one channel
        assert freq_slice.min() >= 420.0 - fmt.chan_bw_mhz
        assert freq_slice.max() <= 425.0 + fmt.chan_bw_mhz