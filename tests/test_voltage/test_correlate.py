"""Tests for casm_io.voltage.correlate()."""

import numpy as np
import pytest

from casm_io.constants import TSAMP_S
from casm_io.voltage import correlate


def _ramp_array(n_time=6, n_chan=4, n_input=3):
    """v[t, f, i] = (t+1) + 1j*i, so products are easy to write down."""
    t = np.arange(1, n_time + 1, dtype=np.float32)[:, None, None]
    i = np.arange(n_input, dtype=np.float32)[None, None, :]
    return np.broadcast_to(t + 1j * i, (n_time, n_chan, n_input)).astype(np.complex64)


class TestCorrelateArray:
    def test_native_resolution_is_pairwise_products(self):
        v = _ramp_array()
        res = correlate(v)
        assert res.tint_samples == 1
        assert res.vis.shape == (6, 4, 3, 3)
        expected = v[:, :, :, None] * np.conj(v[:, :, None, :])
        np.testing.assert_allclose(res.vis, expected)

    def test_integration_averages_products_not_voltages(self):
        v = _ramp_array(n_time=4)
        res = correlate(v, tint_samples=2)
        assert res.vis.shape == (2, 4, 3, 3)
        # autocorrelation of input 0 over t=1,2: mean(1, 4) = 2.5 — averaging
        # the voltages first would give ((1+2)/2)^2 = 2.25
        np.testing.assert_allclose(res.vis[0, 0, 0, 0], 2.5)

    def test_trailing_partial_bin_dropped(self):
        res = correlate(_ramp_array(n_time=7), tint_samples=3)
        assert res.vis.shape[0] == 2

    def test_time_axis_is_bin_centres(self):
        res = correlate(_ramp_array(n_time=6), tint_samples=3)
        np.testing.assert_allclose(
            res.time_s, [1.5 * TSAMP_S, 4.5 * TSAMP_S]
        )

    def test_tint_seconds_rounds_to_samples(self):
        res = correlate(_ramp_array(n_time=6), tint_s=2 * TSAMP_S)
        assert res.tint_samples == 2

    def test_diagonal_is_real_positive(self):
        rng = np.random.RandomState(7)
        v = (rng.randn(8, 4, 3) + 1j * rng.randn(8, 4, 3)).astype(np.complex64)
        res = correlate(v, tint_samples=4)
        auto = np.diagonal(res.vis, axis1=2, axis2=3)
        np.testing.assert_allclose(auto.imag, 0, atol=1e-5)
        assert np.all(auto.real > 0)


class TestCorrelateDict:
    def test_dict_stacks_in_snap_adc_order(self):
        d = {
            3: np.full((4, 2, 2), 3 + 0j, dtype=np.complex64),
            0: np.full((4, 2, 2), 1 + 0j, dtype=np.complex64),
        }
        res = correlate(d, tint_samples=4)
        assert res.inputs == [(0, 0), (0, 1), (3, 0), (3, 1)]
        # cross product of snap 0 (value 1) with snap 3 (value 3)
        np.testing.assert_allclose(res.vis[0, 0, 0, 2], 3.0)

    def test_array_input_has_no_labels(self):
        assert correlate(_ramp_array()).inputs is None


class TestCorrelateErrors:
    def test_both_tint_forms_raise(self):
        with pytest.raises(ValueError, match="not both"):
            correlate(_ramp_array(), tint_s=0.1, tint_samples=2)

    def test_tint_longer_than_data_raises(self):
        with pytest.raises(ValueError, match="longer than"):
            correlate(_ramp_array(n_time=3), tint_samples=10)

    def test_nonpositive_tint_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            correlate(_ramp_array(), tint_samples=0)
