"""Tests for upper-triangular baseline indexing."""

import numpy as np
import pytest

from casm_io.correlator.baselines import (
    triu_flat_index,
    triu_to_ij,
    n_baselines,
    build_baseline_plan,
    extract_baselines,
)


class TestTriuFlatIndex:
    """triu_flat_index must match np.triu_indices ordering."""

    @pytest.mark.parametrize("n", [4, 8, 16, 64, 128])
    def test_matches_numpy(self, n):
        """Every (i, j) from np.triu_indices matches our flat index."""
        rows, cols = np.triu_indices(n)
        for flat, (i, j) in enumerate(zip(rows, cols)):
            assert triu_flat_index(n, i, j) == flat, (
                f"n={n}, i={i}, j={j}: expected {flat}, got {triu_flat_index(n, i, j)}"
            )

    def test_raises_on_lower_triangle(self):
        with pytest.raises(ValueError, match="i <= j"):
            triu_flat_index(4, 2, 1)


class TestTriuToIJ:
    """triu_to_ij must round-trip with triu_flat_index."""

    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_round_trip(self, n):
        nbl = n * (n + 1) // 2
        for flat in range(nbl):
            i, j = triu_to_ij(n, flat)
            assert triu_flat_index(n, i, j) == flat
            assert i <= j


class TestNBaselines:
    def test_formula(self):
        assert n_baselines(4) == 10
        assert n_baselines(64) == 2080
        assert n_baselines(128) == 8256


class TestBuildBaselinePlan:
    def test_ref_less_than_target(self):
        """When ref < target, no conjugation needed."""
        bl_idx, conj = build_baseline_plan(ref=0, targets=[1, 2, 3], nsig=4)
        assert len(bl_idx) == 3
        assert not np.any(conj)

    def test_ref_greater_than_target_conjugates(self):
        """When ref > target, conjugation is True."""
        bl_idx, conj = build_baseline_plan(ref=3, targets=[0, 1, 2], nsig=4)
        assert np.all(conj)

    def test_mixed_conjugation(self):
        bl_idx, conj = build_baseline_plan(ref=2, targets=[0, 1, 3], nsig=4)
        # ref=2 > target=0 → conj, ref=2 > target=1 → conj, ref=2 < target=3 → no conj
        np.testing.assert_array_equal(conj, [True, True, False])

    def test_raises_on_invalid_ref(self):
        with pytest.raises(ValueError):
            build_baseline_plan(ref=5, targets=[0], nsig=4)

    def test_raises_on_duplicate_targets(self):
        with pytest.raises(ValueError, match="Duplicate"):
            build_baseline_plan(ref=0, targets=[1, 1], nsig=4)

    def test_raises_on_ref_in_targets(self):
        with pytest.raises(ValueError, match="targets must not include ref"):
            build_baseline_plan(ref=0, targets=[0, 1], nsig=4)


class TestExtractBaselines:
    def test_known_values(self):
        """Extract baselines from synthetic data where each bl has a unique value."""
        nsig = 4
        nbl = n_baselines(nsig)
        ntime, nchan = 3, 8

        # Each baseline gets a unique value: bl_idx as complex
        data = np.zeros((ntime, nchan, nbl), dtype=np.complex64)
        for bl in range(nbl):
            data[:, :, bl] = complex(bl, bl * 10)

        ref = 1
        targets = [0, 2, 3]
        bl_idx, conj = build_baseline_plan(ref, targets, nsig)

        result = extract_baselines(data, bl_idx, conj)
        assert result.shape == (ntime, nchan, 3)

        # ref=1, target=0: triu(0,1) → flat idx 1, conj=True
        expected_01 = np.conj(complex(1, 10))
        np.testing.assert_allclose(result[0, 0, 0], expected_01)

        # ref=1, target=2: triu(1,2) → flat idx 5, conj=False
        expected_12 = complex(5, 50)
        np.testing.assert_allclose(result[0, 0, 1], expected_12)

    def test_4d_input(self):
        """Extract from (T, F, nbl, 2) real/imag format."""
        nsig = 4
        nbl = n_baselines(nsig)
        ntime, nchan = 2, 4
        data = np.zeros((ntime, nchan, nbl, 2), dtype=np.float32)
        data[:, :, :, 0] = 1.0  # real
        data[:, :, :, 1] = 2.0  # imag

        bl_idx, conj = build_baseline_plan(0, [1, 2, 3], nsig)
        result = extract_baselines(data, bl_idx, conj)
        assert result.dtype == np.complex64
        assert result.shape == (ntime, nchan, 3)
        np.testing.assert_allclose(result[0, 0, 0], 1.0 + 2.0j)
