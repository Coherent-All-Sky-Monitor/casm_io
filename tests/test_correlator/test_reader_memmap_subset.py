"""
Equivalence tests for the memmap baseline-subset read path.

The reader materializes only the requested channels and baselines through
np.memmap, and reads files in parallel. Both must produce bit-identical
output to the original full-file read (np.fromfile of the whole file, reshape,
then subset). The reference implementation below reproduces that original
algorithm so the comparison survives after the old code is gone.

Baseline ordering and conjugation follow docs/CONVENTIONS.md.
"""

import os

import numpy as np
import pytest

from casm_io.correlator import baselines
from casm_io.correlator.formats import VisibilityFormat, load_format
from casm_io.correlator.header import get_header_offset
from casm_io.correlator.mapping import AntennaMapping
from casm_io.correlator.reader import VisibilityReader

REAL_DAT = "/mnt/nvme4/data/casm/visibilities_64ant/2026-08-25-04:38:03.dat.60"
REAL_DIR = os.path.dirname(REAL_DAT)
REAL_BASE = "2026-08-25-04:38:03"
LAYOUT_CSV = (
    "/home/casm/software/dev/antenna_layouts/"
    "casm_antenna_layout_2026-08-31_ant18admitted.csv"
)

needs_real_data = pytest.mark.skipif(
    not (os.path.exists(REAL_DAT) and os.path.exists(LAYOUT_CSV)),
    reason="real visibility file or antenna layout not present on this machine",
)


# ---------------------------------------------------------------------------
# Reference implementation: the original full-file read
# ---------------------------------------------------------------------------

def reference_read_file(
    fpath, fmt, s0, s1, bl_idx=None, bl_conj=None,
    ch_start=0, ch_end=None, freq_order="descending",
):
    """Original algorithm: read the whole file, reshape, then subset."""
    if ch_end is None:
        ch_end = fmt.nchan
    nbaseline = fmt.n_baselines
    offset, _ = get_header_offset(fpath)
    denom = fmt.nchan * nbaseline * 2

    raw = np.fromfile(fpath, dtype=np.int32, offset=offset)
    assert raw.size % denom == 0
    ntime = raw.size // denom
    s1_capped = min(s1, ntime)
    xcorrs = raw.reshape(ntime, fmt.nchan, nbaseline, 2)
    del raw
    xcorrs = xcorrs[s0:s1_capped, :, :, :]
    xcorrs = xcorrs[:, ch_start:ch_end, :, :]

    if bl_idx is not None:
        xsel = xcorrs[:, :, bl_idx, :]
        del xcorrs
        v = xsel[..., 0].astype(np.float32) + 1j * xsel[..., 1].astype(np.float32)
        del xsel
        if np.any(bl_conj):
            v[:, :, bl_conj] = np.conj(v[:, :, bl_conj])
    else:
        v = xcorrs[..., 0].astype(np.float32) + 1j * xcorrs[..., 1].astype(np.float32)
        del xcorrs

    if freq_order == "ascending":
        v = v[:, ::-1, :]
    return v.astype(np.complex64)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_format():
    """6 inputs (21 baselines), 12 channels, 3 integrations per file."""
    chan_bw = 0.030517578125
    return VisibilityFormat(
        name="test_synth",
        nsig=6,
        dt_raw_s=2.0,
        ntime_per_file=3,
        nchan=12,
        chan_bw_mhz=chan_bw,
        freq_top_mhz=468.75,
        freq_bottom_mhz=468.75 - 12 * chan_bw,
        native_order="descending",
    )


def _write_synth_files(tmp_path, fmt, base_str, n_files, seed=0):
    """Write n_files headerless .dat files with deterministic int32 content."""
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n_files):
        data = rng.integers(
            -2_000_000, 2_000_000,
            size=(fmt.ntime_per_file, fmt.nchan, fmt.n_baselines, 2),
            dtype=np.int32,
        )
        p = tmp_path / f"{base_str}.{i}"
        p.write_bytes(data.tobytes())
        paths.append(str(p))
    return paths


class TestSynthetic:
    """The subset path must equal the original full-read path on synthetic data."""

    def test_baseline_subset_matches_reference(self, tmp_path, synth_format):
        fmt = synth_format
        base_str = "2026-08-25-04:38:03"
        paths = _write_synth_files(tmp_path, fmt, base_str, 1)

        ref, targets = 2, [0, 1, 3, 5]
        bl_idx, bl_conj = baselines.build_baseline_plan(ref, targets, fmt.nsig)
        expected = reference_read_file(
            paths[0], fmt, 0, fmt.ntime_per_file, bl_idx, bl_conj
        )

        rdr = VisibilityReader(str(tmp_path), base_str, fmt)
        got = rdr.read(ref=ref, targets=targets, verbose=False)

        assert got.vis.dtype == np.complex64
        assert got.vis.shape == expected.shape
        assert np.array_equal(got.vis, expected)

    def test_baseline_subset_with_channels(self, tmp_path, synth_format):
        fmt = synth_format
        base_str = "2026-08-25-04:38:03"
        paths = _write_synth_files(tmp_path, fmt, base_str, 1, seed=1)

        ref, targets = 4, [0, 5]
        bl_idx, bl_conj = baselines.build_baseline_plan(ref, targets, fmt.nsig)
        expected = reference_read_file(
            paths[0], fmt, 0, fmt.ntime_per_file, bl_idx, bl_conj,
            ch_start=3, ch_end=9,
        )

        rdr = VisibilityReader(str(tmp_path), base_str, fmt)
        got = rdr.read(ref=ref, targets=targets, channels=(3, 9), verbose=False)
        assert np.array_equal(got.vis, expected)

    def test_ascending_subset_matches_reference(self, tmp_path, synth_format):
        fmt = synth_format
        base_str = "2026-08-25-04:38:03"
        paths = _write_synth_files(tmp_path, fmt, base_str, 1, seed=2)

        ref, targets = 0, [3, 1]
        bl_idx, bl_conj = baselines.build_baseline_plan(ref, targets, fmt.nsig)
        expected = reference_read_file(
            paths[0], fmt, 0, fmt.ntime_per_file, bl_idx, bl_conj,
            freq_order="ascending",
        )

        rdr = VisibilityReader(str(tmp_path), base_str, fmt)
        got = rdr.read(
            ref=ref, targets=targets, freq_order="ascending", verbose=False
        )
        assert np.array_equal(got.vis, expected)

    def test_no_subset_still_full_read(self, tmp_path, synth_format):
        fmt = synth_format
        base_str = "2026-08-25-04:38:03"
        paths = _write_synth_files(tmp_path, fmt, base_str, 1, seed=3)

        expected = reference_read_file(paths[0], fmt, 0, fmt.ntime_per_file)
        rdr = VisibilityReader(str(tmp_path), base_str, fmt)
        got = rdr.read(verbose=False)
        assert np.array_equal(got.vis, expected)

    def test_workers_do_not_change_result(self, tmp_path, synth_format):
        fmt = synth_format
        base_str = "2026-08-25-04:38:03"
        _write_synth_files(tmp_path, fmt, base_str, 4, seed=4)

        rdr = VisibilityReader(str(tmp_path), base_str, fmt)
        kw = dict(ref=1, targets=[0, 2, 4], verbose=False)
        seq = rdr.read(workers=1, **kw)
        par = rdr.read(workers=4, **kw)

        assert np.array_equal(seq.vis, par.vis)
        assert np.array_equal(seq.time_unix, par.time_unix)
        assert np.array_equal(seq.freq_mhz, par.freq_mhz)
        assert seq.metadata == par.metadata

    def test_workers_env_override(self, tmp_path, synth_format, monkeypatch):
        fmt = synth_format
        base_str = "2026-08-25-04:38:03"
        _write_synth_files(tmp_path, fmt, base_str, 3, seed=5)

        rdr = VisibilityReader(str(tmp_path), base_str, fmt)
        expected = rdr.read(workers=1, ref=0, targets=[2], verbose=False)

        monkeypatch.setenv("CASM_IO_WORKERS", "3")
        got = rdr.read(ref=0, targets=[2], verbose=False)
        assert np.array_equal(got.vis, expected.vis)

    def test_process_executor_matches(self, tmp_path, synth_format, monkeypatch):
        fmt = synth_format
        base_str = "2026-08-25-04:38:03"
        _write_synth_files(tmp_path, fmt, base_str, 3, seed=6)

        rdr = VisibilityReader(str(tmp_path), base_str, fmt)
        expected = rdr.read(workers=1, ref=3, targets=[0, 1], verbose=False)

        monkeypatch.setenv("CASM_IO_EXECUTOR", "process")
        got = rdr.read(workers=3, ref=3, targets=[0, 1], verbose=False)
        assert np.array_equal(got.vis, expected.vis)
        assert np.array_equal(got.time_unix, expected.time_unix)

    def test_zero_fill_with_workers(self, tmp_path, synth_format):
        fmt = synth_format
        base_str = "2026-08-25-04:38:03"
        _write_synth_files(tmp_path, fmt, base_str, 3, seed=7)
        os.remove(tmp_path / f"{base_str}.1")

        rdr = VisibilityReader(str(tmp_path), base_str, fmt)
        with pytest.warns(UserWarning):
            seq = rdr.read(nfiles=3, ref=0, targets=[1, 2], workers=1, verbose=False)
        with pytest.warns(UserWarning):
            par = rdr.read(nfiles=3, ref=0, targets=[1, 2], workers=3, verbose=False)

        assert np.array_equal(seq.vis, par.vis)
        assert np.array_equal(seq.time_unix, par.time_unix)
        mid = seq.vis[fmt.ntime_per_file:2 * fmt.ntime_per_file]
        assert np.all(mid == 0)

    def test_bad_workers_rejected(self, tmp_path, synth_format):
        fmt = synth_format
        base_str = "2026-08-25-04:38:03"
        _write_synth_files(tmp_path, fmt, base_str, 1, seed=8)
        rdr = VisibilityReader(str(tmp_path), base_str, fmt)
        with pytest.raises(ValueError):
            rdr.read(ref=0, targets=[1], workers=0, verbose=False)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

def _layout_ref_targets(fmt):
    """Reference input and target inputs from the admitted-antenna layout."""
    ant = AntennaMapping.load(LAYOUT_CSV)
    df = ant.dataframe
    admitted = df[df["include_in_beamforming"].astype(int) == 1]
    pidx = sorted(int(p) for p in admitted["packet_index"] if 0 <= int(p) < fmt.nsig)
    assert len(pidx) >= 2, "layout has fewer than two admitted inputs"
    return pidx[0], pidx[1:]


@needs_real_data
class TestRealFile:
    """Bit-exact equivalence on one real 6.5 GB visibility file."""

    def test_layout_subset_full_band(self):
        fmt = load_format("layout_64ant")
        ref, targets = _layout_ref_targets(fmt)
        bl_idx, bl_conj = baselines.build_baseline_plan(ref, targets, fmt.nsig)

        rdr = VisibilityReader(REAL_DIR, REAL_BASE, fmt)
        got = rdr.read(nfiles=1, skip_nfiles=60, ref=ref, targets=targets,
                       verbose=False)
        expected = reference_read_file(
            REAL_DAT, fmt, 0, fmt.ntime_per_file, bl_idx, bl_conj
        )
        assert got.vis.shape == expected.shape
        assert np.array_equal(got.vis, expected)

    def test_layout_subset_channel_slice(self):
        fmt = load_format("layout_64ant")
        ref, targets = _layout_ref_targets(fmt)
        bl_idx, bl_conj = baselines.build_baseline_plan(ref, targets, fmt.nsig)

        rdr = VisibilityReader(REAL_DIR, REAL_BASE, fmt)
        got = rdr.read(nfiles=1, skip_nfiles=60, ref=ref, targets=targets,
                       channels=(1000, 1500), verbose=False)
        expected = reference_read_file(
            REAL_DAT, fmt, 0, fmt.ntime_per_file, bl_idx, bl_conj,
            ch_start=1000, ch_end=1500,
        )
        assert got.vis.shape == expected.shape
        assert np.array_equal(got.vis, expected)
        assert got.metadata["channels"] == (1000, 1500)

    def test_three_file_span_parallel_matches_sequential(self):
        fmt = load_format("layout_64ant")
        ref, targets = _layout_ref_targets(fmt)

        rdr = VisibilityReader(REAL_DIR, REAL_BASE, fmt)
        kw = dict(nfiles=3, skip_nfiles=60, ref=ref, targets=targets,
                  channels=(1000, 1500), verbose=False)
        seq = rdr.read(workers=1, **kw)
        par = rdr.read(workers=3, **kw)

        assert np.array_equal(seq.vis, par.vis)
        assert np.array_equal(seq.freq_mhz, par.freq_mhz)
        assert np.array_equal(seq.time_unix, par.time_unix)
        assert seq.metadata == par.metadata
        # The span must actually cross file boundaries
        assert len(seq.metadata["files"]) == 3


# ---------------------------------------------------------------------------
# inputs= sub-triangle selection
# ---------------------------------------------------------------------------

class TestInputSubset:
    """`inputs=` returns the upper triangle of the selected inputs."""

    def test_matches_full_read(self, tmp_path, synth_format):
        from casm_io.correlator.baselines import triu_flat_index

        fmt = synth_format
        base_str = "2026-08-25-04:38:03"
        paths = _write_synth_files(tmp_path, fmt, base_str, 1, seed=11)
        full = reference_read_file(paths[0], fmt, 0, fmt.ntime_per_file)

        sel = [1, 4, 5]
        rdr = VisibilityReader(str(tmp_path), base_str, fmt)
        got = rdr.read(inputs=sel, verbose=False)

        assert got.vis.shape[2] == len(sel) * (len(sel) + 1) // 2
        assert got.metadata["inputs"] == sel
        assert got.metadata["nsig_subset"] == len(sel)
        for ri, a in enumerate(sel):
            for rj, b in enumerate(sel):
                if rj < ri:
                    continue
                k_sub = triu_flat_index(len(sel), ri, rj)
                k_full = triu_flat_index(fmt.nsig, a, b)
                assert np.array_equal(got.vis[:, :, k_sub], full[:, :, k_full])

    def test_unsorted_and_duplicates_collapse(self, tmp_path, synth_format):
        fmt = synth_format
        base_str = "2026-08-25-04:38:03"
        _write_synth_files(tmp_path, fmt, base_str, 1, seed=12)
        rdr = VisibilityReader(str(tmp_path), base_str, fmt)
        a = rdr.read(inputs=[5, 1, 4], verbose=False)
        b = rdr.read(inputs=[1, 4, 5, 5], verbose=False)
        assert np.array_equal(a.vis, b.vis)
        assert a.metadata["inputs"] == [1, 4, 5]

    def test_inputs_with_channels_and_workers(self, tmp_path, synth_format):
        fmt = synth_format
        base_str = "2026-08-25-04:38:03"
        _write_synth_files(tmp_path, fmt, base_str, 3, seed=13)
        rdr = VisibilityReader(str(tmp_path), base_str, fmt)
        kw = dict(inputs=[0, 2, 3], channels=(2, 10), verbose=False)
        seq = rdr.read(workers=1, **kw)
        par = rdr.read(workers=3, **kw)
        assert np.array_equal(seq.vis, par.vis)
        assert seq.metadata == par.metadata

    def test_inputs_and_ref_conflict(self, tmp_path, synth_format):
        fmt = synth_format
        base_str = "2026-08-25-04:38:03"
        _write_synth_files(tmp_path, fmt, base_str, 1, seed=14)
        rdr = VisibilityReader(str(tmp_path), base_str, fmt)
        with pytest.raises(ValueError):
            rdr.read(inputs=[0, 1], ref=0, targets=[1], verbose=False)

    def test_inputs_out_of_range(self, tmp_path, synth_format):
        fmt = synth_format
        base_str = "2026-08-25-04:38:03"
        _write_synth_files(tmp_path, fmt, base_str, 1, seed=15)
        rdr = VisibilityReader(str(tmp_path), base_str, fmt)
        with pytest.raises(ValueError):
            rdr.read(inputs=[0, fmt.nsig], verbose=False)
