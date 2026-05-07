"""Tests for Phase 1 additive changes: layout-path resolver, OVRO dataclass,
C_LIGHT_M_PER_NS, slot_table, with_snap_output, is_in_beamforming."""

import os
from pathlib import Path

import pandas as pd
import pytest

from casm_io.constants import (
    C_LIGHT_M_S,
    C_LIGHT_M_PER_NS,
    OVRO,
    OVRO_LAT_DEG,
    OVRO_LON_DEG,
    OVRO_ELEV_M,
    resolve_layout_path,
)
from casm_io.correlator.mapping import AntennaMapping, DualLayout


class TestConstants:
    def test_c_light_m_per_ns_value(self):
        assert C_LIGHT_M_PER_NS == pytest.approx(C_LIGHT_M_S * 1e-9)
        # Sanity: roughly 0.3 m / ns
        assert 0.29 < C_LIGHT_M_PER_NS < 0.31

    def test_ovro_dataclass_attributes(self):
        assert OVRO.lat_deg == OVRO_LAT_DEG
        assert OVRO.lon_deg == OVRO_LON_DEG
        assert OVRO.elevation_m == OVRO_ELEV_M
        # Alias for callers that used `.alt_m`
        assert OVRO.alt_m == OVRO_ELEV_M

    def test_ovro_is_frozen(self):
        with pytest.raises(Exception):
            OVRO.lat_deg = 0.0


class TestResolveLayoutPath:
    def test_explicit_path_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CASM_LAYOUT_CSV", "/should/not/be/used")
        explicit = tmp_path / "layout.csv"
        explicit.write_text("antenna_id,snap_id,adc,packet_index\n1,0,0,0\n")
        assert resolve_layout_path(explicit) == explicit

    def test_env_var_used_when_no_explicit(self, tmp_path, monkeypatch):
        env_csv = tmp_path / "from_env.csv"
        env_csv.write_text("x")
        monkeypatch.setenv("CASM_LAYOUT_CSV", str(env_csv))
        assert resolve_layout_path() == Path(str(env_csv))

    def test_symlink_used_when_no_explicit_no_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CASM_LAYOUT_CSV", raising=False)
        layout_dir = tmp_path
        target = layout_dir / "casm_antenna_layout_2026-05-07.csv"
        target.write_text("x")
        (layout_dir / "current").symlink_to(target)
        monkeypatch.setenv("CASM_LAYOUT_DIR", str(layout_dir))
        resolved = resolve_layout_path()
        assert resolved.exists()
        assert resolved.resolve() == target.resolve()

    def test_raises_if_nothing_resolves(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CASM_LAYOUT_CSV", raising=False)
        monkeypatch.setenv("CASM_LAYOUT_DIR", str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError, match="No antenna-layout CSV"):
            resolve_layout_path()


class TestAntennaMappingDefaultLoad:
    def test_load_with_path_arg_unchanged(self, antenna_csv_standard):
        ant = AntennaMapping.load(antenna_csv_standard)
        assert ant.n_antennas == 4

    def test_load_default_resolves_via_env(
        self, antenna_csv_standard, monkeypatch
    ):
        monkeypatch.setenv("CASM_LAYOUT_CSV", antenna_csv_standard)
        ant = AntennaMapping.load()
        assert ant.n_antennas == 4

    def test_load_feng_idx_legacy_alias(self, tmp_path):
        df = pd.DataFrame({
            "antenna_id": [1, 2],
            "snap_id": [0, 0],
            "adc": [0, 1],
            "feng_idx": [10, 20],
        })
        path = tmp_path / "feng.csv"
        df.to_csv(path, index=False)
        ant = AntennaMapping.load(path)
        assert ant.packet_index(1) == 10
        assert ant.packet_index(2) == 20


class TestIsInBeamforming:
    def test_uses_dedicated_column_when_present(self, tmp_path):
        df = pd.DataFrame({
            "antenna_id": [1, 2, 3],
            "snap_id": [0, 0, 0],
            "adc": [0, 1, 2],
            "packet_index": [10, 20, 30],
            "functional": [1, 1, 1],
            "include_in_beamforming": [1, 0, 1],
        })
        path = tmp_path / "bf.csv"
        df.to_csv(path, index=False)
        ant = AntennaMapping.load(path)
        assert ant.is_in_beamforming(1) is True
        assert ant.is_in_beamforming(2) is False
        assert ant.is_in_beamforming(3) is True

    def test_falls_back_to_functional(self, antenna_csv_standard):
        # antenna 3 has functional=0 in the standard fixture
        ant = AntennaMapping.load(antenna_csv_standard)
        assert ant.is_in_beamforming(1) is True
        assert ant.is_in_beamforming(3) is False

    def test_default_true_when_no_columns(self, antenna_csv_legacy, tmp_path):
        # legacy fixture has no functional/include_in_beamforming;
        # but legacy has packet_idx not packet_index, so build a clean one
        df = pd.DataFrame({
            "antenna_id": [1],
            "snap_id": [0],
            "adc": [0],
            "packet_index": [10],
        })
        path = tmp_path / "min.csv"
        df.to_csv(path, index=False)
        ant = AntennaMapping.load(path)
        assert ant.is_in_beamforming(1) is True


class TestSlotTable:
    def test_shape(self, antenna_csv_standard):
        ant = AntennaMapping.load(antenna_csv_standard)
        # standard fixture: snap_id in {0,1}, adc in {0,1}; default 6 snaps * 12 adc
        tbl = ant.slot_table()
        assert len(tbl) == 6 * 12

    def test_wired_slots_match(self, antenna_csv_standard):
        ant = AntennaMapping.load(antenna_csv_standard)
        tbl = ant.slot_table()
        # antenna 1 is at (snap=0, adc=0) -> slot 0
        assert int(tbl.loc[0, "antenna_id"]) == 1
        # antenna 4 is at (snap=1, adc=1) -> slot 13
        assert int(tbl.loc[13, "antenna_id"]) == 4

    def test_unwired_slots_filled(self, antenna_csv_standard):
        ant = AntennaMapping.load(antenna_csv_standard)
        tbl = ant.slot_table()
        # slot 5 (snap=0, adc=5) is not wired
        assert int(tbl.loc[5, "antenna_id"]) == -1
        assert int(tbl.loc[5, "snap_id"]) == 0
        assert int(tbl.loc[5, "adc"]) == 5

    def test_custom_dimensions(self, antenna_csv_standard):
        ant = AntennaMapping.load(antenna_csv_standard)
        tbl = ant.slot_table(n_snaps=2, n_adc=4)
        assert len(tbl) == 8


class TestWithInactive:
    def test_with_inactive_drops_listed(self, antenna_csv_standard):
        ant = AntennaMapping.load(antenna_csv_standard)
        original_active = ant.active_antennas()
        # standard fixture: ants 1,2,3,4 with #3 already functional=0
        ant2 = ant.with_inactive([1, 4])
        new_active = ant2.active_antennas()
        # 1 and 4 dropped; 3 was already inactive; 2 remains
        assert new_active == [2]

    def test_original_unchanged(self, antenna_csv_standard):
        ant = AntennaMapping.load(antenna_csv_standard)
        before = ant.active_antennas()
        _ = ant.with_inactive([1, 2])
        after = ant.active_antennas()
        assert before == after

    def test_unknown_id_silently_ignored(self, antenna_csv_standard):
        ant = AntennaMapping.load(antenna_csv_standard)
        ant2 = ant.with_inactive([999, 1])
        # only 1 should drop; 999 is no-op
        assert 1 not in ant2.active_antennas()
        assert 2 in ant2.active_antennas()

    def test_with_only_keeps_listed(self, antenna_csv_standard):
        ant = AntennaMapping.load(antenna_csv_standard)
        ant2 = ant.with_only([2, 4])
        assert ant2.active_antennas() == [2, 4]

    def test_with_only_then_back(self, antenna_csv_standard):
        ant = AntennaMapping.load(antenna_csv_standard)
        ant2 = ant.with_only([1])
        ant3 = ant2.with_only([1, 2, 4])     # 3 was already functional=0 in source
        # ant3 reflects the new restriction; ant.dataframe still original
        assert ant3.active_antennas() == [1, 2, 4]


class TestWithSnapOutput:
    def test_dual_layout_constructed(self, antenna_csv_standard, tmp_path):
        compute = AntennaMapping.load(antenna_csv_standard)
        # Build a different output mapping with same antenna_ids but different snap/adc
        df_out = pd.DataFrame({
            "antenna_id": [1, 2, 3, 4],
            "snap_id": [2, 2, 3, 3],
            "adc": [5, 6, 5, 6],
            "packet_index": [10, 20, 30, 40],
            "functional": [1, 1, 0, 1],
        })
        path = tmp_path / "out.csv"
        df_out.to_csv(path, index=False)
        output = AntennaMapping.load(path)
        dual = compute.with_snap_output(output)
        assert isinstance(dual, DualLayout)
        # antenna 1: compute(0,0), output(2,5)
        assert dual.compute_snap_adc(1) == (0, 0)
        assert dual.output_snap_adc(1) == (2, 5)
        # antenna 4: compute(1,1), output(3,6)
        assert dual.compute_snap_adc(4) == (1, 1)
        assert dual.output_snap_adc(4) == (3, 6)
