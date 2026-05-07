"""
Antenna mapping: loads a CSV that maps each antenna to its SNAP/ADC/correlator input.

The CSV is the contract between CAsMan (hardware management) and casm_io (data I/O).
As long as the CSV has the required columns, casm_io works for 16, 64, or 256 antennas.

Required CSV columns:
    antenna_id   — 1-indexed antenna number
    snap_id      — Logical SNAP index in data files (0-10 for DADA, feng_id for correlator)
    adc          — ADC port within that SNAP (0-11)
    packet_index — Correlator input index (row/col in visibility matrix)

Optional CSV columns (used if present):
    grid_code    — CAsMan grid position (e.g. CN021E01)
    kernel_index — CAsMan 0-255 correlator position
    pol          — Polarization (A or B)
    x_m, y_m, z_m — ENU coordinates in meters (aliases: x, y, z)
    functional   — 1=active, 0=inactive
    pos_type     — 'antenna' for real antennas, e.g. 'unconnected' for padding rows
    include_in_beamforming — 1=include in beam weights, 0=skip
"""

import pandas as pd
import numpy as np

from casm_io.constants import resolve_layout_path


class AntennaMapping:
    """
    Antenna hardware mapping loaded from CSV.

    Examples
    --------
    >>> ant = AntennaMapping.load("antenna_layout_current.csv")
    >>> ant.packet_index(antenna_id=5)
    30
    >>> ant.snap_adc(antenna_id=5)
    (2, 6)
    >>> ant.antenna_for_input(30)
    5
    >>> ant.format_antenna(5)
    'Ant 5 | S2A6 → input 30'

    Default-path loading (canonical CAsMan-derived layout):

    >>> ant = AntennaMapping.load()      # reads $CASM_LAYOUT_CSV / current symlink
    """

    REQUIRED_COLUMNS = {"antenna_id", "snap_id", "adc", "packet_index"}

    def __init__(self, df: pd.DataFrame):
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Antenna CSV missing required columns: {missing}. "
                f"Required: {self.REQUIRED_COLUMNS}"
            )

        self._df = df.copy()
        self._df["antenna_id"] = self._df["antenna_id"].astype(int)
        self._df["snap_id"] = self._df["snap_id"].astype(int)
        self._df["adc"] = self._df["adc"].astype(int)
        self._df["packet_index"] = self._df["packet_index"].astype(int)

        # Build lookup tables
        self._ant_to_row = {
            int(row["antenna_id"]): idx for idx, row in self._df.iterrows()
        }
        self._input_to_ant = {
            int(row["packet_index"]): int(row["antenna_id"])
            for _, row in self._df.iterrows()
        }

    @classmethod
    def load(cls, csv_path=None) -> "AntennaMapping":
        """Load antenna mapping from CSV file.

        If ``csv_path`` is None, resolves via
        ``$CASM_LAYOUT_CSV`` then ``$CASM_LAYOUT_DIR/current`` symlink
        (see :func:`casm_io.constants.resolve_layout_path`).
        """
        path = resolve_layout_path(csv_path)
        df = pd.read_csv(path)
        # Support legacy column names
        rename = {}
        if "antenna" in df.columns and "antenna_id" not in df.columns:
            rename["antenna"] = "antenna_id"
        if "snap" in df.columns and "snap_id" not in df.columns:
            rename["snap"] = "snap_id"
        if "packet_idx" in df.columns and "packet_index" not in df.columns:
            rename["packet_idx"] = "packet_index"
        if "feng_idx" in df.columns and "packet_index" not in df.columns:
            # bf-imaging legacy: feng_idx == correlator input index
            rename["feng_idx"] = "packet_index"
        if "x" in df.columns and "x_m" not in df.columns:
            rename["x"] = "x_m"
        if "y" in df.columns and "y_m" not in df.columns:
            rename["y"] = "y_m"
        if "z" in df.columns and "z_m" not in df.columns:
            rename["z"] = "z_m"
        if rename:
            df = df.rename(columns=rename)
        # bf_weights_generator legacy: snap_id needs to be derived if only feng_id present
        if "snap_id" not in df.columns and "feng_id" in df.columns:
            df = df.rename(columns={"feng_id": "snap_id"})
        return cls(df)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Raw DataFrame."""
        return self._df

    @property
    def n_antennas(self) -> int:
        """Total number of antennas in the mapping."""
        return len(self._df)

    def packet_index(self, antenna_id: int) -> int:
        """Correlator input index for a given antenna."""
        if antenna_id not in self._ant_to_row:
            raise ValueError(
                f"Antenna {antenna_id} not in mapping. "
                f"Available: {sorted(self._ant_to_row.keys())}"
            )
        idx = self._ant_to_row[antenna_id]
        return int(self._df.loc[idx, "packet_index"])

    def snap_adc(self, antenna_id: int) -> tuple[int, int]:
        """(snap_id, adc) for a given antenna."""
        if antenna_id not in self._ant_to_row:
            raise ValueError(f"Antenna {antenna_id} not in mapping.")
        idx = self._ant_to_row[antenna_id]
        row = self._df.loc[idx]
        return int(row["snap_id"]), int(row["adc"])

    def antenna_for_input(self, packet_idx: int) -> int:
        """Antenna ID for a given correlator input index."""
        if packet_idx not in self._input_to_ant:
            raise ValueError(
                f"Input index {packet_idx} not in mapping. "
                f"Available: {sorted(self._input_to_ant.keys())}"
            )
        return self._input_to_ant[packet_idx]

    def active_antennas(self) -> list[int]:
        """List of antenna IDs that are functional (or all if no 'functional' column)."""
        if "functional" in self._df.columns:
            mask = self._df["functional"].astype(int) == 1
            return sorted(self._df.loc[mask, "antenna_id"].astype(int).tolist())
        return sorted(self._df["antenna_id"].astype(int).tolist())

    def get_positions(self) -> np.ndarray:
        """
        ENU positions as (n_ant, 3) array [x_m, y_m, z_m].
        Raises ValueError if coordinate columns are missing.
        """
        for col in ("x_m", "y_m", "z_m"):
            if col not in self._df.columns:
                raise ValueError(
                    f"Column '{col}' not in CSV. "
                    f"Available: {list(self._df.columns)}"
                )
        return self._df[["x_m", "y_m", "z_m"]].values.astype(np.float64)

    def get_packet_indices(self) -> np.ndarray:
        """All packet_index values as array, ordered by antenna_id."""
        return self._df.sort_values("antenna_id")["packet_index"].values.astype(int)

    def format_antenna(self, antenna_id: int) -> str:
        """Human-readable label: 'Ant 5 | S2A6 → input 30'."""
        snap, adc = self.snap_adc(antenna_id)
        pkt = self.packet_index(antenna_id)
        return f"Ant {antenna_id} | S{snap}A{adc} → input {pkt}"

    def is_in_beamforming(self, antenna_id: int) -> bool:
        """Whether this antenna participates in beamforming.

        Reads the optional ``include_in_beamforming`` column. If absent,
        falls back to ``functional == 1`` (the natural definition for
        legacy CSVs).
        """
        if antenna_id not in self._ant_to_row:
            raise ValueError(f"Antenna {antenna_id} not in mapping.")
        idx = self._ant_to_row[antenna_id]
        row = self._df.loc[idx]
        if "include_in_beamforming" in self._df.columns:
            return int(row["include_in_beamforming"]) == 1
        if "functional" in self._df.columns:
            return int(row["functional"]) == 1
        return True

    def slot_table(self, n_snaps: int = 6, n_adc: int = 12) -> pd.DataFrame:
        """Dense (snap_id, adc) -> antenna table covering every slot.

        Returns a DataFrame indexed by ``snap_input_idx = snap_id * n_adc + adc``
        (0..n_snaps*n_adc-1). For wired slots the row carries
        ``antenna_id``, ``snap_id``, ``adc``, ``packet_index``, plus
        positions/functional flags if present in the source CSV. For
        unwired slots the row is filled with ``antenna_id = -1`` and
        zeros / NaN.

        The default 64-slot dense layout (n_snaps=6, n_adc=12 trimmed to
        64 rows) covers what ``Array64Config`` provided in
        ``bf_weights_generator``.
        """
        n_total = n_snaps * n_adc
        rows = []
        for slot in range(n_total):
            snap_id = slot // n_adc
            adc = slot % n_adc
            wired = self._df[
                (self._df["snap_id"] == snap_id) & (self._df["adc"] == adc)
            ]
            if len(wired) == 1:
                r = wired.iloc[0].to_dict()
                r["snap_input_idx"] = slot
                rows.append(r)
            elif len(wired) == 0:
                rows.append({
                    "snap_input_idx": slot,
                    "antenna_id": -1,
                    "snap_id": snap_id,
                    "adc": adc,
                    "packet_index": -1,
                    "functional": 0,
                    "include_in_beamforming": 0,
                })
            else:
                raise ValueError(
                    f"Slot ({snap_id}, {adc}) has {len(wired)} antennas; "
                    f"expected at most one."
                )
        out = pd.DataFrame(rows).set_index("snap_input_idx").sort_index()
        return out

    def with_snap_output(self, output_mapping: "AntennaMapping") -> "DualLayout":
        """Pair this mapping (compute side) with a SNAP-output mapping.

        Used by ``bf_weights_generator`` when the F-engine receives data on
        a different SNAP wiring than the correlator computes against. Both
        mappings must share the same ``antenna_id`` keys.
        """
        return DualLayout(compute=self, output=output_mapping)

    def __repr__(self) -> str:
        return (
            f"AntennaMapping({self.n_antennas} antennas, "
            f"{len(self.active_antennas())} active)"
        )


class DualLayout:
    """Pair of AntennaMappings: one for compute (correlator), one for SNAP output.

    Both mappings share the same ``antenna_id`` keys. Use
    :meth:`AntennaMapping.with_snap_output` to construct one.
    """

    def __init__(self, compute: AntennaMapping, output: AntennaMapping):
        self.compute = compute
        self.output = output

    def compute_snap_adc(self, antenna_id: int) -> tuple[int, int]:
        """SNAP/ADC the correlator sees for this antenna."""
        return self.compute.snap_adc(antenna_id)

    def output_snap_adc(self, antenna_id: int) -> tuple[int, int]:
        """SNAP/ADC the F-engine receives this antenna on."""
        return self.output.snap_adc(antenna_id)

    def __repr__(self) -> str:
        return f"DualLayout(compute={self.compute!r}, output={self.output!r})"
