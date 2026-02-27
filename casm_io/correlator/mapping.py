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
    x_m, y_m, z_m — ENU coordinates in meters
    functional   — 1=active, 0=inactive
"""

import pandas as pd
import numpy as np


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
    def load(cls, csv_path: str) -> "AntennaMapping":
        """Load antenna mapping from CSV file."""
        df = pd.read_csv(csv_path)
        # Support legacy column names
        rename = {}
        if "antenna" in df.columns and "antenna_id" not in df.columns:
            rename["antenna"] = "antenna_id"
        if "snap" in df.columns and "snap_id" not in df.columns:
            rename["snap"] = "snap_id"
        if "packet_idx" in df.columns and "packet_index" not in df.columns:
            rename["packet_idx"] = "packet_index"
        if rename:
            df = df.rename(columns=rename)
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

    def __repr__(self) -> str:
        return (
            f"AntennaMapping({self.n_antennas} antennas, "
            f"{len(self.active_antennas())} active)"
        )
