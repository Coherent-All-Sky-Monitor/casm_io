"""Result dataclasses for all CASM readers.

Support both attribute access (result.vis) and dict-style access
(result['vis']) for backwards compatibility.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


class _ResultBase:
    """Mixin providing dict-style access for backwards compatibility."""

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)


@dataclasses.dataclass
class VisibilityResult(_ResultBase):
    """Result from VisibilityReader.read() or read_npz()."""
    vis: np.ndarray
    freq_mhz: np.ndarray
    time_unix: np.ndarray
    metadata: dict
    ref: int | None = None
    targets: np.ndarray | list | None = None


@dataclasses.dataclass
class SubbandResult(_ResultBase):
    """Result from VoltageReader.read_subband()."""
    voltages: dict
    header: dict
    freq_mhz: np.ndarray


@dataclasses.dataclass
class CorrelateResult(_ResultBase):
    """Result from casm_io.voltage.correlate()."""
    #: (n_bin, n_chan, n_input, n_input) complex64;
    #: vis[t, f, i, j] = <v_i conj(v_j)> over integration bin t.
    vis: np.ndarray
    #: Bin-centre times in seconds from the start of the input array.
    time_s: np.ndarray
    #: Samples averaged per bin (1 = native 32.768 us resolution).
    tint_samples: int
    #: Input axis labels: (snap, adc) pairs when a snap dict was stacked,
    #: None when the caller passed an array (its input order is kept).
    inputs: list[tuple[int, int]] | None = None


@dataclasses.dataclass
class FullBandResult(_ResultBase):
    """Result from VoltageReader.read_full_band()."""
    voltages: np.ndarray | dict
    header: dict
    freq_mhz: np.ndarray
    utc_start: str
    antenna_df: pd.DataFrame | None
    #: Sub-band indices that were zero-filled because no data was found for
    #: them. Empty for a complete read.
    filled_subbands: list[int] = dataclasses.field(default_factory=list)
