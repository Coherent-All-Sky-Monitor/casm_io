"""Candidate matching for injection recovery testing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from casm_io.filterbank import FilterbankFile

K_DM = 4.148808e3  # MHz^2 s pc^-1 cm^3


class CandidateMatcher:
    """Match hella candidates against injection truth.

    Uses nearest-to-expected matching: filters by DM window AND time
    window around the expected pulse position, then picks highest SNR.
    Reads frequency band and nsamples from the filterbank file.

    Parameters
    ----------
    fb : FilterbankFile
        The filterbank file the candidates were generated from.
    dm_window : float
        Floor of the DM matching window in pc cm-3.
    dm_window_frac : float
        Fractional DM window. Effective window is
        ``max(dm_window, dm_window_frac * dm_true)``.

        The default 0.06 accounts for:
        - 4.63%% systematic offset from the TIME_RESOLUTION bug
          (dt=1.0e-3 vs actual 1.048576e-3)
        - ~1.4%% margin for DM trial spacing scatter (dm_tol=1.3)
    time_window_fwhm_factor : float
        Time window is ``max(time_window_fwhm_factor * fwhm_samples, 10)``
        samples around the expected pulse position.
    position : float
        Pulse position fraction within the usable window (default 0.5).
    """

    def __init__(
        self,
        fb: FilterbankFile,
        dm_window: float = 15.0,
        dm_window_frac: float = 0.06,
        time_window_fwhm_factor: float = 2.0,
        position: float = 0.5,
    ) -> None:
        self._dm_window = dm_window
        self._dm_window_frac = dm_window_frac
        self._time_window_fwhm_factor = time_window_fwhm_factor
        self._position = position

        # Read band parameters from filterbank
        header = fb.header
        self._tsamp = header.get("tsamp", 1.048576e-3)
        self._nsamples = fb.nsamples
        freq_mhz = fb.freq_mhz
        self._f_hi = float(np.max(freq_mhz))
        self._f_lo = float(np.min(freq_mhz))

    def _sweep_samples(self, dm: float) -> int:
        """Dispersion sweep in samples for a given DM."""
        delay_s = K_DM * dm * (self._f_lo ** -2 - self._f_hi ** -2)
        return int(delay_s / self._tsamp)

    def expected_sample(self, dm: float) -> int:
        """Expected pulse sample index for a given DM.

        Parameters
        ----------
        dm : float
            Dispersion measure in pc cm-3.

        Returns
        -------
        int
            Expected sample index of pulse center at highest frequency.
        """
        sweep = self._sweep_samples(dm)
        return sweep + int((self._nsamples - sweep) * self._position)

    def effective_dm_window(self, dm_true: float) -> float:
        """Compute effective DM matching window in pc cm-3."""
        return max(self._dm_window, self._dm_window_frac * abs(dm_true))

    def effective_time_window(self, fwhm_samples: float) -> float:
        """Compute effective time matching window in samples."""
        return max(self._time_window_fwhm_factor * fwhm_samples, 10.0)

    def match(
        self,
        cand_df: pd.DataFrame,
        dm_true: float,
        fwhm_samples: float,
    ) -> dict:
        """Find best candidate within DM and time windows.

        Parameters
        ----------
        cand_df : pandas.DataFrame
            Candidate table from CandidateReader.
        dm_true : float
            True injection DM in pc cm-3.
        fwhm_samples : float
            Pulse FWHM in samples.

        Returns
        -------
        dict
            Keys: detected (0/1), n_matches (int), best (Series or None).
        """
        if cand_df.empty:
            return {"detected": 0, "n_matches": 0, "best": None}

        dm_win = self.effective_dm_window(dm_true)
        exp_samp = self.expected_sample(dm_true)
        time_win = self.effective_time_window(fwhm_samples)

        dm_mask = (cand_df["dm"] - dm_true).abs() <= dm_win
        time_mask = (cand_df["sample_index"] - exp_samp).abs() <= time_win
        within = cand_df.loc[dm_mask & time_mask]

        n_matches = len(within)
        if n_matches == 0:
            return {"detected": 0, "n_matches": 0, "best": None}

        best_idx = within["snr"].idxmax()
        return {"detected": 1, "n_matches": n_matches, "best": within.loc[best_idx]}
