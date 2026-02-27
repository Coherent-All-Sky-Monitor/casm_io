"""Reader for FRB search candidate files."""

import pandas as pd


# Hella T1 output columns -> readable names
_T1_COLUMNS = [
    'snr',
    'sample_index',
    'integration_index',
    'mjd',
    'boxcar_width',
    'dm_index',
    'dm',
    'beam_index',
]

_T1_INT_COLUMNS = [
    'sample_index',
    'integration_index',
    'boxcar_width',
    'dm_index',
    'beam_index',
]


def read_t1_candidates(filepath):
    """Read a Hella T1 candidate list into a DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to whitespace-separated T1 output file (no header).

    Returns
    -------
    pd.DataFrame
        Columns: snr, sample_index, integration_index, mjd, boxcar_width,
        dm_index, dm, beam_index.
    """
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        header=None,
        names=_T1_COLUMNS,
    )

    for col in _T1_INT_COLUMNS:
        df[col] = df[col].astype(int)

    print(f"T1 candidates: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  SNR range: {df['snr'].min():.2f} - {df['snr'].max():.2f}")
    print(f"  DM range: {df['dm'].min():.2f} - {df['dm'].max():.2f}")

    return df
