"""
4+4 bit complex unpacking for CASM voltage dumps.

Each byte encodes one complex sample:
- Upper 4 bits: signed integer (-8 to +7) → real part
- Lower 4 bits: signed integer (-8 to +7) → imaginary part
- Two's complement for negatives
"""

import numpy as np


def unpack_4bit(data: np.ndarray) -> np.ndarray:
    """
    Unpack 4+4 bit complex data.

    Parameters
    ----------
    data : np.ndarray
        uint8 array of raw bytes.

    Returns
    -------
    np.ndarray
        complex64 array with same shape as input.
    """
    real = (data >> 4).astype(np.int8)
    imag = (data & 0x0F).astype(np.int8)
    real = np.where(real > 7, real - 16, real)
    imag = np.where(imag > 7, imag - 16, imag)
    return (real + 1j * imag).astype(np.complex64)
