"""
Physical and observatory constants for CASM.
"""

# OVRO Observatory
OVRO_LAT_DEG = 37.2339
OVRO_LON_DEG = -118.2821
OVRO_ELEV_M = 1222.0

# Frequency band (legacy values for pre-March 2026 data)
# For current frequencies, use VisibilityFormat from file headers.
# The band was shifted on 2026-03-27: FREQ_TOP changed from 468.75 to 484.375 MHz.
FREQ_TOP_MHZ = 468.75
FREQ_BOTTOM_MHZ = 375.0
N_CHAN_TOTAL = 3072
CHAN_BW_MHZ = 125.0 / 4096  # 0.030517578125 MHz

# Timing
TSAMP_US = 32.768  # Voltage dump sample time in microseconds

# Speed of light
C_LIGHT_M_S = 299792458.0
