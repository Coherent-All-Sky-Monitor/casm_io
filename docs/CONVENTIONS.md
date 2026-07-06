# CASM Sign & Ordering Conventions

The one-page contract shared by every CASM offline package
(`casm_io`, `casm_vis_analysis`, `casm_calibrator`, `bf_weights_generator`,
`casm_imaging`, `casm_offline_frb_injector`). If a function applies a phase,
orders baselines, or indexes an antenna, its behaviour must match this page —
and its docstring should cite it. Historical bugs in this ecosystem (fringed
images, destroyed beams) have all been sign or ordering mistakes.

## Geometry and phase

| Quantity | Convention |
|---|---|
| Antenna positions | ENU metres, `[East, North, Up]`, origin at plank N21 element E1 |
| Source direction `s` | ENU **unit vector toward the source**: `sE = cos(alt)·sin(az)`, `sN = cos(alt)·cos(az)`, `sU = sin(alt)`; azimuth clockwise from North |
| Baseline `b` for pair `(i, j)`, `i < j` | `b = pos_j − pos_i` |
| Geometric delay | `tau = (b · s) / c`  (positive sign; `C_LIGHT_M_S` from `casm_io.constants`) |
| Point-source visibility phase | `V[i, j] = ⟨v_i · conj(v_j)⟩` carries `exp(+1j · 2π · f · tau)` |
| Fringe-stopping | multiply by `exp(sign · 1j · 2π · f · tau)` with **`sign = -1`** to flatten the phase toward the source |

Two ways to steer an image, never to be mixed:

- **Absolute steering** (`exp(sign·1j·2πf·(b·s_pix)/c)`) expects **raw**
  visibilities. Used by `casm_imaging.make_altaz_image[_fast]` and
  `image_around_source`.
- **Differential steering** (`exp(sign·1j·2πf·(b·(s_pix − s_target))/c)`)
  expects **fringe-stopped** visibilities. Used by
  `make_altaz_image_real[_fast]` and `make_phased_sum_image`.

Feeding fringe-stopped data to an absolute-steering imager applies the
geometric phase twice and produces fringes instead of a point source
(fixed July 2026; regression test:
`casm_imaging/tests/test_beamformer_steering.py`).

## Calibration

| Quantity | Convention |
|---|---|
| Gain model | `v_a = g_a · v_true`, so SVD/StEFCal on fringe-stopped `V` recovers `g` (not `conj(g)`) |
| SVD gain | `g = exp(1j · angle(U[:, 0]))`, phase-referenced so `angle(g[ref]) = 0` |
| **Beamformer weight** | **`w = conj(g)`** — the runtime beamformer computes `Σ w·v` with *no* internal conjugation (verified in the casm_bfcorr CUTLASS path, 2026-05-06). Deploying `w = g` destroys the beam; this was confirmed on sky. Regression test: `casm_calibrator/tests/test_svd.py::test_cal_round_trip_phase_aligns` |
| Applying cal to visibilities | `V_cal[i, j] = w_i · conj(w_j) · V[i, j]` = `conj(g_i) · g_j · V[i, j]` |

## Baseline ordering

Visibilities are stored as the flattened upper triangle **including the
diagonal**, ordered like `np.triu_indices(n)`: row 0 cols 0..n−1, then row 1
cols 1..n−1, … Flat index for input pair `(i, j)`, `i ≤ j`:

```
k = i·n − i·(i−1)/2 + (j − i)      # casm_io.correlator.baselines.triu_flat_index
```

`n` is the number of correlator **inputs** (`nsig` from the
`VisibilityFormat`), not the number of live antennas. When a caller needs
`V[j, i]` with `j > i`, conjugate the stored `V[i, j]`.

## Frequency

| Context | Order / value |
|---|---|
| Native SNAP / correlator files | **descending** (band top first) |
| Calibration NPZ/HDF5 (`casm_calibrator`) | ascending (`bf_weights_generator` auto-flips) |
| int8 weight HDF5 (deployed) | **descending** (native SNAP order — do not add `[::-1]`) |
| Band, data before 2026-03-27 | 375 → 468.75 MHz top |
| Band, current (`layout_64ant`) | top 484.375 MHz (shifted 2026-03-27) |
| Channel width | 125/4096 = 0.030517578125 MHz, 3072 channels |

Never hardcode band edges: take the frequency axis from
`casm_io.correlator.load_format(...)` / the file header. The legacy constants
in `casm_io.constants` (`FREQ_TOP_MHZ = 468.75`) apply only to pre-March-2026
data.

## Input indexing and antenna layout

| Quantity | Convention |
|---|---|
| SNAP input (signal-slot) index | `snap_input_idx = snap_id · n_adc + adc` (`n_adc = 12`, 6 SNAPs → 72 slots today) |
| Coherent-beamformer (CB) antenna axis | 64 packet-index slots; `AntennaMapping.packet_index(antenna_id)` is the bridge. A live antenna whose packet index falls outside `[0, 64)` cannot be beamformed — the weight generator raises rather than silently zeroing it |
| Incoherent beamformer (IB) | separate `(n_chan × 132)` signal-slot format |
| `antenna_id` | 1-indexed, stable across layout rebuilds |
| Dual-pol plan (256-antenna build) | pol is the fastest axis within an antenna: `input = antenna_index · n_pol + pol_index`; layout CSVs gain an optional `pol` column defaulting to `'A'` |

**Layout flow (CASMAN is the source of truth for positions/wiring):**

```
CASMAN DB ──casm-sync-wiring──▶ casm_wiring.csv (+ overrides CSV)
          ──casm-build-layout─▶ casm_antenna_layout_YYYY-MM-DD.csv
                                 └── `current` symlink in $CASM_LAYOUT_DIR
AntennaMapping.load(None) resolves: path > $CASM_LAYOUT_CSV > $CASM_LAYOUT_DIR/current
```

Analysis code must load positions through `AntennaMapping` (and
`AntennaMapping.get_positions()`), never from a private CSV copy or a
hardcoded array.

## Site

Authoritative OVRO coordinates (from `casm_io.constants`):

```
lat = 37.2339°   lon = −118.2821°   elevation = 1222.0 m
```

(`bf_weights_generator` historically carried `−118.2820`; `casm_io` is the
value to converge on.)

## Time

Analysis timestamps are Unix seconds (UTC). Convert to local time only for
display, and only via named zones (`zoneinfo` / astropy with
`"America/Los_Angeles"`) — never a fixed UTC offset, which breaks across
DST transitions.
