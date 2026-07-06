# Antenna Mapping

`AntennaMapping` loads a CSV that maps each physical antenna to its SNAP/ADC port and correlator input index (`packet_index`). It is the contract between CAsMan (hardware management) and every CASM analysis repo.

## Loading

```python
from casm_io.correlator import AntennaMapping

# Explicit path
ant = AntennaMapping.load("/path/to/antenna_layout.csv")

# No argument: resolves $CASM_LAYOUT_CSV, then $CASM_LAYOUT_DIR/current symlink
ant = AntennaMapping.load()
```

## Per-antenna lookups

```python
ant.packet_index(antenna_id=5)    # int: correlator input index
ant.snap_adc(antenna_id=5)        # (snap_id, adc) tuple
ant.antenna_for_input(30)         # int: antenna_id for a given packet_index
ant.format_antenna(5)             # 'Ant 5 | S2A6 -> input 30'
ant.is_in_beamforming(5)          # bool: reads include_in_beamforming; falls back to functional
ant.active_antennas()             # sorted list of functional antenna IDs
ant.get_positions()               # (n_ant, 3) float64 ENU in meters; raises if x_m/y_m/z_m absent
ant.get_packet_indices()          # (n_ant,) int, ordered by antenna_id
```

`antenna_id` values are 1-indexed in canonical CSVs. `packet_index` values are 0-indexed.

## Runtime filtering (immutable pattern)

Both methods return a new `AntennaMapping`; the original is unchanged.

```python
ant_clean  = ant.with_inactive([3, 7])      # mark ants 3 and 7 as functional=0
ant_subset = ant.with_only([1, 2, 5, 8])    # all others become functional=0
```

Use `with_inactive` after inspecting an autocorrelation plot to drop bad antennas from all downstream stages without editing the CSV.

## Dense slot table (72 slots)

The full CAsMan hardware layout has 6 SNAPs x 12 ADCs = 72 slots. `slot_table()` returns one row per slot, with `antenna_id = -1` for unwired slots.

```python
tbl = ant.slot_table()               # DataFrame, 72 rows, indexed by snap_input_idx
tbl = ant.slot_table(n_snaps=6, n_adc=12)   # explicit; 72 is the default

# Convenience arrays indexed by snap_input_idx = snap_id * 12 + adc
pos  = ant.positions_64()            # (72, 3) float64 ENU; zeros for unwired slots
mask = ant.active_mask_64()          # (72,) bool: wired & functional & include_in_beamforming
aids = ant.antenna_ids_64()          # (72,) int: antenna_id or -1
```

`active_mask_64` requires both `functional == 1` and `include_in_beamforming == 1` when both columns are present. A CSV without either flag treats every wired slot as active.

The legacy name `Array64Config` from `bf_weights_generator` allocated 64 slots; that was wrong before the 6th SNAP was installed. Do not use 64 as a slot count.

## CSV schemas

`AntennaMapping.load` accepts two schemas and auto-detects which to use.

### Canonical schema (what `casm-build-layout` writes)

Required columns: `antenna_id`, `snap_id`, `adc`, `packet_index`.

Optional columns: `grid_code`, `kernel_index`, `pol`, `x_m`, `y_m`, `z_m`, `functional`, `pos_type`, `include_in_beamforming`.

Legacy column aliases are renamed automatically on load:

| Old name | Canonical name |
|----------|---------------|
| `antenna` | `antenna_id` |
| `snap` | `snap_id` |
| `packet_idx` or `feng_idx` | `packet_index` |
| `feng_id` | `snap_id` |
| `x`, `y`, `z` | `x_m`, `y_m`, `z_m` |

### Legacy `bf_weights_generator` schema

Detected when all four of `pos_id`, `snap_A`, `adc_A`, `ant64` are present as column names. Translated in place:

- `antenna_id = ant64 + 1`
- `snap_id = snap_A`, `adc = adc_A`
- `packet_index = snap_A * 12 + adc_A`
- Position columns: `x_east_m` → `x_m`, `y_north_m` → `y_m`, `z_up_m` → `z_m`

Rows with `pos_type != 'antenna'` are filtered out before translation.

The translator raises `ValueError` on:
- Duplicate `ant64` values (which would silently collapse antennas in the old code).
- A translated CSV where zero rows have `include_in_beamforming == 1`.

### Boolean columns

`include_in_beamforming`, `functional`, and `installed` use a permissive token parser that accepts:

| True tokens | False tokens |
|-------------|-------------|
| `true`, `1`, `yes`, `y`, `t` | `false`, `0`, `no`, `n`, `f`, `` (empty) |

Tokens are case-insensitive. Any unrecognized token raises `ValueError` with the offending values listed. This prevents silent fall-through to False when a CSV contains unexpected strings.

## `DualLayout`

For `bf_weights_generator` workflows where the F-engine receives data on different SNAP wiring than the correlator uses:

```python
dual = ant.with_snap_output(output_mapping)
dual.compute_snap_adc(antenna_id)    # SNAP/ADC the correlator sees
dual.output_snap_adc(antenna_id)     # SNAP/ADC the F-engine receives
```

## Gotchas

**`get_positions()` raises** if `x_m`, `y_m`, or `z_m` are absent from the CSV. Check `"x_m" in ant.dataframe.columns` first if your CSV may lack positions.

**`antenna_id` is 1-indexed** in canonical CSVs. The legacy `ant64` column is 0-indexed; `antenna_id = ant64 + 1` is the fixed convention in the translator. Every v1 fixture relied on this.

**`with_inactive` and `with_only` silently ignore IDs not in the mapping.** This is intentional so you can pass a fixed bad-antenna list without checking membership first.
