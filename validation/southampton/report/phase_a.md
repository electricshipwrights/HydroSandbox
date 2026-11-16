# Phase A: corrected dynamic force and moment, closed loop (closure `c6`)

Free running attitude solved with plain thin ship and with the correction fitted without the model being tested. Each cell: mean error / rms error against Report 71.

## Trim [deg]

| | Fn 0.20-0.35 | Fn 0.40-0.60 | Fn 0.65-1.00 | all speeds |
|---|---|---|---|---|
| monohull, thin ship | -0.10 / 0.15 | -0.07 / 0.31 | -0.19 / 0.54 | -0.13 / 0.41 |
| monohull, corrected | +0.06 / 0.13 | -0.01 / 0.15 | -0.02 / 0.12 | +0.00 / 0.13 |
| catamaran, thin ship | -0.17 / 0.23 | -0.26 / 0.55 | -0.37 / 0.60 | -0.29 / 0.52 |
| catamaran, corrected | -0.00 / 0.13 | -0.02 / 0.25 | +0.03 / 0.24 | +0.01 / 0.22 |

## Sinkage [% of draught]

| | Fn 0.20-0.35 | Fn 0.40-0.60 | Fn 0.65-1.00 | all speeds |
|---|---|---|---|---|
| monohull, thin ship | -0.41 / 0.52 | +1.83 / 2.23 | +0.58 / 2.27 | +0.72 / 1.98 |
| monohull, corrected | +0.01 / 0.37 | +0.08 / 0.37 | -0.02 / 0.59 | +0.02 / 0.48 |
| catamaran, thin ship | -0.49 / 0.71 | +2.10 / 2.62 | +1.15 / 2.23 | +1.04 / 2.11 |
| catamaran, corrected | -0.06 / 0.60 | +0.20 / 0.91 | -0.06 / 1.02 | +0.02 / 0.90 |

## Total resistance [% error]

| | Fn 0.20-0.35 | Fn 0.40-0.60 | Fn 0.65-1.00 | all speeds |
|---|---|---|---|---|
| monohull, thin ship | -20.11 / 20.78 | +1.39 / 18.03 | +14.06 / 17.63 | +2.22 / 18.54 |
| monohull, corrected | -18.94 / 19.69 | -1.32 / 15.90 | +14.47 / 18.87 | +1.97 / 18.25 |
| catamaran, thin ship | -27.10 / 27.41 | -3.41 / 17.70 | -1.42 / 16.44 | -8.21 / 19.98 |
| catamaran, corrected | -26.15 / 26.48 | -4.48 / 17.38 | +0.53 / 17.29 | -7.38 / 19.91 |

## By model, all configurations and speeds: rms error, thin ship -> corrected

| model | trim [deg] | sinkage [% draught] |
|---|---|---|
| 3b | 1.40 -> 0.38 | 3.17 -> 1.10 |
| 4a | 0.30 -> 0.34 | 0.99 -> 1.00 |
| 4b | 0.49 -> 0.17 | 2.26 -> 0.89 |
| 4c | 0.40 -> 0.20 | 3.41 -> 0.93 |
| 5a | 0.22 -> 0.13 | 1.01 -> 0.46 |
| 5b | 0.26 -> 0.19 | 1.44 -> 0.63 |
| 5c | 0.19 -> 0.09 | 2.19 -> 0.45 |
| 6a | 0.15 -> 0.12 | 1.20 -> 1.10 |
| 6b | 0.15 -> 0.14 | 1.61 -> 0.70 |
| 6c | 0.15 -> 0.10 | 2.09 -> 0.80 |

Gate (plan): held-out trim within 0.3 deg and sinkage within 1% of draught. Corrected, all points: trim rms 0.20 deg, sinkage rms 0.84% of draught.

Solves that found no equilibrium: 3: [('6a', 'mono', False, 0.95), ('4a', 's0.2', False, 0.9), ('4a', 's0.4', False, 0.85)].
