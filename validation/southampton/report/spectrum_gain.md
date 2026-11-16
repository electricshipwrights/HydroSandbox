# Can a per-hull spectrum gain fix thin ship's wave-pattern resistance?

Southampton series, Fn >= 0.3, thin ship at the measured attitude. Each cell: median predicted / measured, and the typical error factor (exp of the rms log error; x1.00 is perfect). `n` counts measured points.

## Transom closure `auto`

| predictor | trained on | monohull (n=150) | catamaran (n=585) |
|---|---|---|---|
| thin ship | - | 0.66 / x2.20 | 0.84 / x2.19 |
| mono-scaled | the same hull's monohull point | exact by construction | 1.23 / x1.33 |
| flat gain | mono only | (training data) | 1.22 / x1.36 |
| flat gain | leave one model out | 0.86 / x1.35 | 1.03 / x1.28 |
| theta gain | mono only | (training data) | 1.57 / x1.61 |
| theta gain | leave one model out | 0.93 / x1.25 | 1.02 / x1.23 |

Leave-one-model-out theta gain, catamarans, by speed and by spacing:

| | thin ship | mono-scaled | theta gain (held out) |
|---|---|---|---|
| Fn 0.30-0.45 | 3.20 / x3.96 | 1.18 / x1.36 | 1.00 / x1.25 |
| Fn 0.50-0.70 | 0.79 / x1.56 | 1.25 / x1.34 | 1.04 / x1.23 |
| Fn 0.75-1.00 | 0.79 / x1.30 | 1.23 / x1.30 | 1.02 / x1.20 |
| S/L 0.2 | 0.81 / x2.35 | 1.19 / x1.39 | 1.00 / x1.22 |
| S/L 0.3 | 0.84 / x2.17 | 1.23 / x1.33 | 0.99 / x1.24 |
| S/L 0.4 | 0.86 / x2.11 | 1.25 / x1.30 | 1.04 / x1.21 |
| S/L 0.5 | 0.83 / x2.12 | 1.23 / x1.29 | 1.04 / x1.23 |

## Transom closure `c2`

| predictor | trained on | monohull (n=150) | catamaran (n=585) |
|---|---|---|---|
| thin ship | - | 1.83 / x2.30 | 1.82 / x2.39 |
| mono-scaled | the same hull's monohull point | exact by construction | 1.00 / x1.18 |
| flat gain | mono only | (training data) | 0.99 / x1.19 |
| flat gain | leave one model out | 0.98 / x1.16 | 0.99 / x1.18 |
| theta gain | mono only | (training data) | 1.13 / x1.22 |
| theta gain | leave one model out | 0.97 / x1.13 | 1.00 / x1.17 |

Leave-one-model-out theta gain, catamarans, by speed and by spacing:

| | thin ship | mono-scaled | theta gain (held out) |
|---|---|---|---|
| Fn 0.30-0.45 | 3.29 / x4.01 | 1.07 / x1.24 | 0.97 / x1.20 |
| Fn 0.50-0.70 | 1.75 / x1.80 | 0.98 / x1.11 | 1.01 / x1.11 |
| Fn 0.75-1.00 | 1.67 / x1.71 | 1.00 / x1.18 | 0.99 / x1.18 |
| S/L 0.2 | 1.67 / x2.44 | 0.95 / x1.22 | 0.99 / x1.17 |
| S/L 0.3 | 1.82 / x2.36 | 0.99 / x1.21 | 1.00 / x1.18 |
| S/L 0.4 | 1.81 / x2.35 | 1.00 / x1.16 | 1.00 / x1.15 |
| S/L 0.5 | 1.90 / x2.40 | 1.03 / x1.13 | 1.01 / x1.16 |

## Transom closure `c3`

| predictor | trained on | monohull (n=150) | catamaran (n=585) |
|---|---|---|---|
| thin ship | - | 1.73 / x2.05 | 1.73 / x2.14 |
| mono-scaled | the same hull's monohull point | exact by construction | 1.01 / x1.18 |
| flat gain | mono only | (training data) | 1.00 / x1.19 |
| flat gain | leave one model out | 0.98 / x1.16 | 0.99 / x1.18 |
| theta gain | mono only | (training data) | 1.11 / x1.20 |
| theta gain | leave one model out | 0.96 / x1.13 | 1.01 / x1.16 |

Leave-one-model-out theta gain, catamarans, by speed and by spacing:

| | thin ship | mono-scaled | theta gain (held out) |
|---|---|---|---|
| Fn 0.30-0.45 | 2.74 / x3.26 | 1.07 / x1.25 | 0.99 / x1.20 |
| Fn 0.50-0.70 | 1.67 / x1.72 | 1.00 / x1.11 | 1.02 / x1.11 |
| Fn 0.75-1.00 | 1.60 / x1.65 | 0.99 / x1.19 | 1.01 / x1.15 |
| S/L 0.2 | 1.61 / x2.19 | 0.96 / x1.22 | 1.01 / x1.17 |
| S/L 0.3 | 1.74 / x2.12 | 1.00 / x1.21 | 1.01 / x1.16 |
| S/L 0.4 | 1.73 / x2.11 | 1.01 / x1.16 | 1.02 / x1.15 |
| S/L 0.5 | 1.77 / x2.15 | 1.03 / x1.13 | 1.02 / x1.15 |

## Transom closure `c4`

| predictor | trained on | monohull (n=150) | catamaran (n=585) |
|---|---|---|---|
| thin ship | - | 1.59 / x1.82 | 1.63 / x1.91 |
| mono-scaled | the same hull's monohull point | exact by construction | 1.02 / x1.18 |
| flat gain | mono only | (training data) | 1.02 / x1.19 |
| flat gain | leave one model out | 0.97 / x1.16 | 0.99 / x1.17 |
| theta gain | mono only | (training data) | 1.10 / x1.20 |
| theta gain | leave one model out | 0.98 / x1.12 | 1.01 / x1.14 |

Leave-one-model-out theta gain, catamarans, by speed and by spacing:

| | thin ship | mono-scaled | theta gain (held out) |
|---|---|---|---|
| Fn 0.30-0.45 | 2.26 / x2.64 | 1.08 / x1.25 | 0.99 / x1.19 |
| Fn 0.50-0.70 | 1.58 / x1.63 | 1.01 / x1.11 | 1.01 / x1.10 |
| Fn 0.75-1.00 | 1.53 / x1.59 | 0.99 / x1.19 | 1.01 / x1.14 |
| S/L 0.2 | 1.56 / x1.96 | 0.98 / x1.23 | 1.01 / x1.16 |
| S/L 0.3 | 1.64 / x1.89 | 1.01 / x1.21 | 1.00 / x1.15 |
| S/L 0.4 | 1.63 / x1.89 | 1.01 / x1.16 | 1.00 / x1.13 |
| S/L 0.5 | 1.67 / x1.92 | 1.04 / x1.13 | 1.01 / x1.13 |

## Transom closure `c6`

| predictor | trained on | monohull (n=150) | catamaran (n=585) |
|---|---|---|---|
| thin ship | - | 1.40 / x1.49 | 1.44 / x1.56 |
| mono-scaled | the same hull's monohull point | exact by construction | 1.03 / x1.19 |
| flat gain | mono only | (training data) | 1.03 / x1.19 |
| flat gain | leave one model out | 0.97 / x1.17 | 0.99 / x1.17 |
| theta gain | mono only | (training data) | 1.09 / x1.20 |
| theta gain | leave one model out | 0.99 / x1.12 | 1.01 / x1.15 |

Leave-one-model-out theta gain, catamarans, by speed and by spacing:

| | thin ship | mono-scaled | theta gain (held out) |
|---|---|---|---|
| Fn 0.30-0.45 | 1.57 / x1.77 | 1.10 / x1.24 | 1.01 / x1.20 |
| Fn 0.50-0.70 | 1.42 / x1.48 | 1.04 / x1.14 | 1.01 / x1.10 |
| Fn 0.75-1.00 | 1.41 / x1.47 | 0.99 / x1.18 | 1.01 / x1.14 |
| S/L 0.2 | 1.41 / x1.60 | 1.01 / x1.23 | 1.01 / x1.16 |
| S/L 0.3 | 1.43 / x1.53 | 1.03 / x1.20 | 0.99 / x1.16 |
| S/L 0.4 | 1.44 / x1.54 | 1.02 / x1.16 | 0.99 / x1.13 |
| S/L 0.5 | 1.45 / x1.58 | 1.05 / x1.14 | 1.02 / x1.13 |

## Transom closure `couser`

| predictor | trained on | monohull (n=150) | catamaran (n=585) |
|---|---|---|---|
| thin ship | - | 1.46 / x1.57 | 1.49 / x1.65 |
| mono-scaled | the same hull's monohull point | exact by construction | 1.02 / x1.18 |
| flat gain | mono only | (training data) | 1.03 / x1.18 |
| flat gain | leave one model out | 0.98 / x1.16 | 0.99 / x1.16 |
| theta gain | mono only | (training data) | 1.11 / x1.21 |
| theta gain | leave one model out | 0.98 / x1.13 | 1.00 / x1.15 |

Leave-one-model-out theta gain, catamarans, by speed and by spacing:

| | thin ship | mono-scaled | theta gain (held out) |
|---|---|---|---|
| Fn 0.30-0.45 | 1.90 / x2.07 | 1.09 / x1.24 | 1.00 / x1.17 |
| Fn 0.50-0.70 | 1.53 / x1.58 | 1.02 / x1.11 | 1.00 / x1.10 |
| Fn 0.75-1.00 | 1.30 / x1.37 | 0.99 / x1.18 | 1.01 / x1.17 |
| S/L 0.2 | 1.43 / x1.72 | 1.00 / x1.22 | 1.01 / x1.15 |
| S/L 0.3 | 1.52 / x1.62 | 1.03 / x1.20 | 0.99 / x1.17 |
| S/L 0.4 | 1.46 / x1.62 | 1.01 / x1.16 | 1.00 / x1.14 |
| S/L 0.5 | 1.52 / x1.65 | 1.04 / x1.14 | 1.00 / x1.15 |

## Transom closure `none`

| predictor | trained on | monohull (n=150) | catamaran (n=585) |
|---|---|---|---|
| thin ship | - | 0.55 / x1.90 | 0.53 / x2.00 |
| mono-scaled | the same hull's monohull point | exact by construction | 0.99 / x1.25 |
| flat gain | mono only | (training data) | 0.99 / x1.39 |
| flat gain | leave one model out | 1.00 / x1.27 | 0.99 / x1.38 |
| theta gain | mono only | (training data) | 0.92 / x1.32 |
| theta gain | leave one model out | 0.99 / x1.15 | 1.01 / x1.18 |

Leave-one-model-out theta gain, catamarans, by speed and by spacing:

| | thin ship | mono-scaled | theta gain (held out) |
|---|---|---|---|
| Fn 0.30-0.45 | 0.62 / x2.04 | 1.06 / x1.38 | 1.00 / x1.25 |
| Fn 0.50-0.70 | 0.46 / x2.24 | 0.90 / x1.20 | 1.02 / x1.15 |
| Fn 0.75-1.00 | 0.57 / x1.75 | 1.04 / x1.18 | 1.01 / x1.16 |
| S/L 0.2 | 0.51 / x2.12 | 0.95 / x1.32 | 0.94 / x1.22 |
| S/L 0.3 | 0.51 / x2.03 | 0.97 / x1.27 | 1.00 / x1.19 |
| S/L 0.4 | 0.54 / x1.99 | 1.01 / x1.25 | 1.04 / x1.16 |
| S/L 0.5 | 0.55 / x1.88 | 1.01 / x1.17 | 1.04 / x1.16 |

Fitted energy gain G2 (all data), model 5b:

| Fn | theta 0 | 30 | 60 | 90 deg |
|---|---|---|---|---|
| 0.3 | 5.76 | 0.00 | 10.53 | 39.09 |
| 0.4 | 1.90 | 0.41 | 2.17 | 10.06 |
| 0.5 | 1.41 | 3.24 | 1.31 | 3.16 |
| 0.7 | 2.82 | 1.64 | 2.39 | 0.79 |
| 1.0 | 3.02 | 0.76 | 0.78 | 3.40 |

## Transom closure `pr3`

| predictor | trained on | monohull (n=150) | catamaran (n=585) |
|---|---|---|---|
| thin ship | - | 0.76 / x1.42 | 0.92 / x1.36 |
| mono-scaled | the same hull's monohull point | exact by construction | 1.19 / x1.29 |
| flat gain | mono only | (training data) | 1.20 / x1.39 |
| flat gain | leave one model out | 0.87 / x1.32 | 1.03 / x1.33 |
| theta gain | mono only | (training data) | 1.06 / x1.31 |
| theta gain | leave one model out | 1.00 / x1.25 | 0.98 / x1.29 |

Leave-one-model-out theta gain, catamarans, by speed and by spacing:

| | thin ship | mono-scaled | theta gain (held out) |
|---|---|---|---|
| Fn 0.30-0.45 | 0.84 / x1.67 | 1.04 / x1.31 | 0.98 / x1.48 |
| Fn 0.50-0.70 | 0.96 / x1.19 | 1.23 / x1.26 | 1.00 / x1.24 |
| Fn 0.75-1.00 | 0.91 / x1.19 | 1.22 / x1.31 | 0.98 / x1.16 |
| S/L 0.2 | 0.94 / x1.32 | 1.23 / x1.32 | 0.96 / x1.28 |
| S/L 0.3 | 0.98 / x1.38 | 1.26 / x1.35 | 1.00 / x1.36 |
| S/L 0.4 | 0.91 / x1.39 | 1.19 / x1.27 | 0.98 / x1.29 |
| S/L 0.5 | 0.86 / x1.34 | 1.13 / x1.23 | 1.00 / x1.23 |

