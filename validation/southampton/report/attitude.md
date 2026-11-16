# Dynamic sinkage force and trim moment: thin ship against Report 71

Measured values are minus the hydrostatic imbalance at the measured attitude. Force / weight x 100 (positive up); moment / (weight x length) x 1000 (positive bow up). Each cell: median measured, median model, rms difference.

## Force, monohull

| closure | Fn 0.20-0.35 | Fn 0.40-0.60 | Fn 0.65-1.00 |
|---|---|---|---|
| auto | -5.06 / -5.38 / 0.92 | -12.73 / -14.24 / 2.62 | -10.35 / -10.48 / 3.22 |
| c6 | -5.06 / -3.94 / 1.02 | -12.73 / -15.57 / 3.68 | -10.35 / -10.93 / 3.76 |
| none | -5.06 / -2.72 / 2.41 | -12.73 / -9.40 / 3.99 | -10.35 / -6.53 / 4.80 |

## Force, catamaran

| closure | Fn 0.20-0.35 | Fn 0.40-0.60 | Fn 0.65-1.00 |
|---|---|---|---|
| auto | -6.10 / -6.18 / 1.04 | -15.02 / -17.55 / 3.84 | -6.05 / -10.84 / 6.89 |
| c6 | -6.10 / -4.67 / 1.43 | -15.02 / -17.32 / 3.56 | -6.05 / -6.48 / 3.23 |
| none | -6.10 / -3.11 / 3.05 | -15.02 / -9.40 / 6.46 | -6.05 / -4.38 / 3.10 |

## Moment, monohull

| closure | Fn 0.20-0.35 | Fn 0.40-0.60 | Fn 0.65-1.00 |
|---|---|---|---|
| auto | +2.87 / +3.32 / 4.01 | +37.51 / +42.19 / 8.58 | +56.87 / +48.64 / 23.41 |
| c6 | +3.12 / -0.16 / 5.29 | +37.60 / +44.15 / 10.30 | +56.46 / +56.06 / 18.76 |
| none | +3.15 / -1.61 / 6.54 | +37.73 / +39.28 / 12.63 | +56.92 / +47.80 / 24.47 |

## Moment, catamaran

| closure | Fn 0.20-0.35 | Fn 0.40-0.60 | Fn 0.65-1.00 |
|---|---|---|---|
| auto | +4.63 / +4.02 / 4.91 | +56.18 / +47.99 / 19.83 | +61.05 / +39.27 / 29.58 |
| c6 | +4.76 / -0.02 / 7.81 | +56.14 / +50.75 / 20.07 | +60.86 / +47.36 / 22.01 |
| none | +4.83 / -1.88 / 9.69 | +56.62 / +43.88 / 25.80 | +61.10 / +38.23 / 31.05 |

## Monohull by model, Fn 0.65-1.00, first closure listed

| model | L/vol^(1/3) | B/T | force: measured / model | moment: measured / model |
|---|---|---|---|---|
| 3b | 6.27 | 2.0 | -10.24 / -12.93 | +102.34 / +40.87 |
| 4a | 7.4 | 1.5 | -9.96 / -8.86 | +58.72 / +39.40 |
| 4b | 7.41 | 2.0 | -10.46 / -11.08 | +66.66 / +48.36 |
| 4c | 7.39 | 2.5 | -10.84 / -13.15 | +79.99 / +54.97 |
| 5a | 8.51 | 1.5 | -10.65 / -8.11 | +50.31 / +39.32 |
| 5b | 8.5 | 2.0 | -10.45 / -10.24 | +49.16 / +50.72 |
| 5c | 8.49 | 2.5 | -10.16 / -11.91 | +62.89 / +56.21 |
| 6a | 9.5 | 1.5 | -11.27 / -7.59 | +43.40 / +38.28 |
| 6b | 9.5 | 2.0 | -10.86 / -9.44 | +43.25 / +48.06 |
| 6c | 9.5 | 2.5 | -9.46 / -11.04 | +46.69 / +56.85 |

## Is the error learnable? Leave-one-model-out correction

A linear correction in smooth features of Fn, L/vol^(1/3), B/T and 1/spacing, fitted without the held-out model. rms error of force / weight x 100 and moment / (weight x length) x 1000, Fn >= 0.3, monohull and catamaran together.

| closure | force: thin ship | force: corrected | moment: thin ship | moment: corrected |
|---|---|---|---|---|
| auto | 5.05 | 1.79 | 23.24 | 13.41 |
| c6 | 3.22 | 1.61 | 19.18 | 13.42 |
| none | 4.52 | 1.93 | 25.99 | 14.30 |
