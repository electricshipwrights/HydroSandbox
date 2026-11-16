# Dynamic sinkage force and trim moment: thin ship against Report 71

Measured values are minus the hydrostatic imbalance at the measured attitude. Force / weight x 100 (positive up); moment / (weight x length) x 1000 (positive bow up). Each cell: median measured, median model, rms difference.

## Force, monohull

| closure | Fn 0.20-0.35 | Fn 0.40-0.60 | Fn 0.65-1.00 |
|---|---|---|---|
| auto | -5.06 / -2.84 / 1.81 | -12.73 / -15.13 / 3.13 | -10.35 / -11.98 / 3.84 |
| c6 | -5.06 / -3.94 / 1.02 | -12.73 / -15.57 / 3.68 | -10.35 / -10.93 / 3.76 |
| none | -5.06 / -2.72 / 2.41 | -12.73 / -9.40 / 3.99 | -10.35 / -6.53 / 4.80 |

## Force, catamaran

| closure | Fn 0.20-0.35 | Fn 0.40-0.60 | Fn 0.65-1.00 |
|---|---|---|---|
| auto | -6.10 / -3.22 / 2.50 | -15.02 / -16.96 / 4.88 | -6.05 / -10.05 / 6.61 |
| c6 | -6.10 / -4.67 / 1.43 | -15.02 / -17.32 / 3.56 | -6.05 / -6.48 / 3.23 |
| none | -6.10 / -3.11 / 3.05 | -15.02 / -9.40 / 6.46 | -6.05 / -4.38 / 3.10 |

## Moment, monohull

| closure | Fn 0.20-0.35 | Fn 0.40-0.60 | Fn 0.65-1.00 |
|---|---|---|---|
| auto | +3.15 / -1.40 / 5.92 | +37.42 / +40.98 / 11.18 | +56.78 / +49.10 / 23.10 |
| c6 | +3.12 / -0.16 / 5.24 | +37.32 / +44.15 / 10.05 | +56.46 / +56.06 / 18.68 |
| none | +3.15 / -1.61 / 6.49 | +37.45 / +39.28 / 12.33 | +56.92 / +47.80 / 24.40 |

## Moment, catamaran

| closure | Fn 0.20-0.35 | Fn 0.40-0.60 | Fn 0.65-1.00 |
|---|---|---|---|
| auto | +4.83 / -1.34 / 9.01 | +56.20 / +46.63 / 23.47 | +61.03 / +40.01 / 29.03 |
| c6 | +4.76 / -0.02 / 7.77 | +55.82 / +50.75 / 19.84 | +60.86 / +47.36 / 21.95 |
| none | +4.83 / -1.88 / 9.65 | +56.29 / +43.88 / 25.54 | +61.10 / +38.23 / 31.00 |

## Monohull by model, Fn 0.65-1.00, first closure listed

| model | L/vol^(1/3) | B/T | force: measured / model | moment: measured / model |
|---|---|---|---|---|
| 3b | 6.27 | 2.0 | -10.24 / -14.95 | +102.09 / +41.82 |
| 4a | 7.4 | 1.5 | -9.96 / -10.23 | +58.62 / +39.91 |
| 4b | 7.41 | 2.0 | -10.46 / -12.84 | +66.57 / +48.99 |
| 4c | 7.39 | 2.5 | -10.84 / -15.31 | +79.89 / +55.73 |
| 5a | 8.51 | 1.5 | -10.65 / -9.31 | +50.26 / +39.69 |
| 5b | 8.5 | 2.0 | -10.45 / -11.72 | +49.12 / +51.52 |
| 5c | 8.49 | 2.5 | -10.16 / -13.70 | +62.84 / +56.74 |
| 6a | 9.5 | 1.5 | -11.27 / -8.65 | +43.37 / +38.61 |
| 6b | 9.5 | 2.0 | -10.86 / -10.71 | +43.23 / +48.38 |
| 6c | 9.5 | 2.5 | -9.46 / -12.50 | +46.67 / +57.21 |

## Is the error learnable? Leave-one-model-out correction

A linear correction in smooth features of Fn, L/vol^(1/3), B/T and 1/spacing, fitted without the held-out model. rms error of force / weight x 100 and moment / (weight x length) x 1000, Fn >= 0.3, monohull and catamaran together.

| closure | force: thin ship | force: corrected | moment: thin ship | moment: corrected |
|---|---|---|---|---|
| auto | 5.26 | 2.67 | 24.11 | 14.13 |
| c6 | 3.22 | 1.61 | 19.06 | 13.37 |
| none | 4.52 | 1.93 | 25.88 | 14.25 |
