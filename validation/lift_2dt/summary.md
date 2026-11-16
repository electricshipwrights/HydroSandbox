# 2D+t dynamic lift against thin ship's shortfall

Shortfall = measured - thin ship, at the measured attitude. `slope` is the least-squares slope of the shortfall on the 2D+t prediction (through the origin; 1.0 means the term is the missing physics); `r` their correlation; then the rms error of thin ship alone and of thin ship + 2D+t. Force in % of weight, moment in 1e-3 weight x length.

## Fn 0.40-1.00

| family | n | variant | force: slope, r | force rms: thin -> +2D+t | moment: slope, r | moment rms: thin -> +2D+t |
|---|---|---|---|---|---|---|
| Southampton NPL | 637 | trim | 0.37, +0.08 | 3.4 -> 4.0 | 1.77, +0.65 | 20.3 -> 14.8 |
| Southampton NPL | 637 | run | 0.48, +0.41 | 3.4 -> 3.4 | 3.06, +0.88 | 20.3 -> 19.5 |
| Southampton NPL | 637 | trim, no transom fade | 0.34, +0.08 | 3.4 -> 4.2 | 2.07, +0.60 | 20.3 -> 15.7 |
| Southampton NPL | 637 | trim, no fade, no cross-flow drag | 0.48, +0.07 | 3.4 -> 3.5 | 1.98, +0.63 | 20.3 -> 15.3 |
| Series 64 5s | 39 | trim | -0.98, -0.91 | 3.6 -> 6.9 | 2.22, +0.59 | 19.1 -> 12.7 |
| Series 64 5s | 39 | run | 1.26, +0.86 | 3.6 -> 2.0 | -2.74, -0.37 | 19.1 -> 24.4 |
| Series 64 5s | 39 | trim, no transom fade | -0.89, -0.91 | 3.6 -> 7.2 | 2.77, +0.59 | 19.1 -> 13.9 |
| Series 64 5s | 39 | trim, no fade, no cross-flow drag | -1.21, -0.92 | 3.6 -> 6.2 | 2.62, +0.59 | 19.1 -> 13.6 |
| Athena | 7 | trim | 0.25, +0.92 | 5.2 -> 7.0 | -0.11, -0.06 | 14.2 -> 20.9 |
| Athena | 7 | run | -0.28, -0.93 | 5.2 -> 9.6 | 0.42, +0.65 | 14.2 -> 15.2 |
| Athena | 7 | trim, no transom fade | 0.25, +0.91 | 5.2 -> 6.9 | -0.12, -0.08 | 14.2 -> 21.5 |
| Athena | 7 | trim, no fade, no cross-flow drag | 0.31, +0.93 | 5.2 -> 6.1 | -0.11, -0.07 | 14.2 -> 21.9 |

## Fn 0.20-0.35

| family | n | variant | force: slope, r | force rms: thin -> +2D+t | moment: slope, r | moment rms: thin -> +2D+t |
|---|---|---|---|---|---|---|
| Southampton NPL | 200 | trim | -14.63, +0.05 | 1.4 -> 1.4 | 41.62, +0.90 | 7.4 -> 7.2 |
| Southampton NPL | 200 | run | 1.60, -0.39 | 1.4 -> 1.1 | -3.89, -0.53 | 7.4 -> 8.7 |
| Southampton NPL | 200 | trim, no transom fade | -13.13, +0.04 | 1.4 -> 1.4 | 48.03, +0.90 | 7.4 -> 7.2 |
| Southampton NPL | 200 | trim, no fade, no cross-flow drag | -14.12, +0.05 | 1.4 -> 1.4 | 47.81, +0.90 | 7.4 -> 7.2 |
| Series 64 5s | 12 | trim | -3.31, +0.53 | 1.3 -> 1.3 | 42.06, +0.86 | 13.7 -> 13.4 |
| Series 64 5s | 12 | run | 0.85, -0.10 | 1.3 -> 1.1 | -6.69, -0.57 | 13.7 -> 15.5 |
| Series 64 5s | 12 | trim, no transom fade | -2.88, +0.53 | 1.3 -> 1.3 | 52.08, +0.85 | 13.7 -> 13.4 |
| Series 64 5s | 12 | trim, no fade, no cross-flow drag | -3.39, +0.52 | 1.3 -> 1.3 | 51.50, +0.85 | 13.7 -> 13.4 |
| Athena | 4 | trim | -153.02, -0.78 | 7.7 -> 7.8 | 43.85, +0.59 | 8.0 -> 7.9 |
| Athena | 4 | run | 2.98, +0.96 | 7.7 -> 5.1 | -1.27, -0.36 | 8.0 -> 14.3 |
| Athena | 4 | trim, no transom fade | -155.19, -0.78 | 7.7 -> 7.8 | 43.27, +0.59 | 8.0 -> 7.9 |
| Athena | 4 | trim, no fade, no cross-flow drag | -157.46, -0.78 | 7.7 -> 7.8 | 43.21, +0.59 | 8.0 -> 7.9 |

## Southampton monohulls by model, Fn 0.65-1.00, angle of attack = trim

| model | moment shortfall (median) | 2D+t moment (median) | force shortfall | 2D+t force |
|---|---|---|---|---|
| 3b | +50.9 | +19.9 | +2.5 | +10.8 |
| 4a | +10.5 | +8.4 | -1.2 | +3.6 |
| 4b | +10.3 | +11.2 | +0.6 | +4.6 |
| 4c | +16.5 | +14.8 | +3.2 | +6.2 |
| 5a | +3.0 | +6.0 | -2.5 | +2.3 |
| 5b | -9.9 | +6.9 | -0.6 | +2.5 |
| 5c | -0.9 | +9.7 | +2.5 | +3.6 |
| 6a | -2.5 | +4.5 | -3.5 | +1.6 |
| 6b | -11.7 | +5.2 | -0.6 | +1.8 |
| 6c | -14.6 | +6.3 | +3.3 | +2.1 |
