# Bow-wave load scale against thin ship's shortfall

Fn >= 0.4. `slope`: least-squares slope of the shortfall on the load scale (through the origin); `r`: correlation; then the rms shortfall before and after subtracting slope x scale, with the slope fitted on that family and with the NPL family's slope. Force in % of weight, moment in 1e-3 weight x length.

| family | n | moment: slope, r | moment rms: thin -> own slope -> NPL slope | force: slope, r | force rms: thin -> own slope -> NPL slope |
|---|---|---|---|---|---|
| Southampton NPL | 637 | 0.35, +0.03 | 20.3 -> 18.0 -> 18.0 | 0.09, -0.46 | 3.4 -> 3.4 -> 3.4 |
| Series 64 5s | 39 | 0.91, +0.46 | 19.1 -> 10.5 -> 14.4 | -0.86, -0.92 | 3.6 -> 1.2 -> 3.9 |
| Athena | 7 | -0.55, -0.61 | 14.2 -> 12.4 -> 16.7 | 1.03, +0.95 | 5.2 -> 4.4 -> 5.1 |

## Southampton monohulls by model, Fn 0.65-1.00

| model | moment shortfall (median) | load-scale moment x NPL slope | bow-wave height / draught (formula) |
|---|---|---|---|
| 3b | +50.9 | +9.4 | 0.72 |
| 4a | +10.5 | +12.3 | 0.62 |
| 4b | +10.3 | +12.6 | 0.77 |
| 4c | +16.5 | +11.7 | 0.87 |
| 5a | +3.0 | +11.1 | 0.59 |
| 5b | -9.9 | +12.8 | 0.75 |
| 5c | -0.9 | +11.9 | 0.85 |
| 6a | -2.5 | +10.1 | 0.56 |
| 6b | -11.7 | +11.2 | 0.71 |
| 6c | -14.6 | +11.6 | 0.83 |
