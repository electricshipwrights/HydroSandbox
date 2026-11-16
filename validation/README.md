# Validation against towing-tank measurements

How far HydroSandbox's hull model (thin-ship wave resistance, ITTC-57 friction,
transom closure, dynamic sinkage and trim) is from measured data, as a function
of speed and hull form. Each case holds the measurements as CSV with their
source, a script that rebuilds the hull, and a script that runs the model and
writes tables and figures.

| Case | Hull | Fn | Measured | Run |
|---|---|---|---|---|
| `wigley/` | Wigley parabolic hull, L/B 10, B/T 1.6, no transom | 0.10-0.40 | C_w, C_t held fixed (Ju 1983); one measured wave spectrum and catamaran harmonics (Insel 1990) | `run.py`, `insel_spectrum.py`, `insel_fig159.py` (seconds) |
| `athena/` | R/V Athena model 5365, L/B 7.5, B/T 4.2, transom | 0.28-1.00 | C_T, C_WP, sinkage, trim, fixed and free; wave profile along the hull | `run.py` (minutes), `wave_profile.py` |
| `southampton/` | NPL round-bilge series, ten models, L/B 7-15, B/T 1.5-2.5, transom; monohull and catamaran at four spacings | 0.20-1.00 | C_T, C_WP, sinkage, trim; free | `run.py` then `report.py` (hours: 850 free-attitude solves per transom closure); `spectra.py` scores a closure at the measured attitudes in minutes |
| `series64/` | Series 64 model 5s, L/B 12.8, B/T 2.0, C_B 0.54, transom; monohull and two spacings | 0.20-1.00 | sinkage, trim; free | `generalisation.py` (minutes) |

Every script is run from the repository root as `uv run python validation/<case>/<script>.py`.

## Benchmarks and experiments

The benchmark scripts above compare the model as it stands with measurement. The rest are experiments from
November 2026, kept with their results because they are negative and worth not repeating (see `NOTES.md`):

- `southampton/spectrum_gain.py` (after `spectra.py`): a learned per-hull gain on the thin-ship wave spectrum.
  Works within the NPL family, catamarans included.
- `southampton/attitude.py`, `phase_a.py`, `generalisation.py`, `series64/generalisation.py`: a fitted correction
  to the dynamic sinkage force and trim moment. Meets its target inside the NPL family; does not transfer to
  Series 64 or Athena.
- `lift_2dt/`: 2D+t dynamic lift (after boatmath). About half the missing trim moment in two families, and lift
  the measurements do not show.
- `bow_wave/`: a bow-wave load from Noblesse et al.'s analytical relations. No correlation with the shortfall.

`C_WP` is the wave-pattern resistance from wave-cut analysis: the energy in the
far-field wave system. It is the measured counterpart of what the model's wave
spectrum carries, so `C_W / C_WP` tests the spectrum's size, and the catamaran
interference factor `tau = C_WP(cat) / C_WP(mono)` tests whether two hulls'
spectra add as linear theory says.

## Sources

- **Southampton**: Molland, Wellicome & Couser (1994), *Resistance experiments
  on a systematic series of high speed displacement catamaran forms: variation
  of length-displacement ratio and breadth-draught ratio*, Ship Science Report
  71, University of Southampton. Tables 1-20 (`r71_resistance.csv`,
  `r71_attitude.csv`), Table II (model particulars), Table III (form factors).
  The tables were read by OCR from the scanned report and checked: three tables
  against an independent hand transcription (510 numbers, no differences), and
  every minus sign in the trim and sinkage tables against the page images (one
  OCR miss, corrected). Rows flagged `copy_of_monohull` are entries where the
  report repeats the monohull's numbers for a catamaran; they are skipped.
  Geometry: the NPL parent body plan digitised from Robards (2008), UNSW,
  Fig. A.13 (`npl_sections.json`); it reproduces the report's C_B (0.397) and
  LCB (6.4% aft) for all ten models.
- **Athena**: Jenkins (1984), *Resistance characteristics of the high speed
  transom stern ship R/V Athena in the bare hull condition, represented by
  DTNSRDC Model 5365*, DTNSRDC-84/024 (DTIC ADA143794). Tables 1-6. In Table 6
  the bow sinkage at Fn 0.41 is inconsistent with the other three columns
  (a misprint); only the midship sinkage and trim columns are used.
- **Wigley**: Ju (1983), in *Proceedings of the Second DTNSRDC Workshop on Ship
  Wave-Resistance Computations* (DTIC ADA139615), Table 1. Insel (1990), *An
  investigation into the resistance components of high speed displacement
  catamarans*, PhD thesis, University of Southampton: Table 3 (transcribed) and
  Fig. 159 (read off the scan, normalisation inferred; see the CSV header).
- **Series 64**: Wellicome, Molland, Cic & Taunton (1999), *Resistance experiments
  on a high speed displacement catamaran of Series 64 form*, Ship Science Report
  106: Table III, and the body plan of Fig. 1 traced by `trace_bodyplan.py`. The
  figure does not label its stations; they were identified with the parent's
  sectional-area and beam curves in Yeh (1965), Marine Technology 2(3), Fig. 1.
  The traced hull has C_B 0.530 (report 0.537) and L/vol^(1/3) 8.52 (8.5).

## Known limits of the comparisons

- Southampton: the model's wetted surface for the rebuilt hulls is about 5%
  above the report's Table II area, and Series 64's 4.5% above (Athena's is within
  1% of its report), so
  the model's friction coefficient is high by that much on those hulls.
- Athena: the offsets have five waterlines below the design waterline and no
  keel line; the tow point is not given.
- Model-scale friction is ITTC-57 with the hull's `form_factor` (zero unless
  set); turbulence studs, spray and wave breaking are in the measurements and
  not in the model.
