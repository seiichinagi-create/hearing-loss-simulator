# Hearing Loss Simulator — Cochlear Filterbank & Vocoder Resynthesis

A Python tool that simulates degraded hearing — sensorineural hearing loss and
**cochlear-implant listening** — by decomposing audio through a physiologically
grounded auditory filterbank and resynthesizing it through a channel vocoder.

> **Research motivation:** Standard audiometry captures frequency thresholds but
> cannot represent the *qualitative experience* of degraded hearing. This tool
> explores whether parameterized cochlear degradation can serve as a
> patient-experience matching interface — letting patients identify "what my
> hearing sounds like" beyond pure-tone threshold reporting.

> **⚠️ Version note (v2 pivot):** The project began (v1) as an STFT + fixed
> pitch-grid + additive-sine approximation (`app.py`). It is migrating to a
> physiologically correct **gammatone filterbank + vocoder** engine
> (`cochlea_engine.py`). See **Homage & Acknowledgment** below for why, and what
> established research this is now built upon.

---

## Homage & Acknowledgment

When this project was first published, its author was unaware that a mature body
of research already addressed exactly this problem — and did it far better. The
**noise-vocoder simulation of cochlear-implant hearing**, introduced by
Shannon, Zeng, Kamath, Wygonski & Ekelid (1995), showed that intelligible speech
survives with as few as *three* bands of envelope-modulated noise. That single
result is the scientific backbone of every serious "what does implant / impaired
hearing sound like" simulator.

Version 2 of this project (`cochlea_engine.py`) is a deliberate **homage** to
that lineage. It replaces the original STFT-and-sine-grid approximation with the
pipeline the field actually uses:

- an **ERB-scale gammatone filterbank** (ERB after Glasberg & Moore 1990;
  analysis–synthesis form after Hohmann 2002) standing in for basilar-membrane
  frequency decomposition, and
- **envelope extraction + vocoder resynthesis** (Shannon et al. 1995) as the
  channel vocoder that models both cochlear-implant coding and the
  envelope-dominated percept of sensorineural loss.

The irony is instructive: the "degraded, buzzy" sound the first version produced
*by accident* was a crude cousin of the Shannon vocoder all along — it was simply
run at a physiologically meaningless 336 channels instead of the handful a real
implant provides. Committing to the method properly turns that artifact into the
phenomenon.

This section exists so the credit lands where it belongs: with the
auditory-science and cochlear-implant research community, not with this
repository.

---

## v2 Engine — `cochlea_engine.py`

A dependency-light (numpy / scipy only) implementation of the standard auditory
pipeline:

```
Audio  →  complex gammatone filterbank (ERB-spaced)  →  per-channel envelope
       →  [ pathology transforms ]  →  vocoder resynthesis (tone / noise carrier)
```

Correctness is verified **by measurement, not assertion** — run the built-in
self-test:

```bash
python cochlea_engine.py
```

| Check | Measured result |
|-------|-----------------|
| ERB channel spacing uniformity | std/mean = 2.4×10⁻¹⁵ (exactly ERB-equal) |
| Center-frequency accuracy (impulse response vs design) | < 0.15 % error |
| −3 dB bandwidth vs Glasberg–Moore ERB | ratio 0.999, flat across the band |
| Filterbank coverage ripple (150–6000 Hz) | 0.7 dB p-p |
| Resynthesis integrity | no NaN/Inf; fewer channels → implant-style degradation |

Because channels are spaced on the ERB scale (~30 for normal hearing, 4–22 for
implant simulation) and each is resynthesized through its own carrier, the
beating / aliasing / low-frequency doubling artifacts of the v1 sine-grid are
structurally absent.

### Pathology models (layered on the core)

| Condition | Physiology | Implementation |
|-----------|------------|----------------|
| Threshold loss (audiogram) | Hair-cell sensitivity loss | Per-channel gain |
| **Loudness recruitment** | Abnormally rapid loudness growth above elevated threshold — the subjective core of sensorineural loss | Per-channel level-dependent expansion |
| Reduced frequency selectivity | Broadened auditory filters | Multiply gammatone bandwidth (`BW_K`) |
| Dead regions | Non-functioning inner hair cells | Zero the affected channels |
| **Cochlear-implant channel interaction** | Electrode current spread | Envelope leakage across neighboring channels |
| Temporal-fine-structure loss | Only envelope survives (total in implants) | Lower the envelope low-pass cutoff (`env_cutoff`) |

> Integration into the GUI is in progress. `app.py` currently still runs the
> original **v1** STFT engine documented below.

---

## Keywords

`cochlear implant simulation` · `noise vocoder` · `tone vocoder` · `gammatone filterbank` · `ERB` · `basilar membrane model` · `loudness recruitment` · `sensorineural hearing loss` · `dead regions` · `temporal fine structure` · `hearing loss simulator` · `audiogram simulation` · `presbycusis` · `noise-induced hearing loss` · `NIHL` · `auditory processing disorder` · `APD` · `Meniere's disease` · `cookie-bite audiogram` · `psychoacoustics` · `audiology education` · `hearing rehabilitation`

---

## v1 (original STFT approach) — five simulation profiles

The original engine transforms audio through STFT-based spectrotemporal
manipulation and additive resynthesis. It is retained for reference and is what
`app.py` runs today.

| # | Profile | Clinical Basis | Key Mechanism |
|---|---------|----------------|---------------|
| 1 | **Presbycusis** (Age-related) | ISO 7029 | High-frequency roll-off >1 kHz; 50 dB attenuation at 4 kHz |
| 2 | **NIHL — 4 kHz Notch** | Noise-induced | Gaussian notch centered at 4 kHz (σ = 3.5 semitones, depth −55 dB) + broad HF loss |
| 3 | **Auditory Processing Disorder (APD)** | ICD-10 H93.25 | Normal thresholds; random 20–30 ms temporal gaps + Gaussian frequency smearing |
| 4 | **Meniere's Disease** | Endolymphatic hydrops | Time-varying low-frequency attenuation (0.4 Hz fluctuation, −40 dB below 125 Hz) |
| 5 | **Cookie-bite (Mid-frequency)** | Congenital/hereditary | Gaussian loss centered at 1 kHz (σ = 8 semitones, peak −50 dB); normal at extremes |

The v1 pipeline: STFT (hop = 10 ms, 40 ms Hann window) → 336-bin (¼-semitone)
amplitude matrix → hearing-loss profile applied as a frequency × time attenuation
→ additive resynthesis (sine / triangle / sawtooth / square). Stereo is processed
per channel; optional CuPy GPU acceleration.

---

## Installation

```bash
pip install numpy scipy soundfile librosa matplotlib sounddevice
```

Optional GPU acceleration (CUDA 12.x):

```bash
pip install cupy-cuda12x
```

## Usage

```bash
python cochlea_engine.py   # v2 engine self-test (prints verification table)
python app.py              # v1 GUI application
```

---

## ⚠️ Disclaimer

This tool is intended for **educational, research, and awareness purposes only**.
It does **not** constitute a medical device or clinical diagnostic instrument.
Hearing assessment must be performed by qualified audiologists using calibrated
equipment following IEC 60645 standards.

Simulation parameters are based on published audiological data but are
approximations. Individual hearing loss patterns vary significantly.

---

## References

### Auditory modeling & cochlear-implant simulation (v2 foundations)

- Shannon, R. V., Zeng, F.-G., Kamath, V., Wygonski, J., & Ekelid, M. (1995).
  *Speech recognition with primarily temporal cues.* **Science, 270**(5234),
  303–304. https://doi.org/10.1126/science.270.5234.303
- Glasberg, B. R., & Moore, B. C. J. (1990). *Derivation of auditory filter
  shapes from notched-noise data.* **Hearing Research, 47**(1–2), 103–138.
  https://doi.org/10.1016/0378-5955(90)90170-T
- Hohmann, V. (2002). *Frequency analysis and synthesis using a Gammatone
  filterbank.* **Acta Acustica united with Acustica, 88**(3), 433–442.

### Clinical (v1 profiles)

- ISO 7029:2017 — *Acoustics: Statistical distribution of hearing thresholds related to age and gender*
- Bhatt, S. et al. (2017). Prevalence of and risk factors for hearing loss in US adults. *JAMA Otolaryngology*
- Bellis, T. J. (2003). *Assessment and Management of Central Auditory Processing Disorders*
- Sajjadi, H., & Paparella, M. M. (2008). Meniere's disease. *The Lancet, 372*(9636), 406–414.

---

## License

MIT

## Author

[seiichinagi-create](https://github.com/seiichinagi-create)
