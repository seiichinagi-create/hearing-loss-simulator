# Hearing Loss Simulator — Spectrotemporal Resynthesis Tool

A Python desktop application that simulates five clinically-grounded hearing loss profiles by transforming audio through STFT-based spectrotemporal manipulation and resynthesis.

> **Research motivation:** Standard audiometry captures frequency thresholds but cannot represent the *qualitative experience* of degraded hearing. This tool explores whether parameterized spectrotemporal degradation can serve as a patient-experience matching interface — letting patients identify "what my hearing sounds like" beyond pure-tone threshold reporting.

---

## Keywords

`hearing loss simulator` · `audiogram simulation` · `sensorineural hearing loss` · `presbycusis` · `noise-induced hearing loss` · `NIHL` · `4kHz notch` · `auditory processing disorder` · `APD` · `Meniere's disease` · `cookie-bite audiogram` · `mid-frequency hearing loss` · `STFT` · `spectrotemporal masking` · `psychoacoustics` · `audiology education` · `hearing rehabilitation` · `functional hearing loss` · `temporal resolution` · `frequency selectivity`

---

## Five Simulation Profiles

| # | Profile | Clinical Basis | Key Mechanism |
|---|---------|----------------|---------------|
| 1 | **Presbycusis** (Age-related) | ISO 7029 | High-frequency roll-off >1 kHz; 50 dB attenuation at 4 kHz |
| 2 | **NIHL — 4 kHz Notch** | Noise-induced | Gaussian notch centered at 4 kHz (σ = 3.5 semitones, depth −55 dB) + broad HF loss |
| 3 | **Auditory Processing Disorder (APD)** | ICD-10 H93.25 | Normal thresholds; random 20–30 ms temporal gaps + Gaussian frequency smearing (σ = 1.2 semitones) |
| 4 | **Meniere's Disease** | Endolymphatic hydrops | Time-varying low-frequency attenuation (0.4 Hz fluctuation, −40 dB below 125 Hz) |
| 5 | **Cookie-bite (Mid-frequency)** | Congenital/hereditary | Gaussian loss centered at 1 kHz (σ = 8 semitones, peak −50 dB); normal at extremes |

### Spatiotemporal Correlation Erasure (optional overlay)

Beyond fixed audiogram profiles, an experimental **correlation masking** effect is available:

- The amplitude matrix is divided into 1-semitone × 100 ms cells
- For each cell, amplitude correlation with its 4 directional neighbors is computed
- Cells with ≥ 3 correlated neighbors trigger random erasure of an adjacent pitch cell
- Adjustable sensitivity and iteration count
- Models non-linear, context-dependent suppression observed in cochlear damage and auditory neuropathy spectrum disorder (ANSD)

---

## How It Works

```
Audio file (WAV / MP3 / FLAC / OGG)
    │
    ▼
STFT  (hop = 10 ms, window = 40 ms Hann, n_fft = next power of 2)
    │
    ▼
84-key × time-frame amplitude matrix  [A1 (55 Hz) – G#8 (6272 Hz)]
  · 3-bin weighted average per semitone (precision improvement)
    │
    ▼
Hearing loss profile applied  (frequency × time attenuation)
    │
    ▼
Resynthesis  (sine / triangle / sawtooth / square wave per key)
    │
    ▼
PCM-16 WAV output  +  spectrogram visualization
```

Stereo input is processed per-channel. GPU acceleration via CuPy uses a fully vectorized (84, N) matrix computation on VRAM.

---

## Installation

```bash
pip install numpy scipy soundfile librosa matplotlib
```

Optional GPU acceleration (CUDA 12.x):

```bash
pip install cupy-cuda12x
```

## Usage

```bash
python app.py
```

1. Load an audio file — ideally a **single-instrument** or **speech** recording
2. Click **分析（STFT）** to perform spectral analysis
3. Select a hearing loss simulation profile from the dropdown
4. Click **▶ 再合成して再生** — audio is resynthesized with the selected profile and played back
5. Switch profiles or waveform types and press ▶ again to compare instantly (no re-analysis needed)
6. Optionally enable the **Correlation Erasure** effect for additional temporal masking simulation
7. Use **保存** to export the processed WAV; **CSV** to export the raw amplitude matrix

---

## Supported Platforms

- Windows (tested on Windows 11)
- `winsound` is used for playback (Windows-native, no additional audio dependencies)
- macOS/Linux: replace `winsound` block with `sounddevice` or `pygame`

---

## File Structure

```
hearing-loss-simulator/
├── app.py          Main application (GUI + STFT + simulation + synthesis)
├── README.md       This file
└── Freaks.wav      Sample audio for testing
```

---

## ⚠️ Disclaimer

This tool is intended for **educational, research, and awareness purposes only**.  
It does **not** constitute a medical device or clinical diagnostic instrument.  
Hearing assessment must be performed by qualified audiologists using calibrated equipment following IEC 60645 standards.

Simulation parameters are based on published audiological data but are approximations.  
Individual hearing loss patterns vary significantly.

---

## Potential Research Applications

- **Audiology education** — demonstrate what different conditions sound like to normal-hearing students and clinicians
- **Patient counseling** — help patients articulate their hearing experience beyond threshold levels
- **APD and functional hearing loss research** — qualitative patient-experience matching interface
- **Hearing aid fitting simulation** — before/after comparison for counseling
- **Accessibility awareness** — product design and UX testing with hearing loss simulation

---

## References

- ISO 7029:2017 — *Acoustics: Statistical distribution of hearing thresholds related to age and gender*
- Bhatt, S. et al. (2017). Prevalence of and risk factors for hearing loss in US adults. *JAMA Otolaryngology*
- Bellis, T. J. (2003). *Assessment and Management of Central Auditory Processing Disorders*
- Sajjadi, H., & Paparella, M. M. (2008). Meniere's disease. *The Lancet*, 372(9636), 406–414.

---

## License

MIT

## Author

[seiichinagi-create](https://github.com/seiichinagi-create)
