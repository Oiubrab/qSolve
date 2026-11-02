# Binary neutron star track: plan

The BNS track the owner put in scope for [../completed/gw_detection_plan.md](../completed/gw_detection_plan.md)
(G-D1, G-D2): component masses 1–3 M☉, 256 s windows at 2048 Hz, a separate model. The BBH work's tools (noise
from bulk data, injections, time slides, search and vetting) carry over, but almost every size changes by a
factor of 256, so the design below has to be settled before building.

Read the "q gotchas" in [../completed/improvemet_plan.md](../completed/improvemet_plan.md) and the BBH plan's log
first.

**How to use this file:** as the other plans. Tick an item only when its check passes and `q tests/run.q`
passes; add a dated line to the Progress log; commit each finished stage (the owner's rule).

**Current status (2026-10-29):** design only; open questions B-Q1 to B-Q3 below need the owner.

---

## 1. What changes from BBH

Measured on this machine (2026-10-29, shared with other jobs):

| | BBH (done) | BNS |
|---|---|---|
| Signal duration from 20 Hz | under 1 s (10–80 M☉) | 44 s (3+3 M☉), 158 s (1.4+1.4), 277 s (1+1) |
| Window, 2 detectors at 2048 Hz | 2 × 2,048 samples | 2 × 524,288 samples |
| Where the SNR is | mostly the last few cycles | spread over minutes, a little per second |
| One real FFT of a window | 3 ms | 3 s (2^19 points) |
| Merger frequency | 50–400 Hz | above 1 kHz, beyond the 512 Hz band (the inspiral carries most of the SNR anyway) |

Three consequences:

- **Injections by FFT are too slow to generate per epoch** (G-D7): 6,000 windows would take hours of FFTs. A BNS
  inspiral, though, chirps slowly enough that whitening it is, to a good approximation, dividing its amplitude by
  the noise ASD at its instantaneous frequency (stationary phase). So the whitened signal can be made directly in
  the time domain, vectorised, with no FFT: TaylorT2/T4 phase and amplitude at 2048 Hz, divided by the chunk's ASD at
  f(t), projected onto H1 and L1 as now.
- **The network can't take a million inputs.** It needs a compressed view (B-Q1).
- **Noise windows must be long and clean**: 256 s of contiguous analysable data in both detectors. The bulk data
  have that (O4a's day had segments of 4.6, 5.8 and 9.7 h); the event files' 4096 s mostly do too.

## 2. Open questions for the owner

1. **B-Q1, the input representation.** Recommended **(a)**:
   - (a) *Multi-rate time series*: sample each part of the window at the rate the chirp needs then (the first
     ~200 s at 128 Hz, the next ~50 s at 512 Hz, the last few seconds at 2048 Hz; about 40,000 samples per
     detector), into Conv1D layers with large pools. Keeps phase information; used by Schäfer et al. (2020).
   - (b) *Spectrogram channels*: a whitened STFT (0.5 s segments, 64 log-spaced bands, 20–512 Hz), 512 time steps ×
     64 bands × 2 detectors, with the bands as Conv1D channels. Simpler, but loses phase, and per-pixel SNR is tiny.
   - (c) *The GRU* (G5.3) over blocks of the window. Probably too slow at 512+ steps of thousands of inputs.
2. **B-Q2, data generation.** Recommended: the time-domain whitened signals of section 1, generated per epoch.
   The alternative is a fixed set of a few thousand stored examples (simpler, overfits sooner).
3. **B-Q3, expectations.** A machine-learning BNS detector in pure q will be far less sensitive than matched
   filtering, which is near-optimal for long inspirals. The realistic goal is a working pipeline and an honest
   efficiency curve; GW170817 (SNR 32, O2) is the one real BNS loud enough to test on. (GW190425 was seen by L1 and
   Virgo only, H1 being off, so it falls outside an H1–L1 detector.)

## 3. Stages (assuming the recommended answers)

### B1 — Signals and noise

- [ ] **B1.1 Time-domain whitened inspiral.** `.qs.gw.inspiralTD`: TaylorT4 phase (3.5PN, aligned spins
  optional) and leading-order amplitude at 2048 Hz from f_low to 512 Hz, whitened by the ASD at f(t), projected and
  delayed per detector. *Accept:* against the frequency-domain path (TaylorF2, whitened by FFT) on a 3+3 M☉
  signal, overlap above 0.97 and SNR within 3%.
- [ ] **B1.2 Long noise.** `.qs.gw.bulkNoise` with 300 s chunks (256 s kept) from O3 and O4a bulk days that
  aren't searched, split by day into training, validation and test.
- [ ] **B1.3 Multi-rate view.** `.qs.gw.multirate[w;bands]`: anti-aliased decimation of each stretch of the
  window. *Accept:* a chirp's SNR is kept within 2%; a test on synthetic chirps.

### B2 — Model and training

- [ ] **B2.1 Batches.** As `.qs.gw.batch`, with BNS parameters (1–3 M☉, spins up to 0.05, SNR 8–40), the merger
  in the last 10% of the window, glitches and single-detector chirps as negatives.
- [ ] **B2.2 Conv1D detector** on the multi-rate input; early stopping; models saved as `projects/gw/models/gw_bns_*.qsm`.

### B3 — Evaluation and real events

- [ ] **B3.1** Background by time slides on test noise; thresholds for 1 false alarm per hour and per day
  (`.qs.gw.libSlides` with 256 s windows).
- [ ] **B3.2** Efficiency by SNR and by chirp mass; compared with a matched filter on the same injections (the
  BNS case is where matched filtering shines).
- [ ] **B3.3** GW170817: scan the 4096 s around it with the BNS model; record the score and
  false-alarm rate. GW170817 has a famous loud glitch in L1 about a second before the merger, a good test.

## Progress log

- 2026-10-29: Plan written from the BBH work's measurements (FFT speed, chirp durations). Not started.
