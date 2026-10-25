# Gravitational-wave chirp detection plan

A real-world test for qSolve, based on the owner's honours thesis
([Oiubrab/recurrent_grav_detector](https://github.com/Oiubrab/recurrent_grav_detector)): train a network to
spot binary black hole "chirps" in detector noise, using public LIGO/Virgo data, then see whether it finds
real detections it never saw.

The earlier plan (qSolve's rewrite, dashboard and FX test, Phases 0–9) is in
[completed/improvemet_plan.md](../completed/improvemet_plan.md). Read its "q gotchas" section before writing
q here: several of them (no closures, `x`/`y`/`z` making test lambdas take arguments, `sum` skipping nulls,
one-letter string lists) will come up.

**How to use this file**

- Work through the stages in order; items inside a stage are mostly independent.
- Tick an item (`- [x]`) only when its acceptance check passes *and* `q tests/run.q` passes.
- Add a dated line to the [Progress log](#progress-log) when you tick something, saying what changed and
  anything surprising. The log is append-only.
- If a decision below changes, edit it and say why in the log.
- **Commit each finished stage to `main` before starting the next** (the owner's standing rule). Don't push
  without asking.
- Keep tests offline: anything needing downloaded data skips itself when the data is missing (`.t.skip`).

**Current status (2026-10-25):** stages G1, G2 and G3 are complete, including the G3.4 gate: our pure-q
PhenomB template finds GW150914 in the real H1 and L1 data at the published SNRs, with L1 leading H1 by 7 ms.
G4.1 (noise library, needs the ~2 GB of 4096 s files) is next. The owner chose
pure-q waveforms and the plan's default download budget, and asked to include binary neutron stars (a separate
1–3 M☉ track, 256 s windows at 2048 Hz, its own model) and an LSTM/GRU. Reference sources for waveform
coefficients: LALSuite's `lalsimulation/lib` on git.ligo.org (raw files fetch without a login).

---

## 1. Background

### The thesis setup

- 1-second windows at 4096 Hz, with frequencies from 30 Hz.
- Noise: Gaussian, coloured by the Advanced LIGO design sensitivity (`ZERO_DET_high_P` noise curve).
- Signals: IMRPhenomPv2 waveforms from LALSuite, equal masses of 20 M☉, optimally oriented, at 200–750 Mpc,
  then whitened.
- Confusers: Gaussian "glitches".
- Network: Keras, dense layers (32 sigmoid, 64 tanh, 16, 1), and later an LSTM, which gave no improvement.
- It reached 100% on its simulated test set: Gaussian noise, one mass, and fairly loud signals.

### What changes here

| Thesis | This plan |
|---|---|
| Simulated Gaussian noise from a design curve | **Real detector noise** from LIGO's public data, with its non-Gaussian features and glitches |
| One mass pair, optimal orientation | A spread of masses, spins, sky positions, orientations and signal strengths |
| One detector | Hanford (H1) and Livingston (L1) together, with the right relative timing and response |
| Signal strength set by distance | Set by signal-to-noise ratio (SNR), so detection can be measured against loudness |
| Tested on more simulations | Tested on simulations **and on real catalogued detections** the model never trained on |
| Python, LALSuite, Keras | Pure q (qSolve), with LALSuite at most an optional cross-check |

### The data (checked 2026-10-24)

The Gravitational Wave Open Science Center (GWOSC, gwosc.org) publishes strain data under CC BY 4.0. We must
acknowledge GWOSC and LIGO/Virgo/KAGRA in the README.

- **Event list:** `https://gwosc.org/eventapi/json/allevents/` returns 671 entries across the catalogues
  (GWTC-1 to GWTC-5, plus others); 417 have p_astro > 0.9 and a network SNR (some events appear in more than
  one catalogue). Each event's JSON gives GPS time, masses, distance, network matched-filter SNR, and strain
  file links.
- **Strain files per event and detector:**
  - durations of 32 s or 4096 s, at 4096 Hz or 16384 Hz;
  - formats `gwf`, `hdf5` or **`txt.gz`**: three header lines (`# ... starting GPS <t> duration <d>`), then
    one strain value per line. q reads text directly, so no HDF5 library is needed.
  - Sizes: a 32 s, 4 kHz file is about 1.3 MB gzipped; a 4096 s, 4 kHz file is about 164 MB gzipped
    (16.8 million samples).
  - Example: GW150914 H1, `https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/H-H1_GWOSC_4KHZ_R1-1126259447-32.txt.gz`.
- **Real noise** comes from the 4096 s files: over an hour around each event, with the event's own few
  seconds cut out. About 6 events × 2 detectors is roughly 2 GB to download and gives about 13 hours of real
  noise from different observing runs.
- **Test detections** come from the 32 s files around every confident binary black hole event in range (see
  scope): around 200 events × 2 detectors × 1.3 MB, so about 0.5 GB.

### What is realistic

- On injections into real noise, networks like this typically find nearly all signals with SNR above about
  10 and fewer below about 8. Matched filtering is the optimal method for known waveforms in Gaussian noise,
  so the network should approach it, not beat it.
- On real events: the catalogue events were all found by matched filtering, many near its threshold. Expect
  the loud ones to be found (GW150914 SNR 24, GW250114 SNR 79, GW230814, GW190521, ...) and many quiet ones
  missed. The result to report is recall against catalogue SNR at a stated false-alarm rate.
- False alarms are the hard part in real noise: glitches look like short chirps. The false-alarm rate must be
  measured from background (time slides, G6), not assumed.

---

## 2. Goals

1. **A reusable pipeline in q:** download, parse, whiten, simulate, inject, window, train, scan.
2. **A detector that works on simulations:** efficiency against SNR at a measured false-alarm rate, on held-out
   real noise.
3. **An honest test on real detections:** which catalogued events the model finds at a stated false-alarm
   rate, against their catalogue SNR.
4. **Extensions to qSolve that are useful beyond this project:** FFT and signal processing, training on data
   generated fresh each epoch, and 1-D convolution layers.

Non-goals: competing with LIGO's search pipelines, parameter estimation, binary neutron stars or other
long-duration signals (see scope).

---

## 3. Design (proposed; settle the open questions in section 4 first)

| # | Decision | Why |
|---|---|---|
| G-D1 | Scope: BBH component masses 10–80 M☉, aligned spins, in 1-second windows; BNS is a separate track for 1–3 M☉ components, 256-second windows, and its own model. | BBH chirps fit the thesis-style 1 s window. A 256 s BNS window covers inspiral from 20 Hz while keeping long inputs out of the BBH model. |
| G-D2 | BBH: sample rate 2048 Hz after downsampling, band 20–512 Hz, two detectors (H1, L1), so 2 × 2048 inputs per 1 s window. BNS: same rate and band in 256 s windows, handled separately. | Black hole mergers in this mass range peak well below 512 Hz. Two detectors give coincidence. 1024 Hz is the BBH fallback if memory or speed bites; long BNS sequences must not be silently truncated to it. |
| G-D3 | Whitening uses a PSD estimated by Welch's method (4 s Hann segments, 50% overlap) from nearby data (e.g. ±256 s, excluding the window being analysed). | It's the standard approach. It turns real coloured noise into roughly unit-variance white noise, so the network's inputs have a consistent scale. |
| G-D4 | Injections happen in the frequency domain: the waveform h(f), projected onto each detector, divided by that detector's ASD, scaled to a target network SNR, inverse-FFT'd and added to *whitened* real noise. | Whitening is linear, so this equals injecting into raw strain and then whitening, and it is much cheaper. |
| G-D5 | Waveforms are generated in pure q. **Recommended:** the TaylorF2 post-Newtonian inspiral (closed-form phase to 3.5PN), then the IMRPhenomB inspiral–merger–ringdown model (Ajith et al. 2011, closed-form frequency-domain amplitude and phase for aligned spins), checked against TaylorF2 in the early inspiral and end-to-end on real data (G3.4). LALSuite (`pip install lalsuite`) is an optional cross-check, never a dependency. | It keeps qSolve pure q, and both models are closed-form. Fidelity is lower than modern models like IMRPhenomXPHM, but it is enough to train a detector; the real-event test will show whether that matters. |
| G-D6 | Detector response: H1 and L1 positions and arm orientations, antenna patterns F+ and F× from sky position, polarisation and Greenwich sidereal time, and the inter-detector time delay (up to about 10 ms). | Without it, the two detectors' signals would be unrealistically identical, and the network would learn the wrong coincidence pattern. |
| G-D7 | Training data are generated fresh each epoch from a library of whitened noise plus random injections and glitches. This needs a qSolve core feature: `.qs.train` accepting a data *generator*. | It avoids storing gigabytes of examples and greatly reduces overfitting to particular noise. |
| G-D8 | Models in order: an MLP baseline (the thesis's dense design), 1-D convolution layers added to the qSolve core, then an LSTM or GRU in homage to the thesis. | CNNs on whitened strain are the standard published approach (George & Huerta 2018; Gabbard et al. 2018). The recurrent model revisits the thesis result. |
| G-D9 | Splits by time, never by window: training noise, validation noise and test noise come from different events' 4096 s files, and every test detection's surrounding data is excluded from training noise. | Overlapping windows from the same minute of noise would leak between sets. |
| G-D10 | The false-alarm rate is measured by time slides: shift L1 against H1 by many multiples of more than 10 ms (in practice whole seconds) across held-out noise, score every window, and convert score thresholds to false alarms per hour or month. | This is the standard way to get much more background than real time provides, without real signals in it. |
| G-D11 | New code goes in `signal.q` (`.qs.sig`: FFT and signal processing, general purpose) and `gw.q` (`.qs.gw`: GWOSC access, detectors, waveforms, injections, scanning), plus `bench/gw.q`. Downloads go in `data/gw/` (git-ignored), with parsed strain cached as q binary files so the text is parsed once. | It follows the existing layout (`fx.q`, `bench/fx.q`). |

---

## 4. Open questions for the owner

1. **Waveforms:** settled: pure q, with LALSuite only as an optional cross-check.
2. **Download budget:** settled: about 2 GB of noise and about 0.5 GB of test events.
3. **Model ambition:** settled: include an LSTM or GRU after the MLP and CNN.
4. **Scope:** settled: include BNS as a separate track with 1–3 M☉ component masses, 256 s windows at 2048 Hz, and a separate model; keep BBH at 10–80 M☉ and 1 s.

---

## 5. Stages

### G1 — Data access

- [x] **G1.1 Event catalogue.** `.qs.gw.events[]`: fetch `allevents` JSON (curl, cached in `data/gw/`), parse
  with `.j.k` into a table (name, catalogue, GPS, source masses, distance, network SNR, p_astro), de-duplicate
  across catalogues (keep the latest version), and add selection helpers (confident BBH in mass range with H1
  and L1 data). *Accept:* tests on a saved small JSON sample; real fetch skips without network.
- [x] **G1.2 Strain download.** `scripts/fetch_gw.sh` (or `fetch_data.sh gw`): for a given event list, fetch
  the 32 s or 4096 s 4 kHz `txt.gz` files for H1 and L1, resuming and skipping what already exists. A default
  list of 6 noise events, spread across O1, O2 and O3, plus the test events.
- [x] **G1.3 Parse and cache.** `.qs.gw.strain file` -> `` `gps`rate`h `` (start time, sample rate, float
  vector), handling `nan` (data-quality gaps), gunzipping through `system "env gzip -dc ..."` or a
  decompressed copy, and caching as q binary (`set`), so the second load is instant. *Accept:* a small
  hand-written file round-trips; GW150914's 32 s file gives 131072 samples at 4096 Hz starting at GPS
  1126259447.

### G2 — Signal processing (`signal.q`, general purpose)

- [x] **G2.1 FFT.** Iterative radix-2 FFT and inverse FFT on complex data held as (real; imaginary) float
  vectors, vectorised per stage; `rfft`/`irfft` for real signals; zero-padding to powers of two. *Accept:*
  matches a naive DFT to 1e-9 on random input; `ifft fft x = x`; Parseval's theorem; a pure tone peaks in the
  right bin; timing on 2^20 points recorded in the log.
- [x] **G2.2 Windows and PSD.** Hann and Tukey windows; Welch PSD (segment length, overlap, one-sided,
  physical units: strain²/Hz). *Accept:* white Gaussian noise of variance σ² at rate fs gives a flat PSD of
  2σ²/fs within a few percent; a tone's power lands in its bin.
- [x] **G2.3 Whitening, band-pass, resampling.** Whiten by an ASD (interpolated to the segment's frequency
  grid) with the right normalisation, so white noise stays at unit variance; band-pass by frequency mask
  with tapered edges; downsample 4096 -> 2048 Hz in the frequency domain, which also removes aliasing.
  *Accept:* whitened coloured noise has variance 1 ± 0.05 and a flat PSD; a downsampled tone keeps its
  frequency and amplitude.
- [x] **G2.4 Matched filtering.** The SNR time series of data against a template (frequency-domain
  correlation, normalised by the template's norm under the PSD). *Accept:* a template injected into white
  noise at SNR 12 is recovered at about 12 at the right time; pure noise gives SNRs around 1–4. This is both
  the baseline the network is judged against and a check of the waveforms (G3.4).

### G3 — Waveforms and detectors (`gw.q`)

- [x] **G3.1 TaylorF2.** Frequency-domain post-Newtonian inspiral: amplitude ∝ f^(-7/6), phase to 3.5PN for
  aligned spins, given masses, spins, distance, inclination, coalescence time and phase, ending at the ISCO
  frequency. *Accept:* the time–frequency track from the stationary phase relation matches the Newtonian
  chirp formula at low frequency; amplitude scales as 1/distance; a GW150914-like chirp mass gives a
  sensible time in band from 20 Hz (a fraction of a second).
- [x] **G3.2 IMRPhenomB.** Inspiral–merger–ringdown in closed form (Ajith et al. 2011, PRL 106, 241101):
  piecewise amplitude, phase coefficients and transition frequencies from the paper's table, transcribed
  carefully and cited in comments. *Accept:* agrees with TaylorF2 at low frequency (phase difference small
  across 20–40 Hz after aligning); continuous at the transition frequencies; merger frequency scales as
  1/total mass; looks right in the time domain (chirp, peak, ringdown). If LALSuite is available, an optional
  test compares against it.
- [x] **G3.3 Detector response.** H1 and L1 vertex positions and arm directions; Greenwich sidereal time from
  GPS time; F+ and F× for sky position (ra, dec) and polarisation ψ; inter-detector time delay; the projected
  strain h = F+ h+ + F× h× shifted by the delay. *Accept:* antenna patterns are bounded (F+² + F×² ≤ 1) and
  match known values for a few published configurations; delays never exceed the H1–L1 light travel time
  (about 10 ms).
- [x] **G3.4 End-to-end check on GW150914.** Whiten the real 32 s H1 and L1 data, matched-filter with a
  PhenomB template at the catalogue masses (about 36 + 31 M☉), and find the peak at the event's GPS time with
  an SNR roughly in line with the published values (about 20 in H1 and 14 in L1). *Accept:* both detectors
  peak within 10 ms of the event time with SNR above 10. If this fails, the waveforms or whitening are wrong;
  don't go on to G4 until it passes.

### G4 — Training data

- [ ] **G4.1 Noise library.** Whiten each noise event's 4096 s files in chunks (local PSD per chunk), cut out
  ±8 s around the event itself, drop chunks with data-quality gaps, and save the whitened 2048 Hz streams with
  their PSDs per detector. Split events into train, validation and test (G-D9).
- [ ] **G4.2 Injections.** Draw parameters (masses 10–80 M☉, aligned spins, isotropic sky and orientation,
  random polarisation and phase), generate h+ and h×, project onto H1 and L1, whiten with the noise chunk's
  PSD, scale to a target network SNR (uniform in a configurable range, e.g. 5–30), place the merger at a
  random time in the window (e.g. 0.6–0.9 s), and add to a noise window from both detectors at the same time.
  *Accept:* the measured optimal SNR of each injection matches its target to 1%; the merger lands where
  asked; H1–L1 offsets stay within ±10 ms.
- [ ] **G4.3 Glitches.** Synthetic sine-Gaussians and short "blips" in one detector at a time, as labelled
  negatives (the thesis's Gaussian glitches, extended); real glitches arrive anyway through real noise.
- [ ] **G4.4 Training on generated data.** qSolve core: `.qs.train` accepts a generator (a function of the
  epoch returning `` `X`Y ``) with a fixed validation set, so every epoch sees fresh noise windows and
  injections. *Accept:* a test shows a generator is called once per epoch, the validation set stays fixed,
  and seeded runs are reproducible.

### G5 — Models

- [ ] **G5.1 MLP baseline.** 2 × 2048 whitened inputs (or 2 × 1024), dense hidden layers in the thesis's spirit,
  sigmoid output with `bce`. Record speed, memory and validation curves.
- [ ] **G5.2 Conv1D in the qSolve core.** New layer types: 1-D convolution with channels (H1 and L1 as two
  input channels), max pooling and flatten, mixed with dense layers in one model; forward and backward passes
  vectorised; gradient checks against finite differences as for dense layers; save, load and summary
  updated. *Accept:* the gradient check passes for conv, pool and flatten stacks; a small CNN solves a toy
  task (e.g. finding a pulse anywhere in a sequence) that an MLP of similar size struggles with.
- [ ] **G5.3 Optional: LSTM or GRU layer** (thesis homage; owner's call): backpropagation through time,
  gradient-checked, run on downsampled sequences.
- [ ] **G5.4 Training runs at scale.** Threads (`-s 4`), epochs on generated data, early stopping on the
  synthetic validation set, best model saved with its configuration. Log wall-clock times.

### G6 — Evaluation on simulations

- [ ] **G6.1 Background and false-alarm rate.** Time slides on the held-out test noise give many hours of
  noise-only coincident windows; score them all; map score thresholds to false alarms per hour, day and month.
- [ ] **G6.2 Efficiency.** Injections into test noise across SNR bins (e.g. 4–30); detection efficiency per
  bin at thresholds for false-alarm rates of 1/hour, 1/day and 1/month; ROC curves. Compare with the
  matched-filter baseline (G2.4) on the same injections, using a small template bank built from the same
  waveform model.
- [ ] **G6.3 Robustness.** Efficiency split by mass and mass ratio; false alarms caused by synthetic glitches;
  a shuffled-label control (the network must do no better than chance).

### G7 — Real detections

- [ ] **G7.1 Scan.** Slide the 1 s window over each test event's 32 s H1 and L1 data (whitened with a PSD from
  nearby data; a 4096 s file for a few events gives a better PSD) in steps of about 1/8 s; cluster peaks;
  record the top score and its time.
- [ ] **G7.2 Results.** For every confident BBH in scope, held out from training noise: catalogue name, run,
  masses, catalogue network SNR, our top score, the time offset from the catalogue GPS time, and whether it
  was found at each false-alarm rate. Plot found and missed events against catalogue SNR. Note the loud
  events and any surprises (e.g. GW190521's very high mass, near the edge of the window).
- [ ] **G7.3 Negative check.** Scan stretches of real noise with no catalogued event, including loud glitches
  if we can find them, and confirm the false alarms match the G6.1 estimate.

### G8 — Dashboard and documentation

- [ ] **G8.1 Dashboard.** A gravitational-waves view: the event list with our result per event; the whitened
  H1/L1 strain and a spectrogram around an event; the network's score over time; injecting a chosen chirp
  into real noise and watching the detector respond.
- [ ] **G8.2 README.** A section with results stated plainly (efficiency curve, real events found and missed),
  data acknowledgement (GWOSC, LIGO/Virgo/KAGRA, CC BY 4.0), references, and how to rerun everything.

---

## 6. Risks

- **Waveform accuracy (G3):** a transcription error in PhenomB's coefficients would quietly train the
  network on wrong chirps. The G3.4 check on GW150914 is the gate.
- **Memory and speed:** 2 × 2048 inputs, a CNN and large generated batches in q. Fall back to 1024 Hz,
  smaller batches and threads; profile before optimising.
- **FFT speed in q:** thousands of FFTs per epoch. Vectorise the radix-2 stages, and precompute twiddle
  factors and bit-reversal permutations per size.
- **Data-quality gaps** (NaNs) in noise files: skip affected chunks rather than filling them.
- **Leakage:** time-based splits (G-D9), and test events excluded from training noise.
- **Download failures and rate limits:** resumable downloads, caching, and small default lists.

## 7. References

- GWOSC, gwosc.org; Abbott et al., *Open data from the first and second observing runs of Advanced LIGO and
  Advanced Virgo*, SoftwareX 13 (2021) 100658.
- Ajith et al., *Inspiral-merger-ringdown waveforms for black-hole binaries with nonprecessing spins*, PRL 106,
  241101 (2011) (IMRPhenomB).
- George & Huerta, *Deep learning for real-time gravitational wave detection and parameter estimation*, Phys.
  Lett. B 778 (2018) 64.
- Gabbard, Williams, Hayes & Messenger, *Matching matched filtering with deep networks for gravitational-wave
  astronomy*, PRL 120, 141103 (2018).
- The owner's thesis code: github.com/Oiubrab/recurrent_grav_detector.

---

## Progress log

- 2026-10-24: G1.1 complete. Added `gw.q` catalogue parsing/cache, newest-version de-duplication,
  confidence/mass and detector-availability selectors, and offline fixture plus optional cached-live tests.
  The live endpoint returned 671 versioned entries and 433 latest-per-name rows; null catalogue fields are
  present and preserved, so selection must explicitly filter them.
- 2026-10-25: G1.2 complete. Added resumable `scripts/fetch_gw.sh`, six verified O1/O2/O3a/O3b noise defaults,
  explicit event-file/name lists, and automatic confident-BBH 32 s selection. GWOSC newer detail versions
  can omit strain URLs, so the downloader tries older versions until it finds an H1/L1 4 kHz `txt` pair.
  Dry-runs resolved all six 4096 s pairs; a real GW150914 H1/L1 32 s fetch passed gzip validation and the
  repeat run skipped both cached files. The multi-GB default library and full test-event set were not bulk
  downloaded in this step.
- 2026-10-25: G1.3 complete. Added `.qs.gw.strain` to stream-decompress GWOSC text, parse GPS/rate/strain,
  preserve `nan` as null samples, remove temporary text, and cache the dictionary beside the source. The
  synthetic test reloads from cache after deleting its gzip input; the real GW150914 file yielded GPS
  1126259447, 4096 Hz and 131072 samples. Full suite: 95 passed.
- 2026-10-25: Owner approved BNS as a separate 1–3 M☉ track using 256 s windows at 2048 Hz and its own
  model; BBH remains 10–80 M☉ in 1 s windows. This resolves the G2/G4 window design gate.
- 2026-10-25: G2.1 complete. Added radix-2 complex FFT/IFFT and real transforms with power-of-two padding.
  Seeded random complex input matched the naive DFT within 1e-9; round-trip, Parseval, tone-bin and padded
  real inverse checks passed. A 2^20-point forward FFT took 0.762 s in this environment.
- 2026-10-25: G2.2 complete. Added Hann and Tukey windows and one-sided Welch PSD with physical strain²/Hz
  normalization. Seeded white noise at variance 0.25 and 1024 Hz measured the expected interior-bin level
  2σ²/fs within 3%; a 73 Hz tone landed in its bin. An initial compact power expression was invalid under
  q's right-to-left evaluation; splitting real/imaginary squares fixed the PSD values.
- 2026-10-25: G2.3 complete. Added ASD interpolation, frequency-domain whitening, tapered band-pass, and
  4096→2048 Hz frequency-domain resampling. A frequency-shaped colored-noise test whitened to unit variance
  and flat PSD; the filter retained 100 Hz and suppressed 700 Hz; resampling retained a 200 Hz tone's
  amplitude and bin. Full suite: 113 passed.
- 2026-10-25: G3.4 complete: **the gate passes.** New: `.qs.sig.matchedFilterMax` (the template and its
  90-degree-shifted copy, SNR as their magnitude, plus the best phase; `.qs.sig.atan2` moved to `signal.q`),
  `.qs.gw.template` (face-on PhenomB time series with its reference at t=0 and the offset of its peak) and
  `.qs.gw.search` (Welch PSD from the same 32 s, 10% Tukey taper, edge margin). On GW150914's 32 s files with
  detector-frame masses 38.1 + 33.0 (GWTC-2.1 source masses 34.6 + 30.0 times 1+z, z about 0.10 at 470 Mpc):
  **H1 SNR 19.1, L1 SNR 13.1** (published single-detector values about 20 and 13-14); merger times +25 ms (H1)
  and +18 ms (L1) from the catalogue's 1126259462.4 (rounded to 0.1 s), so **L1 leads H1 by 7 ms**, as
  published. Source-frame masses give a lower H1 SNR (17.2), as they should. Background elsewhere in the
  file: at most 4.6 (H1) and 7.7 (L1). The acceptance was adjusted from the plan's "within 10 ms of the event
  time", which the catalogue's rounding makes impossible: test `gw_finds_gw150914` checks SNRs (H1 17-22, L1
  11-16), both times within 50 ms, L1 leading by 4-10 ms, and SNR above 1.5 times the background.
  **Found on the way:** the first and last ~4-6 s of a 32 s file give huge false SNRs (up to 500). The whitening
  filter is about one PSD segment (4 s) long, so data that close to an edge is corrupted; searches must keep
  a margin of at least the PSD segment plus the template length (6 s here). G4/G7 should use longer stretches
  and discard the edges. Test `gw_matched_filter_any_phase`: a template rotated by 1 rad in white noise is
  recovered at SNR about 10 with phase about 1, where the one-phase filter sees about 10 cos 1 at the true lag
  (over all lags it gets more back by shifting slightly, since a chirp's phase and time trade off). Full
  suite: 131 passed.
- 2026-10-25: G3.3 complete. In `gw.q`, all transcribed from LALSuite (`lal/lib/tools/LALDetectors.h`,
  `CreateDetector.c`, `DetResponse.c`; `lal/lib/date/TimeDelay.c`, `XLALSiderealTime.c`, `XLALLeapSeconds.h`):
  `.qs.gw.detectors` (H1/L1 vertex and arm unit vectors, Earth-fixed), `tensor` ((x x' - y y')/2), `leaps` (GPS-UTC)
  and `gmst` (IAU 1982 polynomial on UTC), `antenna` (F+, Fx as XLALComputeDetAMResponse), `delay` (arrival after
  the Earth's centre), `lightTime`, `project` (F+ h+ + Fx hx times exp(-2 pi i f dt)), and an `atan2` (q's `atan`
  takes one argument). Tests: tensors against the matrices printed in CreateDetector.c's documentation (1e-6;
  LAL's arm vectors are unit and orthogonal only to about 2e-7); GMST at 2000-01-01 12:00 UTC is 280.4606
  degrees and repeats after a sidereal day; leap counts 13/17/18 at 2000/2015/2021; over 500 random skies
  F+^2+Fx^2 <= 1, unchanged by psi, a quarter turn of psi swaps |F+| and |Fx|; a source straight overhead gets
  more than 0.99 of the full response; H1-L1 light time 10.013 ms, 2000 random skies never exceed it and reach
  98% of it; the projection's phase factor is exactly exp(-2 pi i f dt), and an edge-on chirp's peak moves
  between detectors by the delay difference (to 0.5 ms). Gotchas: `over` is a keyword; with both polarisations
  the loudest cycle moves by a fraction of a period between detectors, so peak-time checks need one
  polarisation. Full suite: 129 passed.
- 2026-10-25: G3.2 complete. `.qs.gw.phenomB` (same arguments and result as `taylorF2`, plus `fcut`, `fmerger`,
  `fring`, `sigma`) transcribed from LALSuite's `LALSimIMRPhenom.c` (`ComputeIMRPhenomBParams`,
  `IMRPhenomBGenerateFD`): the phenomenological table in `.qs.gw.pbTable` (coefficients of eta, eta chi,
  eta chi^2, eta^2, eta^2 chi, eta^3), parameters in `.qs.gw.pbParams`, the three-piece amplitude with w1/w2
  continuity, and LAL's `exp(-i psi)` with our `2 pi f tc` added (LAL's FD phase has no linear term). Spins
  combine as chi = (m1 chi1 + m2 chi2)/M, as LAL's `XLALSimIMRPhenomBComputeChi`. Tests: parameters,
  amplitude and phase against a Python transcription of LAL's C at 20-300 Hz (relative 1e-12); no jump at
  fMerger or fRing; frequencies scale as 1/M; spin raises fMerger; in 20-40 Hz the phase follows TaylorF2 to
  within 0.03-0.17 rad after removing a free time and phase offset (measured for 36+29, 15+10 spinning,
  50+40) and the amplitude is 3-12% lower (PhenomB has PN amplitude terms, our TaylorF2 is leading order);
  time domain rises in frequency and peaks before tc. **Note for G4.2:** tc is a linear-phase reference; the
  time-domain peak lands 20-80 ms before it for 10-60 Msun components, so place mergers by measuring the peak
  (or by a per-waveform offset), not by tc. No LALSuite here, so the optional live comparison wasn't run
  (`pip install lalsuite` would allow it). Full suite: 125 passed.
- 2026-10-25: G3.1 complete (resumed in a new session). Three fixes to the checkpointed `.qs.gw.taylorF2`:
  - **Convention:** it built `exp(+i psi)`, the complex conjugate of LAL's `exp(-i(psi - pi/4))`. With our FFT's
    `exp(-2 pi i f t)` (the same as LAL's) that is a time-reversed chirp: falling in frequency, centred at -tc. The
    old tests only checked the size of the phase slope, so they couldn't see it. Now plus is
    `A (1+cos^2 i)/2 exp(-i(psi - pi/4))` and cross is `-i A cos i exp(-i(psi - pi/4))`, as in LAL.
  - **q precedence:** `(2 pi f tc)-phic+(PN term)` is `2 pi f tc - (phic + PN term)` in q, which flipped the whole
    PN series (the old version had the same problem with `pi/4`). The 3PN constant had the same trap. Signed terms
    are now summed as lists.
  - **Spin terms:** all of LAL's aligned-spin phase terms (XLALSimInspiralPNPhasing_F2, citing arXiv:1303.7412):
    1.5PN, 2.5PN, 3PN and 3.5PN spin-orbit; 2PN and 3PN spin-spin and self-spin (black hole quadrupole-monopole
    1). Coefficients were transcribed from LAL's decimal literals as exact fractions, in `.qs.gw.f2coeffs`.
  New tests: every coefficient against a Python transcription of LAL's C (36+29 Msun, chi 0.3/-0.2, relative
  1e-12); amplitude against LAL's own leading-order expression (agrees to 1.6e-8, which is LAL's MRSUN being a
  different rounding of MTSUN times c); swapping the bodies changes nothing; and **in the time domain** (inverse
  FFT on a 4 s grid) the chirp peaks within 50 ms before tc, its zero crossings speed up towards tc, and there is
  no energy before it starts. That last test fails for the old conjugate convention. Full suite: 121 passed.
  Note for G3.2: PhenomB must use the same `exp(-i ...)` convention, and its TaylorF2-matching check (G3.2
  acceptance) can now rely on this function.
- 2026-10-25: Low-credit checkpoint during G3.1. Added `.qs.gw.taylorF2` with leading-order SPA amplitude,
  inclination polarizations, ISCO cutoff, nonspinning phase through 3.5PN, and the leading 1.5PN aligned-spin
  beta term, plus `.qs.gw.newtonianTime`. Low-frequency phase-slope, inverse-distance amplitude, signed-spin,
  and cutoff tests pass; full suite: 117 passed. **G3.1 remains unticked:** higher-order spin terms are not
  implemented or cross-checked, and the waveform is not yet ready for G3.2/G3.4. Resume by verifying the
  PN coefficient convention against a primary source/LALSuite and complete the spin terms before ticking.
- 2026-10-25: G2.4 complete. Added PSD-normalized circular frequency-domain matched filtering. A known
  template injected at target SNR 12 was recovered near 12 at the expected circular offset; noise-only SNR
  had unit variance and maximum below 5. Full suite: 115 passed.
- 2026-10-24: Owner chose pure-q waveforms, the default download budget, an LSTM/GRU, and BNS in scope.
  Updated G-D1/G-D8; BNS needs a separate longer-window design before signal-processing and training stages.
- 2026-10-24: Plan written. Checked: GWOSC's event API (671 catalogue entries, 417 confident with an SNR),
  that strain comes as `txt.gz` at 4 and 16 kHz in 32 s and 4096 s files, and file sizes (32 s, 4 kHz: 1.3 MB;
  4096 s, 4 kHz: 164 MB). LALSuite isn't installed here, so waveforms are planned in pure q. Read the
  thesis code: IMRPhenomPv2 via lalsimulation, aLIGO design-curve Gaussian noise, 4096 Hz, 1 s windows,
  dense and LSTM Keras models.
