# Finding gravitational waves in LIGO data

A real-world test of qSolve. It revisits the owner's honours thesis
([recurrent_grav_detector](https://github.com/Oiubrab/recurrent_grav_detector)). A neural network looks for binary
black hole mergers in real detector data from LIGO's Hanford (H1) and Livingston (L1) observatories. The network,
the signal processing, the waveforms and the search are all pure q. The one exception is a small Python converter
for GWOSC's HDF5 files.

**In short:**
- **The best model, `gw_cnn_4`, finds half of all simulated mergers at network SNR 9.8.** That is at 1 false alarm
  per hour. At 1 false alarm per day, it finds half at SNR 12.2.
- **It finds 29 of 35 confident O3 detections at under 1 false alarm per hour.** It was never trained on any of
  them.
- **GW150914 and GW231226 are found clearly.** GW231226 is louder than all 412 h of its day's background.
- **A plain matched filter does better at 1 false alarm per hour, and worse at 1 per day.** At 1 per hour it finds
  half of all signals at SNR 7.1. At 1 per day it needs SNR 16.4, because it has no defence against glitches.
- **A search of about 98 h of public data found no new candidates.** It did recover GW190707_083226, an event
  reported only by an independent group (IAS) and in no LIGO-Virgo-KAGRA (LVK) catalogue.

LIGO's own pipelines find these events at false-alarm rates far below 1 per year, so this detector is not
competitive with them. The aim is a working, honest pipeline built in q.

The development history, every number and every decision are in
[gw_detection_plan.md](../../plans/completed/gw_detection_plan.md) and
[gw_open_search_plan.md](../../plans/completed/gw_open_search_plan.md).

---

## How it works

### Signal processing and waveforms

- **Signal processing** is in [src/signal.q](../../src/signal.q), part of the base library because none of it is
  specific to gravitational waves. It covers:
  - FFT;
  - Welch power spectra;
  - whitening, band-pass and resampling;
  - spectrograms;
  - phase-maximised matched filtering.
- **Waveforms and detectors** are in [src/gw.q](src/gw.q).
  - Two waveform models, TaylorF2 and IMRPhenomB, in the frequency domain. Both are checked against LALSuite's
    conventions.
  - Antenna patterns and light-travel delays for H1 and L1.
  - With these templates, a matched filter finds GW150914 at SNR 19 in H1 and 13 in L1, 7 ms apart.

### Training data

New training data is made at the start of every epoch:
- **The noise is real.** It is whitened detector noise taken from the public 4096 s files around six events, and
  from O4a bulk data.
- **Chirps** are injected into that noise at a chosen network SNR: 10–80 M☉, aligned spins, random sky positions.
- **Three kinds of negatives** are labelled noise, and they are what made the detector usable on real data:
  - simulated glitches (blips and sine-Gaussians);
  - real glitches mined from the noise, and windows an earlier model scored high (hard negatives);
  - chirps in one detector only. H1 and L1 are nearly aligned, so a real signal shows up in both.

### The model

`gw_cnn_4` is a 1-D convolutional network with about 35k parameters. Its input is 1 s of both detectors at
2048 Hz, given as two channels of 2048 samples.

| Layer | Shape |
|---|---|
| Conv1D, 8 filters, kernel 32, ReLU, max pool 4 | 2×2048 → 8×504 |
| Conv1D, 16 filters, kernel 16, ReLU, max pool 4 | → 16×122 |
| Conv1D, 32 filters, kernel 8, ReLU, max pool 4 | → 32×28 |
| Dense, 32 units, ReLU | 896 → 32 |
| Dense, 1 unit, sigmoid | 32 → 1 |

Triggers are ranked by the output before the sigmoid (the logit), because the sigmoid's output saturates near 1.

Each model fixed a problem the one before it showed on real data:

| Model | What changed | O3 detections found, of 35 (under 1/h) |
|---|---|---|
| `gw_cnn_1` | O1–O3 noise and simulated glitches, 20 epochs | 12 |
| `gw_cnn_2` | adds real glitches and hard negatives; early stopping | 29 |
| `gw_cnn_3` | adds O4a noise | — |
| `gw_cnn_4` | adds single-detector chirps as negatives | 29 |

- **`gw_cnn_3` fired on a chirp in one detector alone.** On the O4a day, its loudest background triggers were
  GW231226 in H1 slid against L1 noise. `gw_cnn_4` fixed that: its background tail dropped from logit 16 to 4,
  while GW231226 stayed at 17.
- **A recurrent network didn't learn the task.** A GRU front end (in the core, gradient-checked) reached
  chance-level validation loss on raw whitened samples. The natural next try is a GRU over the CNN's features,
  which [model_expansion_plan.md](../../plans/future/model_expansion_plan.md) makes possible.

## Results

**How false-alarm rates are measured.** Each rate comes from time slides: one detector's noise is slid against the
other's by more than the light-travel time, so nothing found there can be a real signal. The background is always
measured this way and never on the data the rate is applied to.

### Simulated signals: the CNN against a matched filter

The test noise is GW150914's held-out O1 noise: 77 chunk pairs, 67 minutes, never used for training or validation.
Its time slides give 84.5 h of background. Both detectors score the same 3000 injections: network SNR uniform from
4 to 30, masses 10–80 M☉, random sky positions, orientations and spins.

The matched filter is a baseline, not a LIGO pipeline:
- 41 IMRPhenomB templates (10–80 M☉, mass ratio up to 4, zero spins);
- network SNR is H1's SNR plus L1's loudest within 12 ms, combined in quadrature;
- its thresholds come from its own time-slide background, as the CNN's do.

Share of injections found, by network SNR:

| Network SNR | CNN, 1/hour | Matched filter, 1/hour | CNN, 1/day | Matched filter, 1/day |
|---|---|---|---|---|
| 6–8 | 10% | 48% | 1% | 0% |
| 8–10 | 38% | 94% | 7% | 0% |
| 10–12 | 70% | 100% | 30% | 0% |
| 12–14 | 83% | 100% | 62% | 0% |
| 14–16 | 93% | 100% | 83% | 10% |
| 16–18 | 96% | 100% | 89% | 65% |
| 20 and up | 95–98% | 100% | 93–97% | 98–100% |
| **Half found at** | **SNR 9.8** | **SNR 7.1** | **SNR 12.2** | **SNR 16.4** |

- **At 1 false alarm per hour the matched filter wins clearly.** Its threshold (network SNR 7.4) is close to what
  Gaussian noise alone allows.
- **At 1 false alarm per day the CNN wins.** At least four of the matched filter's background triggers in 84 h are
  above network SNR 16. Gaussian noise essentially never gets that loud, so these are glitches in the O1 noise.
  - A bare matched filter has nothing to reject them; LIGO's pipelines add a χ² signal-consistency test.
  - The CNN learned to reject glitches in training.
- **Efficiency varies little with the binary.** At SNR 10 and up, the CNN finds 86–94% at 1/hour for total masses
  from 20 to 100 M☉, and 99% above 120 M☉. Mass ratio makes almost no difference (92–93% throughout).
- **About 3–4% of signals above SNR 20 are still missed.** Why hasn't been examined yet.

### Real detections

- **35 confident O3 detections**, all in the trained mass range and never trained on. The background is 4.85 h of
  time slides within each event's own data. `gw_cnn_4` finds 29 of them at under 1 false alarm per hour, and 17 at
  under 1 per 5 hours.
- **GW150914**: logit 12.2 within 0.1 s of the merger. Elsewhere in the file the logit is at most −2.2.
- **GW231226** (O4a): the loudest trigger of its whole day, above all 412 h of that day's background.

### Controls

- **Shuffled training labels.** `gw_cnn_4`'s recipe was trained with its labels shuffled. Its validation loss was
  0.6935, against 0.693 for pure guessing. It finds 0–2.5% of signals at every SNR. So nothing in the pipeline
  lets a model "find" signals without learning them.
- **Simulated glitches.** 1000 windows, each with a blip or a sine-Gaussian in one detector. None scores above
  either threshold.
- **Real noise.** On the O4a day (20.6 h, with GW231226 excluded), 11 triggers exceed the 1/hour threshold, against
  20.6 expected. None exceeds the 1/day threshold, against 0.86 expected. So the simulated background predicts real
  false alarms, if anything pessimistically.

## Looking for missed events

The search covered six whole days of public bulk data, about 98 h in all:
- **The O4a day**, around GW231226;
- **Four O3a days**, chosen around events reported only by the independent IAS pipeline (Olsen et al. 2022);
- **One O2 day**, around 170720, a marginal candidate in LIGO's first catalogue (GWTC-1).

The search uses:
- **The same data-quality cuts as LIGO's own binary searches.** It only searches time when both detectors have
  data, the CBC_CAT1 and CBC_CAT2 vetoes are clear, and no hardware injection is running.
- **Whitening done as in training:** 64 s chunks, each whitened with its own spectrum, keeping the middle 52 s.
- **A trigger every 0.25 s.** Triggers are local maxima at least 1 s apart.
- **Background from 10 time slides per day**, or 20 for the O4a day.

| Day around | Searched | What happened |
|---|---|---|
| GW231226 (O4a) | 20.6 h | GW231226 is the loudest trigger, above all background. Nothing else stands out |
| GW190707_083226 (O3a, IAS only) | 19.7 h | **Found** at 1 false alarm per 47 h, the day's second-loudest trigger. GW190706_222641 is above all background. GW190707_093326 (SNR 12.7, a low-mass system) is missed |
| GW190818_232544 (O3a, IAS only) | 16.9 h | Missed: SNR 8.1 |
| GW190906_054335 (O3a, IAS only) | 12.0 h | Missed: SNR 7.9, scored at 3.1 false alarms per hour |
| GW190814_192009 (O3a, IAS only) | 14.3 h | Not searchable: L1's good data around it is only 52 s long, shorter than the 64 s chunk. GW190814 itself falls in a second the bulk release marks as missing H1 data |
| 170720 (O2, GWTC-1 marginal) | 14.4 h | Not seen by the CNN or by the matched filter |

Missing two IAS events at SNR 8 is expected. At that strength the CNN finds 38% of simulated signals at 1 false
alarm per hour.

**Vetting.** The 29 loudest uncatalogued triggers were followed up with the 41-template matched filter and the
data-quality flags.
- A trigger passes the check (`consistent`) when all of these hold:
  - each detector's SNR is above its own background;
  - the arrival times differ by less than 12 ms;
  - no CBC-relevant data-quality flag is set.
- 28 of the 29 fail.
  - Two are loud glitches in H1 (matched-filter SNR 17.9 and 89.6, with L1 below 3.4).
  - One sits seconds before both detectors' data is flagged bad, near 170720.
  - The rest have one detector at noise level.
- The one trigger that passes has network SNR 6.6, which is noise level.

**Public alerts.** GraceDB's public alerts for the six days hold nothing new.
- O2 had no public alerts; they began in O3.
- The O3a alerts are all catalogued events.
- The O4a day has six low-significance alerts, none in the GWTC-4 catalogue.
  - At five of them, the search's loudest trigger scores 41 to 1,300 false alarms per hour.
  - The sixth fell outside the data the search could use.

**Conclusion: no new candidates.**
- On every day, the number of uncatalogued triggers at each false-alarm rate matches what the background predicts.
  For example, on the O4a day, 2 triggers fell under 0.1 per hour where 2.1 were expected, and 15 under 1 per hour
  where 20.6 were expected.
- GW190707_083226, which is in no LVK catalogue, was found.
  - The matched filter confirms it in both detectors: network SNR 7.9, with H1 at 6.3 and L1 at 4.8, 3.9 ms apart.
  - At 1 false alarm per 47 h, though, it would not stand out on its own in a blind search of many days.

## Limits and next steps

- **The CNN is about 2.7 SNR units less sensitive than a matched filter on clean noise** (1 false alarm per hour).
  Deeper networks (residual blocks, batch norm), and a GRU or attention over the conv features, are the obvious
  tries. Both need the general layer core in
  [model_expansion_plan.md](../../plans/future/model_expansion_plan.md).
- **The matched filter needs a χ² test before it is a fair comparison at low false-alarm rates.**
- **1 false alarm per month and below** needs about 720 h of background, which hasn't been measured.
- **O1 bulk data** needs `gwosc_bulk.py` to support its older LOSC file names.
- **Binary neutron stars** (1–3 M☉, minutes-long signals) are a separate track:
  [gw_bns_plan.md](../../plans/future/gw_bns_plan.md).
- **An unexplained crash.** One whole-day search failed three times with `'stop` while loading a segment, and
  succeeded when split into two runs. The cause hasn't been found.

## Reproducing

Run everything from the repository root. A model given by name (`gw_cnn_4`) is read from and saved to
`projects/gw/models/`.

```bash
projects/gw/scripts/fetch_gw.sh                                   # 4096 s event files for the noise library (a few GB)
projects/gw/scripts/fetch_gw.sh --test-events --duration 32       # 32 s files around the O3 test detections
q projects/gw/experiments/gw_cnn.q -s 4 -q -save gw_cnn_1         # train (options in the file's header)
q projects/gw/experiments/gw_events.q -s 4 -q -model gw_cnn_1     # the O3 detections, with time-slide background
q projects/gw/experiments/gw_eval.q -s 4 -q -model gw_cnn_1       # background, efficiency and glitch checks
q projects/gw/experiments/gw_mf_eval.q -s 4 -q -model gw_cnn_1    # the same injections against a matched filter
python3 -m venv .venv && .venv/bin/pip install h5py numpy         # for bulk data only
projects/gw/scripts/search_days.sh gw_cnn_1 10 O4a:1387577000     # fetch, search and time-slide whole days
q projects/gw/experiments/gw_vet.q -s 4 -q -run O4a -start 1387577000 -stop 1387663400   # follow up the top triggers
q tests/run.q -only gw                                            # this project's tests (they skip without the data)
```

`gw_cnn_4`'s full recipe (hard negatives mined with `gw_cnn_3`, O4a noise, single-detector negatives, early
stopping) is in the 2026-10-28 entries of [gw_open_search_plan.md](../../plans/completed/gw_open_search_plan.md).

The dashboard's **Waves** tab (`q app/dashboard.q -s 4`) shows either a downloaded event or a chirp you inject into
the held-out noise. For either one it plots:
- the whitened strain;
- a spectrogram;
- the network's output over time.

It also gives the verdict at each measured false-alarm rate.

## Layout

```
projects/gw/
  README.md         this write-up
  src/gw.q          catalogue, waveforms, detectors, noise library, injections, glitches, search, background,
                    matched-filter baseline, vetting (.qs.gw)
  experiments/      gw_mlp.q, gw_cnn.q (training); gw_eval.q, gw_mf_eval.q (simulated signals);
                    gw_events.q (O3 detections); gw_bulk_scan.q, gw_vet.q (open-data search and vetting)
  scripts/          fetch_gw.sh (event strain), gwosc_bulk.py (bulk HDF5 to raw files), search_days.sh
                    (fetch and search whole days), gw_noise_events.txt (the noise library's events)
  tests/            test_gw.q, test_bulk.q, fixtures/
  data/             downloads: allevents.json, strain/, noise/, bulk/                       (git-ignored)
  results/          eval/ (thresholds, injections), search/ (triggers, background, vetting)  (git-ignored)
  models/           gw_cnn_1-4, gw_gru_1, control_shuffled_cnn                              (git-ignored)
```

## Data and references

**Data.** This uses data obtained from the Gravitational Wave Open Science Center (https://gwosc.org), a service of
the LIGO Scientific Collaboration, the Virgo Collaboration and KAGRA, under the CC BY 4.0 licence. LIGO is funded
by the U.S. National Science Foundation; see https://gwosc.org/acknowledgement/ for the full acknowledgement.

**References.**
- Abbott et al., *Open data from the first and second observing runs of Advanced LIGO and Advanced Virgo*,
  SoftwareX 13 (2021).
- Ajith et al., *Inspiral-merger-ringdown waveforms for black-hole binaries with nonprecessing spins*, PRL 106
  (2011): IMRPhenomB.
- LALSuite (https://git.ligo.org/lscsoft/lalsuite), for waveform and detector conventions.
- George and Huerta, *Deep learning for real-time gravitational wave detection and parameter estimation*, PLB 778
  (2018), and Gabbard et al., *Matching matched filtering with deep networks for gravitational-wave astronomy*,
  PRL 120 (2018): CNN detectors on whitened strain.
- Olsen et al., *New binary black hole mergers in the LIGO-Virgo O3a data*, PRD 106 (2022): the IAS events.
