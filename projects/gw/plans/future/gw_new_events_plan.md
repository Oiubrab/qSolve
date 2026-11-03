# New-events plan: giving the GW search a real chance of finding something

Follow-on to [../completed/gw_open_search_plan.md](../completed/gw_open_search_plan.md). That plan searched about
98 h of public LIGO data with `gw_cnn_4` and found nothing novel. It recovered GW190707_083226, but IAS had already
reported that event. The results are written up in [../../projects/gw/README.md](../../README.md).

The owner, 2026-11-03: write up what it would take to have a real chance of finding something new. There are three
routes, and they combine:
1. **A more sensitive detector** (stage N1).
2. **Far more data**: weeks or months, not days (stage N2).
3. **Signals the standard searches cover poorly**: high masses, very unequal masses, precessing spins, eccentric
   orbits (stage N3).

Read the "q gotchas" in [../completed/improvemet_plan.md](../../../../plans/completed/improvemet_plan.md) and both GW plans' logs
first.

**How to use this file:** the same as the other plans. Tick an item only when its check passes and
`q tests/run.q` passes. Add a dated line to the Progress log. Commit each finished stage to main (the owner's
rule).

**Current status (2026-11-03):** design only. The open questions N-Q1 to N-Q4 below need the owner.

---

## 1. Why the first search found nothing, and what changes the odds

- **These data have been searched many times, by better searches.**
  - LIGO-Virgo-KAGRA's matched-filter pipelines searched them, and so did independent groups (IAS, and the open
    catalogues from Nitz et al.).
  - Their searches all have signal-consistency tests against glitches.
  - Anything loud enough for us to see, they almost certainly saw first.
- **Our detector is weakest exactly where a new event would most likely be: at low SNR.**
  - At 1 false alarm per hour, `gw_cnn_4` finds half of all signals at network SNR 9.8, where our own 41-template
    matched filter manages 7.1.
  - At SNR 8–10 it finds 38%.
- **98 h is about four days**, out of years of public data.
- **Our training covers the same signals the standard banks cover:** 10–80 M☉, mass ratio up to about 4, aligned
  spins, circular orbits.
  - A network has no template bank.
  - It can be trained on signals outside the standard banks, where those searches lose sensitivity.
  - That is the one place a small, independent search can plausibly add something.

So route 3 is the most realistic way to find something new. It needs route 1 to be worth running, and route 2 to
see enough data. The stages below are in the recommended order. The N2 tooling doesn't depend on N1, so it can be
built in parallel.

## 2. Open questions for the owner

1. **N-Q1, order.**
   - Recommended: N1 first (it is measurable straight away, and its later steps share the model-expansion work).
   - Then N2's runner with the best model.
   - Then N3.
   - The alternative is N2 now with `gw_cnn_4`. That gives a month-scale result sooner, but it repeats the first
     search's sensitivity.
2. **N-Q2, which data.**
   - The least-searched public data is the newest.
   - Check first whether a catalogue covering O4b (April 2024 – January 2025) has been published.
   - If it hasn't, O4b is the best target.
   - If it has, search O3b and the rest of O3a, where the IAS and open-catalogue events give a check on what we
     recover.
3. **N-Q3, reference waveforms for the tests.**
   - N3 needs waveform models we don't have (IMRPhenomD, higher modes, precession, eccentricity), transcribed into q
     as PhenomB and TaylorF2 were.
   - Recommended: install `lalsuite` in the existing `.venv`, only to write reference values into test fixtures, as
     `h5py` is used only for HDF5 conversion. The search itself stays pure q.
   - The alternative is to check by hand against published values. That is slower, and it is easy to get a
     convention wrong.
4. **N-Q4, machine time and disk.**
   - Measured on this shared machine, a day of data with 10 time slides takes 40–110 minutes to search (12–20 h
     analysable).
   - Converted bulk data is about 2.5 GB per day of O3a.
   - The disk has 208 GB free.
   - So a month of data is roughly 30–50 h of machine time. If converted files were kept, it would also need
     75 GB of disk, so N2.1 deletes them after each day is searched.
   - A whole observing run (5–6 months) is one to two weeks of machine time.
   - What budget is acceptable?

## 3. Stages

### N1 — A more sensitive detector

The benchmark throughout is the G6 test from [../completed/gw_detection_plan.md](../completed/gw_detection_plan.md):
- **Noise:** GW150914's held-out O1 noise;
- **Injections:** the same 3000, from the same seed;
- **Scripts:** `projects/gw/experiments/gw_eval.q` and `gw_mf_eval.q`.

`gw_cnn_4`'s marks to beat: half found at SNR 9.8 (1/hour) and 12.2 (1/day). Every new model must also keep 29 of
the 35 held-out O3 detections at under 1/hour, and must not raise the O4a day's background tail.

- [ ] **N1.1 Cheap gains with the current core.** Each change is trained with `gw_cnn_4`'s recipe and evaluated
  on the G6 test:
  - wider and deeper plain CNNs (more filters, a fourth conv layer);
  - training SNRs reaching lower (from 5, not 8);
  - more training noise (more O3 and O4a days from the bulk data already fetched);
  - longer training.

  *Accept:* a table of what each change bought, with the best model saved as `gw_cnn_5`.
- [ ] **N1.2 Modern architectures.** This is [model_expansion_plan.md](../../../../plans/future/model_expansion_plan.md)'s M6.1 (residual
  CNN with batch norm) and M6.2 (a GRU or attention over the conv features). It is tracked there, not here.
  *Accept:* as M6.1; the result feeds N2 and N3.
- [ ] **N1.3 A χ² test for the matched filter.** Add the standard signal-consistency test, which splits the
  template into frequency bands and checks that each band contributes its share. Then rank by the reweighted SNR.
  *Accept:* the matched filter's 1/day threshold falls from network SNR 16 towards Gaussian levels on the G6 test.
  - This makes the comparison with the CNN fair at low false-alarm rates.
  - It also gives N1.4 a glitch-proof second opinion.
- [ ] **N1.4 Rank by both.** Combine the CNN's logit and the matched filter's reweighted SNR into one ranking
  statistic, fitted on the training noise and never on the test noise.
  - The CNN is strong against glitches and the matched filter is strong in Gaussian noise, so the combination
    should beat either alone.
  - Cost: matched-filtering every CNN trigger above a loose threshold, which is cheap (one FFT is 50 ms).

  *Accept:* at 1/day, better than both on the G6 test. Goal: half found below SNR 10.

### N2 — Far more data

- [ ] **N2.1 An unattended runner.** It grows `projects/gw/scripts/search_days.sh` into something that runs for
  weeks:
  - takes a list of days, and skips days already searched;
  - resumes after a crash or a reboot;
  - deletes each day's converted bulk files once it is searched, keeping triggers, background and the summary;
  - downloads the next day while searching the current one, as now;
  - writes one log line per day (searched hours, background hours, loudest uncatalogued trigger).

  *Accept:* a week of days runs untouched, and is killed and restarted midway without losing or repeating work.
- [ ] **N2.2 Fix the `'stop` crash.**
  - The GW190707 day failed three times while loading its third segment after two long ones, and worked when split
    into two runs.
  - Reproduce it, find the cause (memory growth across segments is the first suspect), and fix it, or run each
    segment in its own process.

  *Accept:* that whole day searches in one run.
- [ ] **N2.3 Speed.**
  - Profile a day's search (whitening, window scoring, time slides) and fix the largest cost.
  - Time slides dominate at 10 slides, and they reuse the same whitened data, so batching them is the likely win.

  *Accept:* a measured speed-up with identical triggers.
- [ ] **N2.4 False-alarm rates across days.**
  - A real claim needs rates of 1 per year or less, so a year or more of background.
  - Pool each day's time slides into one background per run (per observing run, not per day).
  - Add slides between days where both detectors were stable.
  - Report each trigger's rate against the pooled background.
  - Count the trials: a month searched means about 30 triggers at 1 per day by chance.

  *Accept:* on a month of data, the count of uncatalogued triggers at each rate matches the pooled background, as
  it did per day.
- [ ] **N2.5 Recovery check against the catalogues.**
  - For every catalogued event in the searched time (GWTC, IAS, the open catalogues), record whether it was found
    and at what rate.
  - Compare the recovery against what the G6 efficiency curve predicts at each event's SNR.

  *Accept:* a table of found against expected on real events. This is the strongest check that the search works
  as measured.
- [ ] **N2.6 Automatic follow-up.**
  - For everything below a chosen rate (start at 1 per week): run the vetting (`gw_vet.q`), the data-quality flags
    and the catalogue and GraceDB checks.
  - Write one candidate report per month.
  - O1 data needs `gwosc_bulk.py` to support LOSC file names (`H-H1_LOSC_4_V1-<gps>-4096.hdf5`).
- [ ] **N2.7 Search a month.** Run on the data chosen in N-Q2, with the best model from N1.
  *Accept:* searched time, pooled background, the recovery table, and the candidate report.

### N3 — Signals the standard searches cover poorly

- [ ] **N3.0 Where do the banks lose?**
  - Collect the literature on the sensitivity of aligned-spin, circular-orbit template searches to:
    - precessing systems;
    - high mass ratios;
    - higher harmonics at high mass;
    - eccentric orbits.
  - Record which regions are said to be least covered, and by how much SNR.
  - Then measure the same thing for our own bank. Inject the new waveforms (N3.1) and record the fraction of SNR
    the 41-template bank recovers.
  - Our bank is far coarser than LVK's, so use our measurement only to rank regions, not to claim the size of the
    gap.

  *Accept:* a short table of target regions, with the expected SNR loss from the literature and from our bank.
- [ ] **N3.1 Waveforms in q.** Each model is transcribed from LALSuite as PhenomB was, and checked against
  reference values (N-Q3), with tests. In order:
  - **IMRPhenomD:** aligned spins, calibrated to mass ratio 18. It replaces PhenomB for the existing space too.
  - **Higher harmonics (PhenomHM style):** they matter at high mass and unequal masses, where PhenomB's dominant
    mode alone misses a large share of the signal.
  - **Eccentric inspiral (TaylorF2Ecc):** this covers low-mass systems with long inspirals. Eccentric
    inspiral-merger-ringdown models are beyond this stage.
  - **Precession by "twisting up" an aligned-spin model (PhenomPv2 style).**
- [ ] **N3.2 Training outside the box.** Extend the injections stage by stage, retraining and evaluating as G6
  each time:
  - total mass up to 200–300 M☉;
  - mass ratio up to 10;
  - precessing spins;
  - eccentricity.

  **The known danger:** short, low-frequency high-mass signals look like blip glitches. Every step must keep the
  glitch false-alarm rate and the O4a background tail where `gw_cnn_4` has them.

  *Accept:* for each region, efficiency at 1/hour and 1/day for the new model, `gw_cnn_4` and our matched filter
  (with N1.3's χ² test), all on the same injections. The interesting result is any region where the network beats
  the bank clearly.
- [ ] **N3.3 Search there.** Rerun N2's searched data with the extended model, ranking as N1.4.
  - Look again at the IAS events missed earlier (GW190818_232544, GW190906_054335).
  - Look at catalogued events at the edge of the new space, such as GW190521 at high mass and GW190814 at a high
    mass ratio.
  - Same follow-up as N2.6.

## 4. If something survives

Something would have to clear all of these before anyone should take it seriously:
- a false-alarm rate small against the time searched (1 per year or less, with the trials counted);
- both detectors above their own backgrounds, arrival times consistent, and no data-quality flags;
- no hardware injection at that time;
- not in any catalogue or GraceDB alert;
- it survives the matched-filter follow-up with the best-fitting waveform from N3.1.

Even then it is a candidate for scrutiny, not a discovery. What to do next:
1. Write it up in the project README with everything needed to reproduce it.
2. Rerun it with a different noise estimate and different slides.
3. Then the owner decides whether to contact anyone.

## Progress log

- 2026-11-03: Plan written from a discussion with the owner after the open search found nothing novel. Not
  started.
