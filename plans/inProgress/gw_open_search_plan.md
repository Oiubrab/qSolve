# Open-data search plan: looking for gravitational-wave events not in the catalogues

Follow-on to [gw_detection_plan.md](gw_detection_plan.md), whose CNN finds GW150914 and 12 of 35 held-out O3
detections at under 1 false alarm per hour. The owner asked: can we run it over recent public LIGO data and spot
events that haven't been reported?

Read the "q gotchas" in [completed/improvemet_plan.md](../completed/improvemet_plan.md) and the GW plan's log first.

**How to use this file:** as the other plans. Tick an item only when its check passes and `q tests/run.q` passes;
add a dated line to the Progress log; commit each finished stage before starting the next (the owner's rule).

**Current status (2026-10-28):** S1 done; S2.1 and S2.2 done on the hour around GW231226 (recovered, but at only
1.7 false alarms/hour: O4a noise fools the O1-O3-trained CNN). Glitch tools (G-B below) built; full-day search and
an O4a-noise retrain next.

---

## 1. What is realistic

- **Data.** GWOSC publishes bulk strain for whole runs: O4a (May 2023 - Jan 2024) and O4b (Apr 2024 - Jan 2025) are
  the most recent, about 3,900 and 3,700 files of 4096 s per detector at 4 kHz. Files are HDF5 (or GWF), about
  30 MB each, with the strain (gzip, NaN where there is no data) and per-second data-quality and hardware-injection
  masks. A day of both detectors is about 42 files, 1.3 GB.
- **Odds.** These runs have already been searched by LIGO-Virgo-KAGRA's pipelines (the GWTC-4 catalogue covers
  O4a) and, for older runs, by independent groups. Our detector is far less sensitive than those matched-filter
  pipelines: it found about a third of known O3 events at 1 false alarm per hour, where they found all of them at
  rates below 1 per year. So **the most likely outcome is recovering known events plus a list of candidates
  consistent with noise.** That is still a real result, and the pipeline (ingest, scan, background, vetting) is
  what any later, better detector needs.
- **What a claim would need.** A candidate is only interesting if its false-alarm rate is small compared with the
  time searched (a week searched means hundreds of triggers at 1/hour), if it isn't explained by a glitch in one
  detector, isn't a hardware injection, sits in good-quality data in both detectors, isn't already catalogued or
  in GraceDB's public alerts, and its timing and amplitudes are consistent between detectors. Even then it would
  be a candidate for scrutiny, not a discovery.

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| S-D1 | HDF5 is converted by a small Python tool (`scripts/gwosc_bulk.py`, `h5py` in a project-local `.venv`, git-ignored) into raw little-endian files q reads with `1:`: float32 strain, int32 per-second data-quality and injection masks, and a JSON description. Everything after conversion is q. | The owner asked for HDF5 ingest; h5py is the standard reader. float32 keeps 7 significant digits, ample for whitening (the noise in band is far above float32 rounding of the strain). |
| S-D2 | Analysable time: seconds where both detectors have DATA, CBC_CAT1 and CBC_CAT2 set and no CBC hardware injection (NO_CBC_HW_INJ set). | The vetoes the CBC searches themselves use; hardware injections are LIGO's own test signals. |
| S-D3 | Whitening and scanning reuse the noise library's method (64 s chunks, own PSD, middle 52 s), so scanned data look like training data. | Consistency with training. |
| S-D4 | Background by time slides over the searched data itself, enough slides for the false-alarm rates being quoted. | As G6.1, in the same conditions as the search. |

## 3. Stages

### S1 — Bulk data ingest

- [x] **S1.1 Converter.** `scripts/gwosc_bulk.py`: list a run's 4 kHz files for detectors and a GPS range through
  GWOSC's API (v2 `strain-files`), download (resuming), convert to `<base>.f32`, `.dq`, `.inj`, `.json` under
  `data/gw/bulk/<run>/`, optionally removing the HDF5. *Accept:* a fixture HDF5 in GWOSC's layout converts exactly;
  a real O4a file converts and loads.
- [x] **S1.2 q loader.** `.qs.gw.bulk` (one converted file) and `.qs.gw.bulkIndex` (what's converted). *Accept:*
  tests on fixtures written by q; real file loads with matching GPS, rate and NaN count.
- [x] **S1.3 Analysable segments.** `.qs.gw.analysable[run;gpsStart;gpsEnd]`: per second, both detectors pass S-D2;
  contiguous segments of at least 64 s. *Accept:* tests on fixture masks; a real day's analysable time reported.

### S2 — Scan

- [x] **S2.1** Whiten analysable segments in chunks (S-D3) and score sliding 1 s windows; keep triggers (local
  maxima at least 1 s apart) above a low threshold, with times. (`.qs.gw.segmentStreams`, `.qs.gw.searchStreams`,
  `bench/gw_bulk_scan.q`; triggers rank by the logit, `.qs.predictRaw`, and carry each detector's peak amplitude.)
- [x] **S2.2** Recover the catalogued events in the scanned stretch (sanity check).

### G-B — Glitch tools (the false-alarm problem, both for O3 and here)

- [x] **Logit ranking**: `.qs.predictRaw` returns the output layer's pre-activation, so scores that round to 1
  still rank.
- [x] **Amplitude veto**: triggers where either detector's whitened amplitude in the window exceeds 15 (a BBH at
  network SNR 35 peaks near 6), applied alike to zero lag and background (`-veto` in `bench/gw_bulk_scan.q`).
- [x] **Hard negatives**: `.qs.gw.mine` (real glitches by whitened amplitude in a noise library) and
  `.qs.gw.mineScores` (noise windows an earlier model scores high); `.qs.gw.batch` with `mined` puts a share
  `real` of the glitchy windows on them; `bench/gw_cnn.q -mine 6 -hard <model> -patience 8`.
- [x] **O4a training noise**: `.qs.gw.bulkNoise` builds a noise library from a run's bulk data (analysable
  segments, away from catalogued events), from days that aren't searched.
- [x] Retrain on O4a noise with hard negatives; compare background and GW231226's rank with gw_cnn_1.
- [x] **Exclusive background**: `.qs.gw.far` drops background triggers where a catalogued event sits in either
  detector's (slid) window.
- [x] **Single-detector chirps as negatives**: `.qs.gw.batch` `single` (`bench/gw_cnn.q -single`).
- [ ] Retrain with single-detector negatives (gw_cnn_4) and compare.

### S3 — Background and significance

- [ ] **S3.1** Time slides over the same segments; false-alarm rate per trigger score.
- [ ] **S3.2** Candidate list: triggers ranked by false-alarm rate, with the catalogue cross-match.

### S4 — Vetting

- [ ] **S4.1** For each top candidate: single-detector scores (is one detector doing all the work?), H1-L1 time
  difference within the light travel time, data-quality flags around it, nearby loud glitches, a matched-filter
  follow-up with a small template bank (SNR, best masses, consistency between detectors).
- [ ] **S4.2** Check against GraceDB public alerts (including retracted) and the latest catalogues.
- [ ] **S4.3** Write-up: searched time, background, known events recovered, candidates and why each is or isn't
  interesting.

## Progress log

- 2026-10-27: Plan written. Checked GWOSC: O4a and O4b bulk strain are public as HDF5/GWF only (no text), about
  30 MB per 4096 s 4 kHz file; the first O4a H1 file is 77% NaN (detector down) and carries `quality/simple/DQmask`
  (DATA, CBC_CAT1-3, BURST, STOCH, CW bits) and `quality/injections/Injmask` (NO_CBC_HW_INJ and others).
  `.venv` with h5py 3.16 and numpy 2.5 created (git-ignored).
- 2026-10-27: S1.1 and S1.2 done. `scripts/gwosc_bulk.py` (`fetch`, `convert`): converts a GWOSC HDF5 file into
  `<det>-<gps>-<duration>.f32/.dq/.inj/.json` in 0.4 s (67 MB float32 strain per 4096 s file); on the real O4a
  file the NaN pattern is identical and the largest float32 error is 3e-8 of the strain's range. **The API ignores
  GPS ranges and serves 20 files per page (391 pages per detector for O4a)**, so `fetch` builds URLs directly from
  the archive's pattern (`<run>_4KHZ_R1/<gps rounded down to 2^20>/<site>-<det>_GWOSC_<run>_4KHZ_R1-<gps>-4096.hdf5`,
  files on multiples of 4096 s, 404 meaning no data) and keeps the API listing, cached per run and detector, behind
  `--use-api`. q side in `gw.q`: `.qs.gw.bulk` (0.22 s per file), `.qs.gw.bulkIndex`, `.qs.gw.bit`, `.qs.gw.usable`
  (DATA, CBC_CAT1, CBC_CAT2, NO_CBC_HW_INJ), `.qs.gw.segments`, `.qs.gw.analysable[run;start;stop;minimum]`. Tests
  `tests/test_bulk.q`: segment edges, mask bits, a fixture HDF5 in GWOSC's layout (`tests/fixtures/make_gwosc_hdf5.py`)
  converted and read back exactly with H1/L1 vetoes intersected, the real file (skips without data or `.venv`).
  The first real O4a H1 file has 922 usable seconds of 4096. Fetching one day around GW231226_101520 (SNR 34.7)
  for S1.3 and S2.2.
- 2026-10-28: S1.3, S2.1, S2.2. The hour around GW231226 (GPS 1387618304-1387622400) has 3897 analysable s in one
  segment. `bench/gw_bulk_scan.q` with gw_cnn_1 (step 0.25 s, 10 slides of 7 s; 7 min on a machine shared with 17
  other jobs): 1668 zero-lag triggers in 1.07 h, 16548 background triggers in 10.7 h. **GW231226 is recovered**
  (implied merger 1387620938.25 against the catalogue's .3, logit 6.04) **but ranks second and has a false-alarm
  rate of 1.7/hour.** The top trigger (logit 6.29) is a 118-sigma glitch in H1 alone (L1's peak 3.8), which the
  amplitude veto removes; the next ones look Gaussian (peaks 3-4 sigma in both) yet score logits of 4-5, so the
  O1-O3-trained CNN is overconfident on O4a noise generally, not just on glitches. Hence the O4a noise library.
  Real glitches above 6 sigma in the O1-O3 training noise: 42 in 6.3 h; with the windows gw_cnn_1 scores above
  0.5, 372 hard negatives in the training noise and 128 in validation. A retrain with these (`-mine 6 -hard`, early stopping)
  is running.
- 2026-10-28: Retrain `gw_cnn_2` (`bench/gw_cnn.q -mine 6 -hard models/gw_cnn_1.qsm -epochs 60 -patience 8`,
  43 min on the shared machine): on its validation batch (half the glitchy windows real) it finds 89% of SNR 5-10
  signals at 1% false alarms (gw_cnn_1: 63% on the simulated-glitch batch), 98% at 10-15, all above; false alarms
  on real glitches 3.2%. **On the GW231226 hour GW231226's logit goes from 6.0 to 21.8, louder than all 10.7 h of
  background (false-alarm rate under 0.09/hour)**; the loudest noise trigger stays near 4.7 for both models.
  The fetched day (1387577000-1387663400; one L1 file absent) has 20.7 h analysable in 5 segments, the longest
  9.7 h, so `.qs.gw.searchStreams` now scores windows in blocks of `.qs.gw.searchBlock` (8192).
- 2026-10-28: `gw_cnn_3`: as gw_cnn_2 plus the first 80% of 12 h of O4a noise before the searched day
  (`.qs.gw.bulkNoise`, 1387534000-1387577000: 256 chunk pairs, 3.7 h), hard negatives mined with gw_cnn_2; early
  stopping at epoch 51 (68 min). Validation on the last 20% of that O4a noise: 76% of SNR 5-10 found at 1% false
  alarms, 95% at 10-15; real-glitch false alarms 4.8%. On the GW231226 hour, loudest unvetoed logits:

  | model | GW231226 | loudest zero-lag noise | background median |
  |---|---|---|---|
  | gw_cnn_1 | 6.0 | 5.1 | -2.1 |
  | gw_cnn_2 | 21.8 | 4.7 | -4.2 |
  | gw_cnn_3 | 16.2 | 2.7 | -3.3 |

  **The loudest background triggers (logits 12-16) are all GW231226 itself in H1, slid against L1 noise**: the
  network fires on a chirp in one detector alone. Two fixes: the false-alarm rate now uses an exclusive
  background (`.qs.gw.far`; the inclusive one put GW231226 at 0.09/h with gw_cnn_3), and training can include
  single-detector chirps as negatives (H1 and L1 are nearly aligned, so a real signal is in both).
