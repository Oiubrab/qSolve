# Open-data search plan: looking for gravitational-wave events not in the catalogues

Follow-on to [gw_detection_plan.md](gw_detection_plan.md), whose CNN finds GW150914 and 12 of 35 held-out O3
detections at under 1 false alarm per hour. The owner asked: can we run it over recent public LIGO data and spot
events that haven't been reported?

Read the "q gotchas" in [completed/improvemet_plan.md](../completed/improvemet_plan.md) and the GW plan's log first.

**How to use this file:** as the other plans. Tick an item only when its check passes and `q tests/run.q` passes;
add a dated line to the Progress log; commit each finished stage before starting the next (the owner's rule).

**Current status (2026-10-27):** S1.1 and S1.2 done (HDF5 -> q works); S1.3 (analysable time on a real day) next.

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
- [ ] **S1.3 Analysable segments.** `.qs.gw.analysable[run;gpsStart;gpsEnd]`: per second, both detectors pass S-D2;
  contiguous segments of at least 64 s. *Accept:* tests on fixture masks; a real day's analysable time reported.

### S2 — Scan

- [ ] **S2.1** Whiten analysable segments in chunks (S-D3) and score sliding 1 s windows; keep triggers (local
  maxima at least 1 s apart) above a low threshold, with times.
- [ ] **S2.2** Recover the catalogued events in the scanned stretch (sanity check).

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
