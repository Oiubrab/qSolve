# Market cycles plan: looking for hidden cycles and lags in prices

Follow-up to [completed/fx_trends_plan.md](../completed/fx_trends_plan.md), which found no out-of-sample edge from
slow trends (momentum, distance from the long-run mean, volatility, month of year) in 27 years of daily ECB rates.
Its main limit was sample size: about 330 independent months per pair.

The owner's challenge to the "too few business cycles" argument: **cycles don't have to be macro.** A product that
takes 60 days from raw material to sale, or a coffee shop that sees a jump every Thursday because that's payday
round the corner, repeats hundreds or thousands of times. Can a network find cycles like that in market data, if
they are there?

Read the "q gotchas" section of [completed/improvemet_plan.md](../../../../plans/completed/improvemet_plan.md) and the FX trends
plan's log before writing q here.

**Status (2026-11-02): future, not started.** Written from a discussion with the owner. Start after the GW plans in
`inProgress/` are finished or paused, and move this file to `inProgress/` when it starts. Decide the data sources and
their terms (Stage C0) first.

**How to use this file:** as the other plans. Tick an item (`- [x]`) only when its acceptance check passes *and*
`q tests/run.q` passes; add a dated line to the [Progress log](#progress-log) for each tick; commit each finished
stage to `main` before starting the next; don't push without asking. Tests stay offline: anything needing
downloaded data skips itself when the data is missing.

---

## 1. What we are testing, and why the FX trends plan doesn't settle it

The FX models had trailing returns, distances from long-run means, volatility and month of year as inputs. With
those inputs a Thursday effect, a month-end effect or a 60-day lag between two series **could not be seen at all.**
A network only finds combinations of the inputs it is given; it does not discover a periodicity in a feature
set that doesn't encode the phase. Most of this plan is about building the right inputs.

**Fundamentals vs prices.** The coffee shop's Thursday jump is large and obvious in its sales. Its share price will
not jump every Thursday: once a pattern is predictable, traders buy ahead of it until it disappears from the price.
The cycles that survive in prices are either **not yet noticed** or driven by **forced flows** (someone must trade
on a schedule, whatever the price). Documented examples, all close to the owner's idea:

- **Payment cycles.** Etula, Rinne, Suominen & Vaittinen (2020, "Dash for cash"): a predictable fall and recovery
  in stock returns around month-end, tied to institutions selling to meet monthly payments. Older turn-of-month
  work (Lakonishok & Smidt 1988; McConnell & Xu 2008) finds much of the equity premium earned in a few days around
  the month's turn.
- **Treasury auction cycle.** Lou, Yan & Zhang (2013): yields rise before scheduled auctions and fall after.
- **Fed meeting cycle.** Lucca & Moench (2015): stocks drift up in the day before scheduled FOMC announcements.
  Cieslak, Morse & Vissing-Jorgensen (2019): returns concentrate in alternate weeks of the roughly 6-week FOMC
  calendar.
- **Supply-chain lags.** Cohen & Frazzini (2008): news in a customer company's stock reaches its suppliers' stocks
  with a lag. Menzly & Ozbas (2010): the same between industries linked by input-output flows.
- **FX month-end.** Hedging flows around the London 4pm fix at month-end (Melvin & Prins 2015).

**Three kinds of cycle, each tested differently:**

1. **Calendar-anchored** (payday, weekday, month-end, auctions, FOMC, option expiry). The phase is tied to known
   dates, so it is the easiest to test and has the most repeats.
2. **Lead-lag between series** (supply chains). A 60-day pipeline is not a fixed-date cycle; it starts whenever
   orders were placed. It shows up as one series leading another by some lag, not as a repeating pattern in either.
   Expect the price lag to be much shorter than the physical one: prices react to news about the flow (days to
   weeks), not to the goods arriving.
3. **Free-running periods of unknown length.** Scan every period and see whether anything stands out. The least
   likely to exist in prices, but cheap to check, and the closest to the GW work (a weak repeating pattern in
   noise, with a look-elsewhere problem).

**What success looks like.** The FX lesson applies: a large edge means a bug. Two numbers to keep in mind:

- *Detectable:* a calendar effect of 0.1% a day on 4 days around each month-end, against 1% daily volatility, is a
  signal-to-noise of 0.1 per day. Over ~1,200 month-ends (1926 onwards, ~4,800 days) that is t ≈ 7: easy to see
  in sample.
- *Small skill:* the same effect reduces next-day MSE by about 0.1² × 4/21 ≈ 0.2%. **A real calendar effect has
  tiny MSE skill**, so skill alone is the wrong headline (C-D5).

A good outcome is a calendar or lead-lag effect that is significant after the trials count, holds after its
publication date, and is consistent across industries or pairs. A clear, well-measured null is equally valid.
Published effects typically shrink after publication (McLean & Pontiff 2016 put it at about a third).

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| C-D1 | **Daily** data throughout. Calendar targets are the next-day (h=1) return and the return over the event window (e.g. last trading day to day 3 of the month). Lead-lag targets are h = 1, 5, 20. | Calendar effects live at the scale of days; monthly data averages them away. |
| C-D2 | **Calendar features** built from the trading calendar in the data itself: weekday (one-hot), trading day of month counted from both ends (−5..+5 around the turn, clipped), days since and until the last/next scheduled FOMC announcement, FOMC-cycle week, option-expiry week (third Friday), and for rate-sensitive series days to the next Treasury auction of each tenor. | Gives the network the phase of each candidate cycle directly, as month-of-year sin/cos did for season. |
| C-D3 | **Only dates known in advance.** Scheduled FOMC meetings and announced auction dates, never unscheduled moves (an emergency cut is not known the day before). Event dates come from a hand-checked file committed to the repo, with its source noted. | The calendar version of look-ahead. |
| C-D4 | **Lead-lag features** are the other series' returns summed over lag windows (1-5, 6-20, 21-60, 61-120 days back), not hundreds of raw lags. | The FX lesson (F-D2): summaries, not 250 noisy inputs. |
| C-D5 | **Headline metrics:** for calendar effects, the mean return in the event window minus the mean on other days, with a block-bootstrap interval; for forecasts, skill vs zero change *and* vs training mean, information coefficient, and count of positive years (as F-D6). | Real calendar effects give ~0.2% MSE skill (Section 1); the conditional mean is where they show. |
| C-D6 | **Background by calendar slides.** The null for any calendar effect is the same statistic with the calendar shifted relative to the returns by a whole number of months (or weeks), many times. For lead-lag, shift one series against the other by more than the longest lag tested. | The time-slide idea from the GW search: keeps the returns' own structure (volatility clustering, fat tails) and breaks only the alignment. |
| C-D7 | **Trials count, with a look-elsewhere correction.** Count every calendar feature × series × horizon, every industry pair × lag window, every period scanned. For scans, compare the *loudest* peak against the loudest in each slide, not each peak against its own null. | Scanning 49 industries × 49 × 4 lags is ~9,600 tests; some will look strong by chance. |
| C-D8 | **Publication split.** For each documented effect, score separately before and after the paper's data end / publication date. | Tests whether it was real and whether it survived being known. |
| C-D9 | **Purged walk-forward by year** as F-D4, with the FX code (`.qs.fx.split`, `walkRows`, `summary`) reused or generalised, not copied. | Same leakage risk; the code is tested. |
| C-D10 | **Fix the experiment list before running each stage**, written into this file. Industry pairs for C3 are chosen first from economic links (input-output suppliers and customers), then the all-pairs scan is run separately with its own trials count. | Researcher degrees of freedom (FX risks section). |
| C-D11 | New code in `src/cycles.q` (`.qs.cyc`: calendar features, event files, slides, lead-lag features, period scan), experiments in `bench/cycles/`, tests in `tests/test_cycles.q`. Data in `data/cycles/` (git-ignored); fetches in `scripts/fetch_data.sh`. FFT and periodograms come from `signal.q`. | Follows the existing layout. |

## 3. Stages

### Stage C0 — Data and sources

- [ ] **C0.1 Equity data.** Ken French's data library: daily 49 industry portfolios (value-weighted, from 1926) and
  daily market factors with the risk-free rate. Check and record the terms of use before adding a fetch.
  `.qs.cyc.french file` parses the CSV (header lines, the `-99.99` missing-value code, percent units).
  *Accept:* parser test on a hand-made file; on the real file, row counts and date range logged, no gaps in the
  trading calendar beyond known closures.
- [ ] **C0.2 Event dates.** Scheduled FOMC announcement dates (Federal Reserve historical materials; the pre-1994
  meetings had no same-day announcement, so record the announcement date, not the meeting date) and Treasury
  auction dates (TreasuryDirect / Fiscal Data). Committed as a small CSV with source and retrieval date.
  *Accept:* counts per year match the published schedule (8 scheduled FOMC meetings a year since 1981); spot-checks
  of a few dates by hand.
- [ ] **C0.3 Optional: retail demand data** for Stage C7, e.g. UCI "Online Retail" (check the licence).

### Stage C1 — Calendar features and controls on the FX data already in the repo (cheapest)

- [ ] **C1.1 Calendar features.** `.qs.cyc.calendar[dates;events]` per C-D2.
  *Accept:* tests on a hand-made calendar (a month with a holiday on its last weekday gets the right turn-of-month
  index; FOMC days-since/until are correct around a meeting; a perturbation test that no feature at t uses an event
  dated after t unless it was scheduled).
- [ ] **C1.2 Synthetic truth.** Plant a month-end effect (0.05%/day over 4 days in 0.5%/day noise, about FX scale)
  and a Thursday effect in random-walk data. The pipeline must find both, with the conditional-mean statistic and
  the slide background, and must find nothing with the effects removed.
  *Accept:* tests for both.
- [ ] **C1.3 Calendar slides.** `.qs.cyc.slides` per C-D6.
  *Accept:* on pure noise, the slide p-values are roughly uniform (a test on many simulated series).
- [ ] **C1.4 Real FX.** Weekday and turn-of-month effects in the six FX pairs, next-day return. Note the ECB rate is
  set at about 14:15 CET, before the London 4pm fix, so a fix-driven month-end flow may be only partly captured.
  *Accept:* results table in the log with the trials count.

### Stage C2 — Calendar-anchored cycles in equities

- [ ] **C2.1 Volatility positive control.** Weekday and FOMC-day patterns in *squared* returns (well established).
  *Accept:* clearly found; if not, stop and debug.
- [ ] **C2.2 Published effects as fixed rules.** Turn-of-month, pre-FOMC day (from 1994), FOMC even weeks, weekday,
  on the market and each industry, with the publication split (C-D8).
  *Accept:* results table: effect size, interval, slide p-value, before/after publication.
- [ ] **C2.3 Linear and MLP models** on the calendar features plus the FX plan's slow features, per industry and
  pooled across industries, next-day and event-window targets, against the C2.2 rules.
- [ ] **C2.4 Interactions.** Does the network beat the rules by combining features (e.g. month-end only in high
  volatility, FOMC drift only in rate-sensitive industries)? Compare MLP vs linear vs rules; ablate feature groups
  as F3.4.

### Stage C3 — Lead-lag between series (the supply-chain idea)

- [ ] **C3.1 Synthetic truth.** Two noisy series where one follows the other at a 60-day lag with realistic
  strength (and a variant at 5 days). The pipeline must find the lag and its size; with the link removed, nothing.
- [ ] **C3.2 Linked pairs.** A fixed list of supplier→customer industry pairs chosen from economic links before
  looking at returns (e.g. Oil→Trans, Steel→Autos, Chips→Hardw, Agric→Food, Mines→Steel), with C-D4 lag windows.
  Cross-correlation by lag window, slide background (C-D6).
- [ ] **C3.3 All-pairs scan.** Every ordered industry pair × lag window, with the look-elsewhere correction
  (C-D7). Report how many pairs beat the slide background and whether C3.2's linked pairs are over-represented.
- [ ] **C3.4 Pooled model.** One network forecasting each industry from all others' lag-window returns (C-D4), scored
  per industry, against the C3.2 pairs as rules.

### Stage C4 — Free-running periods

- [ ] **C4.1 Periodogram scan.** Welch periodograms (`.qs.sig.welch`) of returns and of squared returns, per
  series, periods 10 to 250 trading days. Background: the loudest peak in each of many block-shuffled series (blocks
  of 5 days, keeping short-range dependence but breaking any period ≥ 10).
  *Accept:* a planted 60-day cycle in synthetic data is found at the right period; on noise, the loudest peak
  matches the background.
- [ ] **C4.2 Split-half replication.** Any peak above background must reappear in the other half of the data at
  the same period *and phase*. A peak that moves or flips phase is noise.
- [ ] **C4.3 Out-of-sample check.** For any surviving period, sin/cos at that period as features in the walk-forward.

### Stage C5 — Is it real?

- [ ] **C5.1 Significance and trials** for every headline number (C-D5, C-D7), with the count next to the best
  result.
- [ ] **C5.2 Stability** by decade and by industry/pair; flag anything that depends on one period (1929-33,
  1987, 2008, 2020).
- [ ] **C5.3 Shuffled-target and leak controls** on the real features, as F4.3.
- [ ] **C5.4 Cost-aware check.** Calendar effects are a day or a few days long, so costs matter far more than at
  monthly horizons: trade only in event windows, with a cost per round trip, and ask whether anything survives.

### Stage C6 — Docs

- [ ] **C6.1 README** section next to the FX results, stated plainly, null or not.

### Stage C7 — Optional contrast: where cycles are strong

- [ ] **C7.1 Retail demand.** The same calendar features and pipeline on daily sales data (C0.3), where weekday,
  payday and seasonal cycles are large. Shows what the method finds when cycles have not been traded away, which
  makes the market result easier to read.

## 4. Risks and things to watch

- **Trials.** Calendar features × 49 industries × horizons × lags multiply quickly; the slide background and the
  trials count (C-D6, C-D7) are the main defence. Report the count next to every best result.
- **Look-ahead in event dates.** Using the date a meeting *happened* rather than when it was *scheduled*, or
  revised calendars. C-D3 and the perturbation test in C1.1.
- **Calendar quirks.** Holidays, half-days, the 1968 Wednesday closures and the 2001 and 2012 closures shift
  "trading day of month". Build the calendar from the data's own dates and test the edge cases.
- **Old data.** Pre-1960s industry returns are thinner (fewer firms per industry) and may have stale prices,
  which creates fake lead-lag (a stale series appears to follow a liquid one by a day). Check lag-1 results with
  and without the early decades.
- **Publication decay.** A known effect that faded is not a failure of the pipeline; report before and after
  separately.
- **Small skill, real effect.** Don't throw away a significant conditional-mean effect because its MSE skill is
  0.2% (Section 1).

## Progress log

- 2026-11-02: Plan written, not started. Open questions for the owner: whether to include Treasury auctions (needs
  a rates series too, e.g. FRED yields, to see the effect directly), and whether Stage C7 is wanted.
