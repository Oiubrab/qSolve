# Psychohistory plan: learning how the world normally moves, and what surprises it

A new project, `projects/geopolitics/`. It's inspired by Asimov's *Foundation*. Hari Seldon couldn't predict any
one person, but he could predict the statistics of very large populations. Here the units are countries, not
people, and the aim is modest. The plan is to learn the ordinary year-to-year dynamics of the world's economic,
political and military state, then look at where reality departs from them. Do some departures come before coups,
wars or changes of regime?

The owner, 2026-11-03: this is for fun and for generalised patterns that might mean something, not for a
leaderboard score. So the headline outputs are:
- a world model we can run forward and look at;
- a map of countries;
- any lead-lag pattern that survives honest checks.

Forecast skill is still measured honestly, against strong baselines (section 1). Otherwise "patterns" are just
noise.

Read the "q gotchas" in [improvemet_plan.md](../../../../plans/completed/improvemet_plan.md) first. Also read the
markets project's [fx_trends_plan.md](../../../markets/plans/completed/fx_trends_plan.md). It is the closest past
work: a panel of slow-moving series, where the persistence baseline won.

**How to use this file:** the same as the other plans.
- Tick an item only when its check passes and `q tests/run.q` passes.
- Add a dated line to the Progress log.
- Commit each finished stage to main.
- Don't push without asking.
- Tests stay offline: anything that needs downloaded data skips itself when the data is missing.

**Current status (2026-11-03):** design only. The open questions P-Q1 to P-Q5 below need the owner. Move this file
to `inProgress/` when P0 starts.

---

## 1. What has been done before, and what's left that's fun

Forecasting political instability is an established field:
- **Political Instability Task Force (Goldstone et al. 2010).** Four variables predicted instability onsets two
  years ahead with about 80% accuracy:
  - regime type;
  - infant mortality;
  - conflict in neighbouring countries;
  - state-led discrimination.

  The main lesson is that a handful of slow variables carries most of what's predictable.
- **ViEWS (Uppsala, PRIO).** Monthly conflict forecasts for every country, and for grid cells in Africa and the
  Middle East. It's the state of the art for conflict-fatality prediction and publishes its scores.
- **ICEWS (DARPA, later the US government).** Event-data forecasting of crises.
- **Coups.** A literature of its own, built on the Powell & Thyne data.
- **Cliodynamics (Turchin).** Long secular cycles of state strength and instability. The Seshat databank supports
  it.
- **World models.** World3 (*The Limits to Growth*, 1972) is the classic. It simulates the whole world as a
  handful of aggregate stocks, not as countries that interact.

So "predict coups" on its own is not new, and we will not beat ViEWS. What's less explored, and more in Seldon's
spirit, is the combination in this plan:
- **A learned world model.** One network steps every country's state forward a year together, with each country
  affected by its neighbours and trading partners.
- **Run it forward** for decades, and compare the distribution of simulated worlds with what actually happened.
- **Surprises as signals.** Study the residuals, meaning what the world model didn't expect, as possible
  precursors of events. Check the method by planting known patterns and recovering them, as with GW injections.
- **A political map** from UN votes, joined to all of the above.

**Things to be careful about.** They decide whether anything found means anything.
1. **Persistence.** Most of these variables are almost the same as last year. Every model must beat "no change"
   and a linear model on the same inputs. Otherwise the network has only learned persistence, as in FX.
2. **Rare events.** There have been nearly 500 coup attempts since 1950, and a few hundred conflict onsets since
   1946.
   - Most country-years have neither.
   - So test only on held-out later decades, never on a random split.
   - Report calibrated probabilities, not just accuracy. "Never" is about 97% accurate.
3. **Hindsight in the data.** Some sources are written or revised after the fact.
   - V-Dem's experts code past years knowing what came next.
   - GDP is revised for years.
   - A year's data is mostly published in the middle of the next year.
   - Decision P-D3 handles this, and P1.4 has a test for it.
4. **The world changes.** The Cold War, its end, and the post-2008 period may behave differently. Errors are
   reported per decade, and a pattern that holds in only one era is reported as such.
5. **Many tries.** Many variables, lags and event types mean that something will look significant by chance.
   Every scan counts its trials and uses permutation tests (P4).
6. **Reflexivity.** Governments react to the same signals we measure, so a pattern can vanish because it was
   noticed. That's the *Foundation* problem, and there's nothing to do about it except note it.

This is a toy, and descriptive. It uses only country-level aggregates and no data on individuals. Nothing from
it should inform real decisions.

**What a good outcome looks like** (any one is enough):
- a world model that clearly beats persistence on some variables, and a forward simulation whose spread of
  outcomes contains what happened;
- a surprise pattern before an event type that survives the permutation test and the planted-pattern check, and
  holds in more than one decade;
- a UN-vote map whose axes match published ideal points, with movement on it that comes before alignment
  changes.

A clear, well-measured null result is a valid outcome, as in FX.

## 2. Data

| Source | What | Years | Notes |
|---|---|---|---|
| World Bank WDI (API) | GDP and its growth, inflation, trade share, debt, population, urbanisation, infant mortality, military spending share | 1960– | CC BY 4.0; CSV through the API. Past releases are kept in the "WDI Database Archives", which lets us use the numbers as first published (P-D3). |
| SIPRI Military Expenditure | military spending, real and as a share of GDP | 1949– | xlsx, free for non-commercial use |
| SIPRI Arms Transfers | trend-indicator values between pairs of countries | 1950– | for the trade and influence links |
| V-Dem | democracy indices; Regimes of the World category | 1789– | CC BY-SA; about 4,000 columns, of which we keep a few dozen. Coded with hindsight (point 3 in section 1). |
| UCDP/PRIO Armed Conflict | conflict-years, intensity, onsets | 1946– | CC BY 4.0 |
| Powell & Thyne | coup attempts, successful and failed | 1950– | plain-text file |
| Correlates of War | direct contiguity (neighbours); national material capabilities | 1816–2016 | the neighbour links in P3 |
| CEPII BACI or IMF trade | trade between pairs of countries | 1995– (BACI); 1948– (IMF) | the trading-partner links in P3; P-Q3 chooses |
| Voeten et al., UNGA votes | each country's vote on each recorded resolution, with published ideal points | 1946– | Harvard Dataverse. Several thousand roll calls; P6. |
| Global Sanctions Data Base | sanctions cases and their type | 1950– | free on request; an event type for P2 and P6 |
| Laeven & Valencia | banking, currency and sovereign-debt crises | 1970– | an economic event type |
| *Later:* GDELT, ICEWS/POLECAT | coded news events: who did what to whom, and when | 1979– (GDELT) | P8 only |

**The country list is the hard part.** Countries split, merge and are renamed, for example the USSR, Yugoslavia,
Czechoslovakia, the two Germanys, Sudan and South Sudan, and Yemen. The sources use different code systems:
Correlates of War, Gleditsch & Ward, and ISO3.

The master list is Gleditsch & Ward's state list, which says when each state exists. Every other source joins to
it through one hand-checked crosswalk file committed to the repo. Successor states start their own series. We
never splice a predecessor's history onto a successor's.

## 3. Decisions

| # | Decision | Why |
|---|---|---|
| P-D1 | **Countries by year** as the main table: one row per country per year, about 190 countries and 1960–2024, so roughly 11,000 rows. | Most sources are annual. It's small, fast in q, and enough for an MLP with a few dozen inputs. |
| P-D2 | **Two models, not one.** A *state model* predicts next year's continuous state vector (mse, linear outputs). An *event model* predicts next year's event probabilities, one sigmoid output per event type (bce). | The library trains one loss per model. The state model drives the forward simulation, and the event model turns states into risks. |
| P-D3 | **Only what was known at the time.** The target is year t+1. The features use data through year t, and the strict variant uses data through t−1. Where first releases exist (WDI archives), the test decades are also scored with first-release numbers. | Hindsight leakage is this field's look-ahead bias. The strict variant measures how much the publication lag costs. |
| P-D4 | **Splits by time, with rolling origins:** train to 1989 and test 1990–99; train to 1999 and test 2000–09; train to 2009 and test 2010–latest. Validation is the last 5 years of each training period. | Rare events and a changing world (section 1, points 2 and 4). |
| P-D5 | **Baselines come first and are always reported:** no change; each country's own mean; a pooled AR(1); a linear or logistic model on the same inputs; for events, "had one in the last 10 years", and a PITF-style four-variable logistic model. | The network has to beat all of them before anything is called a pattern. |
| P-D6 | **Missing values:** each input with gaps gets a 0/1 "present" flag next to it, and the value itself is filled with the country's last known value, or the year's world median if there is none. Missing *targets* are left out of that variable's score. If the state model needs masked targets in training, that's a library change, filed in the root `plans/`. | Gaps aren't random: weak states report less. The flag lets the model use that. |
| P-D7 | **Event scores:** Brier score, log loss, calibration curves, and "top-k": of the 10 highest-risk countries each year, how many had the event. | Accuracy is meaningless at a 3% base rate. Top-k is the readable one. |
| P-D8 | **Layout** as the other projects: `projects/geopolitics/src/geo.q` (namespace `.qs.geo`, home `.qs.geo.home`), `experiments/`, `scripts/fetch_geo.sh`, `tests/`, and git-ignored `data/`, `results/`, `models/`. | Same conventions as GW and markets; `tests/run.q` picks it up unchanged. |

## 4. Open questions for the owner

1. **P-Q1, which events.** Recommended for the first build: coup attempts, conflict onset (UCDP, 25 or more
   battle deaths), and changes of regime type (V-Dem Regimes of the World). Others to add once those work:
   sanctions imposed, economic crises, sharp rises in military spending.
2. **P-Q2, helper Python for file conversion.** SIPRI publishes xlsx, and V-Dem's CSV is very wide.
   - Recommended: use `openpyxl` in the existing `.venv` to convert, as `h5py` does for GW. The project itself
     stays pure q.
   - The alternative is to convert by hand once and commit the instructions.
3. **P-Q3, trade data.** BACI is cleaner but only starts in 1995. IMF trade data goes back to 1948 but has more
   gaps. Recommended: IMF, so the trade links cover the Cold War decades too.
4. **P-Q4, licences.**
   - SIPRI and Seshat are for non-commercial use, and the GSDB is free on request.
   - The data stays git-ignored and is fetched by a script, as now.
   - Is the repository going to stay non-commercial? If it isn't, we drop SIPRI and use WDI's military-spending
     series, which comes from SIPRI under WDI's terms.
5. **P-Q5, how far to go.** P0–P5 is the core. P6, the UN map, is cheap and recommended. P7, the dashboard tab, and
   P8, event data, are optional.

## 5. Stages

### P0 — Sources and the country list

- [ ] **P0.1 Check every source:** URL, licence, update date, file format, and whether it can be downloaded by
  script or needs a form. Record it in the project README's data section.
  *Accept:* the table in section 2 confirmed or corrected, with a download method for each source.
- [ ] **P0.2 Crosswalk.** Build `projects/geopolitics/src/countries.csv`, committed to the repo, mapping Gleditsch
  & Ward codes and their dates to Correlates of War, ISO3 and each source's own names.
  *Accept:* every country-year in every source either joins or is listed in an "unmatched" report that someone
  has read. Tests cover the USSR→Russia, Yugoslavia→Serbia and Sudan/South Sudan cases.
- [ ] **P0.3 `scripts/fetch_geo.sh`.** It downloads what it can into `data/raw/`, prints instructions for anything
  behind a form, and checks file hashes. It supports `--dry-run`, like `fetch_gw.sh`.

### P1 — The panel

- [ ] **P1.1 Loaders in `src/geo.q`**, one per source, each returning a table keyed by country and year with the
  country code from the crosswalk. Tests use small fixture files in `tests/fixtures/`.
- [ ] **P1.2 State vector.** Choose about 20 continuous variables, each with a transform (log, growth rate or
  share) and a robust scaling fitted on the training years only. Proposed:
  - real GDP per head and its growth;
  - inflation (signed log);
  - trade share;
  - debt;
  - military spending share, and its growth;
  - population growth;
  - urban share;
  - infant mortality;
  - V-Dem electoral and liberal democracy indices;
  - years since the last regime change;
  - battle deaths (log);
  - material capability share.

  Record the list and the reason for each choice in the README.
- [ ] **P1.3 Neighbour and partner features.** For each state variable, add the average over land neighbours and
  the trade-weighted average over trading partners, plus counts of neighbours in conflict and of recent coups in
  the region.
  - This is where contagion can enter.
  - Partner weights come from year t trade, never t+1.
- [ ] **P1.4 The hindsight test.** Build the panel, then plant a large spike in one country's year t+1 data and
  rebuild. Every feature at year t or earlier must be unchanged, for every country.
  *Accept:* the test passes, and also fails when a deliberate one-year leak is added to a copy of a loader.
- [ ] **P1.5 Event table.** For each event type from P-Q1, a 0/1 flag per country-year, with the onset rules
  written down (for example, a conflict "onset" after two or more inactive years).
  *Accept:* counts per decade printed and checked against the counts the sources publish.

### P2 — Baselines and the evaluation harness

- [ ] **P2.1 `experiments/geo_baselines.q`.** All P-D5 baselines on all three rolling splits. State variables are
  scored per variable as MSE relative to "no change". Events are scored as in P-D7.
  *Accept:* one table per split in the README. This is the bar for everything after.
- [ ] **P2.2 A PITF replica.** The four-variable logistic model on our own definitions, as a sanity check that our
  data behaves as the literature says.
  *Accept:* top-k and Brier scores in the same range as published, with the differences explained.

### P3 — The world model

- [ ] **P3.1 State model.** An MLP taking the last 3 years of own state, the neighbour and partner features, and
  the "present" flags. It outputs the next year's change in the state vector, so persistence is the zero output.
  Train with the library's grid search over width, depth, dropout and L2, using the validation years.
  *Accept:* per-variable skill against "no change" and against the linear model, on each test decade. We report
  which variables the network beats the linear model on, and which it doesn't.
- [ ] **P3.2 Event model.** Inputs are the state, the neighbour features, and the state model's predicted change.
  Outputs are one sigmoid per event type.
  *Accept:* P-D7 scores against the P2 baselines, with calibration plots in `results/`.
- [ ] **P3.3 What it uses.** For each input group, shuffle it on the test years and see how much the score drops,
  for both models. Check whether neighbours and partners matter beyond a country's own history.
  *Accept:* a ranked table. This is the first readable "pattern" result.

### P4 — Surprises

- [ ] **P4.1 Residuals.** For each country-year in the test decades, take the state model's error per variable,
  scaled by its typical size, giving a z-score per variable.
  *Accept:* the residual table in `results/`. Its distributions are checked for heavy tails, and the extremes are
  listed by name. They should read like a list of known crises, which is itself a sanity check.
- [ ] **P4.2 Event studies.** For each event type, average each variable's surprise over the 5 years before the
  event, compared with matched country-years without one (same region, decade and regime type).
  - Significance comes from a permutation test that circularly shifts each country's event years 1,000 times.
  - Count the trials: variables × lags × event types.

  *Accept:* a table of the surprises that survive the trials count, with their sizes, in each decade separately.
- [ ] **P4.3 Planted patterns (the injection test).** In a copy of the real panel:
  - Plant a synthetic precursor, for example raise the coup probability after a 2-year inflation surprise, at a
    few strengths.
  - Run P4.2 unchanged on each copy.

  *Accept:* a recovery curve: the planted strength the method detects half the time. With nothing planted, it
  "finds" things at the rate the trials count predicts. This shows what size of real pattern P4.2 could have
  seen.
- [ ] **P4.4 Do surprises help?** Add the last 1–3 years of residuals to the event model's inputs, retrain on the
  same splits, and compare with P3.2.
  *Accept:* the change in out-of-sample Brier score and top-k, with uncertainty from training over several seeds.
  This is the honest test of whether surprises carry information.

### P5 — Running the world forward

- [ ] **P5.1 The simulator.** Start every country at its real 1990 state.
  - Step all countries together each year, so neighbour and partner features come from simulated states.
  - Add noise by resampling the model's own training residuals.
  - Run 100 worlds to the latest year.
  - Use the state model trained to 1989 only.
- [ ] **P5.2 Compare with history.**
  - For each world-level series, check where the real path falls within the simulated spread: share of democracies,
    countries in conflict, median growth, military spending share.
  - Also take the rank correlation between simulated and real final states, by country.
  - Feed the simulated states to the event model to get simulated coup and conflict rates by decade.

  *Accept:* a figure per series, real against simulated spread, and a short honest reading of each. Did 1990's
  model see the 2000s growth of China and India, the democratic recession of the 2010s, the coup wave in the
  Sahel in the 2020s? Most likely not, and that's an answer too.
- [ ] **P5.3 Behaviour of the model.** Run from today to 2100, labelled clearly as a toy, to see what the dynamics
  do:
  - Does it explode, settle into a fixed point, or cycle?
  - If it cycles, at what period? Compare with Turchin's secular cycles.

  This is the closest thing in the plan to a "Seldon plan", and purely for curiosity.

### P6 — A map from UN votes

- [ ] **P6.1 Load the Voeten et al. votes.** Each vote becomes yes = 1, abstain = 0.5, no = 0, with absence
  flagged separately.
- [ ] **P6.2 Learn the map.** Train an autoencoder: dense layers with a bottleneck of 2–3 linear units, on each
  country's votes in one session. The bottleneck gives every country-year a position.
  *Accept:* the first axis correlates strongly with the published ideal points, which is the known-answer check.
  We report what the second and third axes seem to mean.
- [ ] **P6.3 Movement and decisions.** Check whether a country's movement on the map comes before:
  - joining or leaving alliances (ATOP or Correlates of War);
  - being sanctioned (GSDB);
  - changes in arms suppliers (SIPRI transfers).

  Use the event-study method from P4.2, including its planted-pattern check. Then add map position and movement
  to the P3 state and see whether the scores change.

### P7 — Optional: a World tab in the dashboard

- [ ] **P7.1** Pick a country to see its state over time, the model's predictions, its surprises and its path on
  the vote map. Pick a year to see the whole map, and the simulated spread from P5. Same conventions as the Waves
  tab, with a UI check in `tests/ui/check.js`.

### P8 — Later: who does what to whom

- [ ] **P8.1 Event data (GDELT, or ICEWS/POLECAT)**, by pair of countries and by month.
  - Predict the next month's cooperation–conflict balance (Goldstein scale) between a pair from the recent event
    history of the whole graph.
  - Look for an "escalation grammar": common sequences of event types before violent ones.

  This is a large data-engineering job, hundreds of millions of rows, so it only starts if P0–P5 have been fun.
  It gets its own plan when it does.

## Progress log

- 2026-11-03: Plan written from a discussion with the owner. They wanted a less scientific, more left-field
  project, inspired by *Foundation*. Not started.
