# FX trend research plan

Follow-up to Phase 9 of [completed/improvemet_plan.md](../completed/improvemet_plan.md). That phase asked
whether the last few days of returns predict tomorrow's return, and the answer was no (skill below 0 in 16 of
17 walk-forward years). This plan asks a different question: **do slow patterns, over months and years, carry
information about the next weeks or months?**

Read the "q gotchas" section of the earlier plan before writing q here.

**Status (2026-10-25): complete**, moved to `plans/completed/`. Stages F1-F4 done; F5 not triggered (nothing to extend). Result: **no reliable
out-of-sample edge** from slow trends in 27 years of daily ECB rates. See the Progress log for the numbers and the
list of places where the work departed from this plan.

**How to use this file**

- Work through the stages in order; tick an item (`- [x]`) only when its acceptance check passes *and*
  `q tests/run.q` passes.
- Add a dated line to the [Progress log](#progress-log) for each tick, including anything surprising. The log is
  append-only. If a decision changes, edit it and say why in the log.
- Commit each finished stage to `main` before starting the next. Don't push without asking.
- Keep tests offline: anything needing downloaded data skips itself when the data is missing.

---

## 1. What we are testing, and why Phase 9 doesn't settle it

Phase 9 used 10 or 5 lagged *daily* returns and forecast the next day (or 5 days). That is the setting where
the signal-to-noise ratio is lowest: daily FX moves are about 0.5%, while any slow drift is perhaps a few
hundredths of a percent per day. A slow trend is invisible in a window of 10 days and swamped in a 1-day target.

The hypothesis here is that predictability, if it exists, sits in:

1. **Momentum / trend.** Past returns over 1–12 months have a weak positive relationship with the next
   1–3 months (time-series momentum is a documented, if modest, FX effect).
2. **Mean reversion to long-run value.** Rates far from their multi-year average drift back over years
   (purchasing-power-parity style). Slow, and weak at horizons under a year.
3. **Seasonality.** Month-of-year or turn-of-month effects, which are the least likely to be real.
4. **Volatility regimes.** Not trend, but the most reliably predictable quantity in finance. Worth including as
   a positive control: if the pipeline can't forecast volatility, it is broken.
5. **Carry** (interest-rate differentials). Not in the ECB file; needs another data source (see Stage F5).

**What success looks like.** Be honest about the odds: FX is among the most efficient markets, and published
momentum/value effects are small and unstable. A realistic good outcome is a small positive skill (say 1–3%)
at a monthly-plus horizon that is *consistent across years and currencies*. An equally valid outcome is a clear,
well-measured null. A large edge means a bug (look-ahead, overlapping-window leakage), not a discovery.

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| F-D1 | Targets are log returns over a horizon h of 5, 20, 60 and 120 trading days (about 1 week, 1, 3 and 6 months), plus the sign of the return. | The hypothesis is about months, so test months, and see where (if anywhere) skill appears as h grows. |
| F-D2 | Features are **summaries of long history**, not raw lagged returns: trailing returns over 5, 20, 60, 120, 250 days; distance from the 250- and 750-day mean (in log terms); trailing volatility (20, 60 days); month of year as sin/cos. | A 250-day window of raw returns is 250 noisy inputs for ~7,000 rows; summaries give the model the slow structure directly with a handful of inputs. |
| F-D3 | All features at time t use only data up to and including t; target uses t+1..t+h. Reuse `.qs.data.lagged` conventions and its alignment tests. | The Phase 9 rule, unchanged. |
| F-D4 | Evaluation is **purged walk-forward by year**: train on years before Y, drop the last h days of training (their targets reach into year Y), test on year Y. | With h-day targets, neighbouring rows share most of their return. Without purging, training rows leak the test period. |
| F-D5 | Baselines: zero change; training mean; and two fixed rules with no fitting, the **sign of the past 250-day return** (trend) and **minus the distance from the 750-day mean** (reversion). A model must beat the rules, not just zero. | If a one-line rule does as well as the network, the network found nothing extra. |
| F-D6 | Headline metric is skill vs zero change as in Phase 9, **per test year**, with the count of years positive. Add the correlation between forecast and outcome (information coefficient) and directional accuracy vs up-share. | One pooled number hides instability; the count of positive years is the honest summary. |
| F-D7 | Significance: with overlapping targets, the effective sample is about (years × 252 / h), not the row count. Judge results with a **block bootstrap** (blocks of at least h days) or a **permutation test on whole years**, never a naive standard error. | Overlap makes naive t-statistics wildly optimistic. |
| F-D8 | Count every configuration tried and report the count next to the best result. Choose model settings on a validation slice *inside* the training years, never on the test years. | Trying many currencies × horizons × features guarantees a few lucky hits by chance. |
| F-D9 | Pool currencies where it helps: one model trained on many pairs (features are scale-free returns), tested per pair. Pairs are all quoted against EUR, so crosses are derived ratios, not independent series. | Roughly 27 years of one pair is only ~330 independent months; pooling raises the sample, though correlated pairs raise it less than the pair count suggests. |
| F-D10 | New code in `fx.q` (features, purged walk, rules, bootstrap) and `bench/fx_trends.q`. Reuse existing scoring and controls. | Follows the existing layout. |

## 3. Stages

### Stage F1 — Features, targets and purging

- [x] **F1.1 Trend features.** `.qs.fx.features[rate;dates]`: from one rate series, the F-D2 feature columns,
  null until enough history exists; rows with any null dropped consistently in features and targets.
  *Accept:* tests on a hand-made series (a constant series has zero momentum and distance; a straight line in
  log space has constant trailing returns); a perturbation test showing changing a value at time t+1 never changes
  features at time t.
- [x] **F1.2 Horizon targets.** Targets for h in F-D1 (return and sign), reusing `laggedRows` so a gap in the
  data drops the row rather than counting as zero.
  *Accept:* alignment test as in Phase 9.
- [x] **F1.3 Purged walk-forward.** `.qs.fx.walkPurged`: per F-D4, scaling from training rows only.
  *Accept:* a test that no training row's target window overlaps any test row; with h=1 it reproduces `walk`.

### Stage F2 — Controls and baselines (before any real result)

- [x] **F2.1 Synthetic truth.** A simulated series with a known slow trend (e.g. a persistent drift that flips sign
  every few months, buried in noise at a realistic level). The pipeline must find it with the skill the theory
  predicts, and must find nothing in the same series with the drift removed.
  *Accept:* tests for both, as in the AR(1) test of Phase 9.
- [x] **F2.2 Overlap trap.** Deliberately run *unpurged* on pure random-walk data at h=60 and show it reports
  fake skill, then show the purged version does not. This proves F-D4 is doing its job.
- [x] **F2.3 Volatility positive control.** Forecast the next 20-day realised volatility from trailing
  volatility. Skill vs the training mean should be clearly positive on real data.
  *Accept:* if it isn't, stop and debug before going further.
- [x] **F2.4 Rule baselines.** The F-D5 rules scored with the same per-year metrics and bootstrap.

### Stage F3 — The main experiments

Run on the ECB data for EUR/USD, GBP/USD, USD/JPY, EUR/CHF, AUD/USD, USD/CAD (the major pairs available), each at
h = 5, 20, 60, 120. In `bench/fx_trends.q`, results as a table in the log.

- [x] **F3.1 Rules only.** Trend and reversion rules per pair, horizon and year. Shows whether the raw
  hypotheses have any life before a model is involved.
- [x] **F3.2 Linear model** on the F-D2 features (no hidden layer; with weight shrinkage if qSolve has it,
  otherwise early stopping on an inner validation slice).
- [x] **F3.3 MLP** (one small hidden layer) on the same features. Compare with F3.2: extra skill from
  non-linearity is the interesting claim, and the first to distrust.
- [x] **F3.4 Feature ablation.** Drop each feature group in turn (momentum, reversion, volatility, seasonality)
  to see which one carries any skill.
- [x] **F3.5 Pooled model** across all pairs (F-D9), scored per pair.
- [x] **F3.6 Direction.** Sign-of-return classification at the best horizon, against up-share.

### Stage F4 — Is the result real?

- [x] **F4.1 Significance** per F-D7 for every headline number; report the multiple-comparisons count (F-D8).
- [x] **F4.2 Stability.** Skill by year and by pair as a table; flag any result that depends on one year (2008,
  2015 CHF, 2020 are likely to dominate).
- [x] **F4.3 Shuffled-target and leak controls** run on the real features, as in Phase 9.
- [x] **F4.4 Cost-aware check (optional).** A simple monthly position rule from the forecast sign, with a
  transaction cost assumption, to ask whether any edge would survive trading. Skill can be real and still not
  be tradable.

### Stage F5 — Extensions (only if F3/F4 show something, or the owner asks)

- [ ] **F5.1 Carry.** *Not run: the trigger (F3/F4 showing something) wasn't met.* Add interest-rate differentials (central-bank policy rates, e.g. from FRED or the ECB data
  portal). Decide the source and licence before building; needs a new fetch script.
- [ ] **F5.2 Regimes.** *Not run (trigger not met).* Condition on volatility level or trend strength, and see whether the skill concentrates
  there.
- [ ] **F5.3 Longer history.** *Not run (trigger not met); still the main thing that would change the answer.* The ECB file starts in 1999; a longer series (pre-euro synthetic rates, or another
  source) would add decades of independent months, which is the main limit on this whole study.
- [x] **F5.4 Dashboard and README.** (README only; no dashboard change.) Add the findings to the README next to the Phase 9 section, stated plainly,
  including a null result if that is what we get.

## 4. Risks and things to watch

- **Few independent samples.** Around 27 years is the hard limit; no modelling cleverness changes it.
- **Look-ahead in features.** The most likely bug: a trailing mean that includes the future. The perturbation
  test in F1.1 exists for this.
- **Researcher degrees of freedom.** Each extra feature, horizon or pair raises the chance of a false find
  (F-D8). Fix the experiment list *before* running it.
- **Regime change.** The 2000s and the 2020s are different markets; a model that works in one may not in the
  other, which is what the per-year table is for.
- **Overfitting by a network on ~7,000 overlapping rows.** Keep hidden layers small, compare against the linear model.

## Progress log

- 2026-10-25: Plan written, not started. Open question for the owner: whether to look for a second data source
  (rates and longer history) now or after F3.
- 2026-10-25: F1 and F2.1, F2.2, F2.4 done (tests in `tests/test_fx.q`, 8 new, about 17 s). `fx.q` gained
  `features`, `rows`, `without`/`keep`/`col`, `leakRows`, `forecast`, `split` (the purge), `walkRows`, `walkPooled`,
  `walkRule`, `byYear`, `interleaved`, `boot1`/`summary` (circular block bootstrap). `score` now also returns `ic`
  and `skillVsMean`. Findings while building:
  - **F2.1:** a drift flipping sign about every 120 days (0.1%/day, noise 0.5%/day) is found: information
    coefficient 0.26, skill +0.064; with the drift removed skill -0.006 and the interval covers 0. (A first try
    with 0.04%/day drift was too weak to see in 27 years' worth of rows: worth remembering for how small a real
    effect would be.)
  - **F2.2 differs from the plan.** On a random walk at h=60, the *unpurged walk-forward* gives no fake skill
    (-0.19 for a small MLP; only the h rows next to each test year overlap, 1-2% of the training rows). The leak
    shows up with an *interleaved* split (train on all rows except every 5th, test on those): skill +0.59 and
    direction 78% on pure noise. `.qs.fx.interleaved` and a test show it; purging matters for any split that mixes
    train and test rows, and costs little in walk-forward.
  - Harness/q: `ij` is a keyword (as a local it gave `'ij`); `mdev` of a constant gives null (rounding makes the
    variance slightly negative), so volatility is `sqrt 0f|E[r^2]-E[r]^2`; a row's own return is part of the raw
    laggedRows output, so a row whose own return is missing is dropped even though that return isn't a feature.
  - The suite has one failing test, `gw_strain_parse_cache`, from uncommitted work on `gw.q`/`tests/test_gw.q` that
    isn't part of this plan; every other test passes.
- 2026-10-25: F2.3, F3, F4 done (`q bench/fx_trends.q -s 4`, about 25 minutes; deterministic, a rerun gave identical
  numbers). 2010-2026 walk-forward by year, 6 pairs (EURUSD GBPUSD USDJPY USDCHF AUDUSD USDCAD) x h 5/20/60/120.
  **Result: no reliable edge.**
  - **F2.3 volatility control passes, less strongly than hoped.** Next-20-day log variance, skill vs the training
    mean: EURUSD 0.30, USDCAD 0.30, USDJPY 0.22, AUDUSD 0.17, GBPUSD 0.05, USDCHF 0.00, positive in 12-14 of 17
    years. (A first version reported 0.99 because it scored against zero change instead of the mean; fixed with
    `skillVsMean`.)
  - **Counts (F-D8):** 204 configurations scored (192 forecast configurations + 12 classifiers). 29 of the 192 had
    skill above 0; 2 had skill above 0 with one-sided p <= 0.05 (reversion rule on EURUSD h=120, p=0.026, skill
    +0.030; reversion on USDCHF h=20, p=0.024, skill +0.005); the smallest p is 0.024, Bonferroni-adjusted 1.
  - **By model and horizon (mean over the 6 pairs):**

    | model | h=5 | h=20 | h=60 | h=120 |
    |---|---|---|---|---|
    | trend rule (sign of 250-day return), skill | -0.0003 | -0.0013 | -0.0045 | -0.020 |
    | reversion rule (distance from 750-day mean), skill | -0.0002 | -0.0014 | -0.0061 | -0.016 |
    | linear (11 features), skill | -0.061 | -0.111 | -0.205 | -0.316 |
    | MLP 16, skill | -0.047 | -0.150 | -0.380 | -0.614 |
    | pooled linear (one model, 6 pairs), skill | -0.003 | -0.009 | -0.032 | -0.042 |
    | pooled MLP, skill | -0.013 | -0.061 | -0.117 | -0.127 |
    | trend rule, information coefficient | 0.000 | -0.009 | 0.012 | 0.052 |
    | pooled linear, information coefficient | -0.006 | 0.005 | 0.000 | 0.024 |

    Models with more freedom did worse, and worse at longer horizons (fewer independent targets, more
    extrapolation). Nothing beat "no change" on average anywhere. Pairs (6) with skill above 0 at h=120: trend 3,
    reversion 3, pooled linear 3, linear 0, MLP 0.
  - **The rules disagree across pairs** (information coefficient of the reversion rule at h=120: EURUSD +0.24,
    GBPUSD +0.21, USDCHF +0.13, AUDUSD -0.09, USDCAD -0.25, USDJPY -0.35). That is what you'd see if there were no common effect.
  - **Best by skill:** pooled linear EURUSD h=120 +0.059 (p=0.076, interval -0.024 to +0.134, positive in 10 of 17
    years, from -0.80 in 2013 to +0.43 in 2026), pooled linear GBPUSD h=120 +0.038, trend USDCHF h=120 +0.033
    (p=0.17), reversion GBPUSD +0.031, reversion EURUSD +0.030, pooled linear EURUSD h=60 +0.028. Dropping the best
    year barely changes them (`skillExBest`), but none is significant, and 2026 (a partial year) contributes the
    largest positive values.
  - **F3.4 ablation (linear, h 20/60):** removing a feature group never made the skill positive; removing
    momentum or reversion made it slightly less negative (-0.111 -> -0.085/-0.080 at h=20), i.e. those features
    add overfitting, not signal. Seasonality and volatility features changed almost nothing.
  - **F3.6 direction (MLP, h 20 and 60):** accuracy 40-54% against 45-57% for always guessing the training
    majority; best z = +1.4 (USDCHF h=60), worst -2.8 (USDJPY h=60). None significantly better, two significantly
    worse. (z counts each h days as one independent observation.)
  - **F4.2 stability:** mean skill across the 192 configurations was above zero only in 2020 (+0.011) and 2021
    (+0.013); the other years were -0.04 to -0.40, worst in 2012-2013 and 2015. Of the 5 best configurations each
    had 3 to 8 negative years out of 17.
  - **F4.3 controls on real features:** shuffled targets -0.025 to -0.084 (none), answer leaked +0.95 to +0.96
    (found), so the real-data pipeline behaves as in Phase 9.
  - **F4.4 trading check (h=20, forecast sign held 20 days, 2 bp per switch):** mean gross return -0.59%/yr, net
    -0.70%/yr over 36 pair/model combinations, against -0.09%/yr for always long; best net were trend USDJPY
    +4.6%/yr (hold +3.1%), pooled linear EURUSD +2.6%, MLP EURUSD +2.0%, with several of similar size going the
    other way. No edge that survives the number of tries.
- 2026-10-25: l2 sensitivity (`bench/fx_trends_l2.q`, linear model, l2 1 and 10, h 20/60/120): more shrinkage moves
  skill towards zero from below but never above (mean skill -0.038/-0.021 at h=20, -0.103/-0.063 at 60,
  -0.178/-0.124 at 120; 0 of 36 positive, 0 significant), and the information coefficient turns negative. So the
  negative skill is overfitting noise, not a signal held back by too little regularisation.
- 2026-10-25: Conclusion and departures. **Slow trends in daily ECB rates (momentum over 1-12 months, distance
  from the 3-year mean, volatility, season) gave no out-of-sample edge for forecasting the next week to six
  months, for any of the six pairs, with simple rules, linear models, small networks or pooled models, and no
  trading rule made money after costs.** There are hints worth a longer sample, not a conclusion: mean reversion
  at 6 months in EUR/USD and GBP/USD (information coefficient about 0.2), and the volatility control shows the
  pipeline can find structure when it is there. Departures from the plan, all deliberate:
  - F-D1: sign targets were done as a separate classifier at h=20 and 60 only, not at every horizon.
  - F-D5: "rules with no fitting" are scored two ways: the fixed sign rule by direction and mean return versus
    holding (`ruleReturn`, `holdReturn`), and a one-parameter version (slope fitted on training rows through the
    origin) for skill and bootstrap, since a bare sign has no scale to compare to MSE.
  - F-D8: model settings were fixed in advance (30 epochs, learning rate 0.001, l2 0.1 for the linear model,
    16 hidden units and dropout 0.1 for the MLP), not chosen on an inner validation slice; the l2 check above covers
    the main risk of that. Test years were never used to choose anything.
  - F-D7: block length is max(h, 20) days, 500 resamples; the p-value is the share of resamples with skill <= 0
    (a bootstrap p-value for "no skill", not a permutation test).
  - F1.1/F1.3 names and signatures differ (`.qs.fx.features[rate;dates]`, `.qs.fx.walkRows`); the "h=1 reproduces
    `walk`" acceptance was replaced by direct split tests since the feature sets differ.
  - F3 uses USD/CHF, not EUR/CHF (same pairs as Phase 9).
  - Pooled models scale all pairs together, so higher-volatility pairs weigh more in the fit.
  F5 was not started: its trigger (F3/F4 showing something) was not met. The biggest limit remains the sample
  (about 330 independent months per pair); longer history (F5.3) is the thing most likely to change the answer.
