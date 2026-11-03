# Forecasting exchange rates

A real-world test of qSolve on a hard problem with an honest answer. Can a network forecast exchange rates from
their own history better than "no change"? Two studies use 27 years of the European Central Bank's daily euro
reference rates:
- **daily**: tomorrow's return from the last few days;
- **slow trends**: the next week to six months from slow structure (momentum, distance from the long-run mean,
  volatility, season).

**In short: nothing beats "no change" out of sample.** That is the expected answer for liquid currency markets.
The controls show the pipeline itself is sound:
- given the answer, it finds it (skill +0.99);
- given shuffled targets, it finds nothing;
- on a quantity that *is* predictable (volatility), it finds real skill (up to +0.30).

The same tools would show an edge if one existed.

The details and every number are in Phase 9 of [improvemet_plan.md](../../plans/completed/improvemet_plan.md) and in
[fx_trends_plan.md](plans/completed/fx_trends_plan.md).

---

## Data and scoring

- **Data**: the ECB's daily euro reference rates, since 1999, for about 30 currencies (`scripts/fetch_data.sh fx`).
  [src/fx.q](src/fx.q) builds any currency pair from the euro rates: for example, EUR/USD directly, or GBP/USD as a
  ratio of two euro rates.
- **Skill** compares a forecast's mean squared error with that of forecasting no change:
  `1 - MSE(model) / MSE(no change)`. Above 0 beats "no change"; below 0 is worse.
- **No look-ahead.** Every score is on later data than the model trained on, and features at day t use data up to
  day t only.

```q
t:.qs.fx.ecb `:projects/markets/data/eurofxref-hist.csv
in:.qs.fx.inputs[t;`EURUSD`GBPUSD`USDJPY`USDCHF`AUDUSD`USDCAD]   / daily log returns of each pair
d:.qs.data.lagged[in;`EURUSD;5;1;0.8]                            / last 5 days of all six -> tomorrow's EUR/USD
.qs.fx.fit[d;`hidden`act`out`loss`optimiser`lr`epochs!(enlist 16;`tanh;`linear;`mse;`adam;0.001;30)]
```

## Study 1: tomorrow's return

The test set is the last 20% of days, from April 2021 onwards. ECB data runs to 23 October 2026.

| Forecast of tomorrow's return | Skill | Direction right | Up days |
|---|---|---|---|
| EUR/USD from its last 10 days, linear | +0.005 | 51.0% | 48.9% |
| EUR/USD from its last 10 days, neural network | −0.001 | 50.3% | 48.9% |
| EUR/USD from 6 major pairs' last 5 days, linear | −0.010 | 49.8% | 48.9% |
| EUR/USD from 6 major pairs' last 5 days, neural network | −0.022 | 48.4% | 48.9% |
| the same, 5 days ahead | −0.018 | 51.1% | 48.5% |
| GBP/USD from its own or the majors' history | −0.008 to −0.012 | 47.6–49.6% | 50.8% |
| Control: training targets shuffled | −0.008 | 50.5% | 48.9% |
| Control: the answer leaked into the inputs | **+0.986** | **99.1%** | 48.9% |

**Walk-forward.** The model is retrained every year on all earlier years, and forecasts EUR/USD from the six
majors, 2010–2026.
- Skill was above zero in 1 of 17 years, for both the linear model and the network. Mean skill was −0.028.
- An up/down classifier was right 51.1% of the time. Always guessing the more common direction was right 51.6% of
  the time.

**What this says:** daily exchange rates behave like a random walk.

## Study 2: slow trends, months and years instead of days

A daily window can't see a slow drift: daily moves are about 0.5%, and a slow drift might be a few hundredths of a
percent a day. So this study asks a different question. Does slow structure help forecast 5, 20, 60 or 120 trading
days ahead (about a week, one month, three months, six months)?

**Features**, using only data up to the forecast day:
- trailing returns over 5 to 250 days;
- distance from the 250- and 750-day mean;
- trailing volatility;
- month of the year.

**Models**:
- two fixed rules with nothing fitted: trend (the sign of the past year's return) and reversion (minus the distance
  from the 3-year mean);
- linear models and small networks, fitted to each pair;
- one model pooled over all six pairs.

**Evaluation**:
- **Purged walk-forward by year** over 2010–2026 and six pairs. Training rows whose targets reach into the test year
  are dropped.
- **Intervals from a block bootstrap.** Overlapping targets make ordinary standard errors far too narrow.
- **Every configuration counted.** 204 were scored, and the best result is reported next to that count.

### Result: no edge

**Of 192 forecast configurations, 29 had skill above zero.**
- 2 of those had a one-sided p of 0.05 or less. If every configuration were noise, about 10 would reach that level
  by chance.
- None survives a correction for the 204 configurations tried.

Mean skill over the six pairs:

| Model | 1 week | 1 month | 3 months | 6 months |
|---|---|---|---|---|
| trend rule | −0.0003 | −0.0013 | −0.0045 | −0.020 |
| reversion rule | −0.0002 | −0.0014 | −0.0061 | −0.016 |
| linear, 11 features | −0.061 | −0.111 | −0.205 | −0.316 |
| small network (16 units) | −0.047 | −0.150 | −0.380 | −0.614 |
| pooled linear (one model, 6 pairs) | −0.003 | −0.009 | −0.032 | −0.042 |
| pooled network | −0.013 | −0.061 | −0.117 | −0.127 |

- **More flexible models did worse, and worse again at longer horizons.** Longer horizons have fewer independent
  targets, so there is more room to overfit.
  - More L2 shrinkage moves the linear model's skill towards zero, but never above it.
  - Removing the momentum or reversion features makes the skill slightly *less* negative: they add overfitting,
    not signal.
- **The best result is a pooled linear model of EUR/USD at six months.** Its skill is +0.06 (interval −0.02 to
  +0.13, p = 0.08), and it ranged from −0.80 to +0.43 between years.
- **Mean reversion at six months looks mildly useful for EUR/USD and GBP/USD, and harmful for USD/JPY and USD/CAD.**
  The correlation between forecast and outcome is about +0.2 for the first two, and −0.25 to −0.35 for the other
  two. Rules that disagree across pairs like this are what you'd see if there were no common effect.
- **Trading on it loses money.** Holding the forecast's sign for 20 days lost 0.70% a year on average after costs,
  across 36 pair and model combinations.

### The checks behind it

- **A hidden drift is found.** A synthetic series with a drift that flips sign every 120 days gives positive
  skill. The same noise without the drift gives none.
- **The usual controls behave.** Shuffled targets give no skill, and a leaked answer gives +0.95.
- **Volatility is predictable, and the pipeline finds it.** Forecasting the next 20 days' volatility gives skill up
  to +0.30.
- **The purge is needed.** A split that mixes training and test rows shows fake skill (+0.59, 78% direction) on a
  pure random walk. That is the leak purging prevents.

**The limit is the sample.** 27 years is about 330 independent months per pair. Longer history, or data with many
more cycles, is what would most likely change the answer.

## Reproducing

Run these from the repository root.

```bash
scripts/fetch_data.sh fx                         # the ECB file, into projects/markets/data/
q projects/markets/experiments/fx.q -s 4              # study 1, about 30 s
q projects/markets/experiments/fx_trends.q -s 4       # study 2, about 25 minutes
q projects/markets/experiments/fx_trends_l2.q -s 4    # does more shrinkage change study 2's answer?
q tests/run.q -only fx                           # this project's tests
```

`eurusd` and `gbpusd` are also datasets in the dashboard and in `app/train.q`, with the naive forecast shown
alongside.

## Layout

```
projects/markets/
  README.md       this write-up
  src/fx.q        ECB data, currency pairs, features, purged walk-forward, rules, block bootstrap, forecast
                  scoring and controls (.qs.fx); also the eurusd and gbpusd datasets
  experiments/    fx.q (study 1), fx_trends.q (study 2), fx_trends_l2.q (its shrinkage check)
  tests/          test_fx.q
  plans/          this project's plans: future/, inProgress/, completed/
  data/           eurofxref-hist.csv (git-ignored)
```
