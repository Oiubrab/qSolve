# qSolve

A small multilayer perceptron (feed-forward neural network) written in pure q/kdb+.
No Python and no dependencies beyond q itself (plus `curl` and `gunzip` to download the datasets).

- Layers with `sigmoid`, `tanh`, `relu`, `linear` or `softmax` activation
- Losses: mean squared error, binary and categorical cross-entropy
- Optimisers: SGD, momentum, Adam; optional L2 weight decay
- Mini-batches, shuffling, validation each epoch, history table, confusion matrix, save/load
- Dropout, learning-rate schedules (step, cosine), early stopping that can keep the best epoch's model
- Multi-threaded training and prediction (`-s N`)
- Decision-boundary plots of 2-D models as SVG
- Vectorised backpropagation, checked against finite differences in the test suite
- Built-in datasets, from logic gates up to MNIST and Fashion-MNIST
- Train on any kdb+ table or CSV: numeric columns standardised, categorical ones one-hot encoded,
  missing values filled; regression or classification chosen from the target column
- Time-series forecasting from lagged windows, with a time-ordered split and naive baselines
- Real-world test: daily exchange-rate forecasting with honest baselines, controls and walk-forward
- Hyperparameter search (grid or random) returning a ranked results table, and k-fold cross-validation
- Serve a trained model to other kdb+ processes over IPC
- A web dashboard for all of the above: train with live charts, compare runs, search, predict

MNIST with one hidden layer of 128 relu nodes reaches 97.5% test accuracy in 5 epochs (under 20 seconds with 4 threads).

## Dashboard

```bash
q app/dashboard.q -s 4                    # then open http://127.0.0.1:8080/   (-port to change, -host to bind elsewhere)
```

The dashboard is a web page served by q itself. Nothing else needs installing:
- **Train:** pick a built-in dataset or upload a CSV (choose the column to predict, or a column to forecast
  from its past), set the layers and options, and watch loss and accuracy update after every epoch. When
  training finishes you get validation scores, a confusion matrix, the decision boundary for 2-D data,
  predicted-against-actual or forecast-against-actual for numeric targets, optional cross-validation, and
  the layer table.
- **Runs:** every run this session; tick up to 8 to compare their validation-loss curves.
- **Search:** try combinations of learning rates, layer sizes, activations, batch sizes and dropout on the
  Train tab's data; results stream in ranked by validation loss, and one click copies the best into Train.
- **Predict:** says what the model learned, what it takes and what it answers with, then asks for inputs
  in the data's own terms: a form for table models (with each column's training range), a clickable map
  for 2-D models, a drawing pad for MNIST and Fashion-MNIST, or recent values for a forecast. Answers come
  by name with each class's probability. "Try a validation example" fills in a row the model didn't train
  on and shows whether it got it right. Inputs outside the training range get a warning.
- **Models:** models saved to `models/`, which can be loaded back for predictions.

Training runs in the dashboard's q process, so the page waits for each run to finish, but every epoch is
pushed to the browser as it completes. Runs live for the session; save the ones you want to keep. The
dashboard listens on 127.0.0.1 only, unless you pass `-host`.

### Using it from KX Dashboards

The same port also answers ordinary q queries, so KX Dashboards (or any q process) can read the
session's results. Start the dashboard, then add `app/kx/qsolve.json` as a connection: copy it into
your Dashboards `data/connections/` folder, or enter the same host and port in the connection editor.
Useful queries:

```q
0!.dash.runs                                  / one row per run: data, model, epochs, validation scores, time
.dash.history[]                               / every epoch of every run, with a run column
select epoch,trainLoss,valLoss from .dash.history[] where run=1
```

These queries can run any q code, which is why the dashboard listens on localhost only by default.

## Quick start

```bash
q app/train.q -data moons -hidden 16 -epochs 100 -optimiser adam -lr 0.01 -plot moons.svg

q app/train.q -data csv -file sales.csv -target price -features area rooms city -optimiser adam -lr 0.01 -epochs 100 -save price.qsm
q app/train.q -data moons -optimiser adam -epochs 30 -search lr=0.01,0.001 hidden=4,16,16x16 act=tanh,relu

q app/train.q -data csv -file load.csv -series demand -lags 48 -horizon 1 -hidden none -optimiser adam -lr 0.003 -epochs 100

scripts/fetch_data.sh                 # downloads MNIST, Fashion-MNIST and Iris into data/
q app/train.q -s 4 -data mnist -hidden 128 -act relu -optimiser adam -lr 0.001 -epochs 5 -batch 64 -save mnist.qsm
```

Start q with secondary threads (`q app/train.q -s 4 ...`, with the script first) and large batches and
predictions are spread across them: MNIST trains about 1.5x faster with `-s 4`.

`train.q` infers the output layer from the data. It picks softmax with cross-entropy for several classes,
sigmoid with binary cross-entropy for one 0/1 output, and linear with MSE for numeric targets. Its header
lists every option.

## Using it from q

```q
\l src/defineEquation.q
\l src/data.q
\l src/plot.q                                           / only needed for .qs.plot2D

d:.qs.data.split[.qs.data.spirals[2000;0.02];0.8]      / `X`Y`Xval`Yval
m:.qs.init[2;32 32 2;`tanh`tanh`softmax;42]             / 2 inputs, two hidden layers, 2 outputs
r:.qs.train[m;d;`loss`optimiser`lr`epochs`verbose!(`xent;`adam;0.005;300;50)]
r`history                                               / one row per epoch
.qs.predict[r`model;0.1 0.2]                            / one example, or a matrix of rows
.qs.evaluate[r`model;d`Xval;d`Yval;`xent]               / `loss`acc
.qs.confusion[r`model;d`Xval;d`Yval]                    / rows true class, columns predicted
.qs.save[`:spirals.qsm;r`model]
.qs.plot2D[r`model;d;`:spirals.svg;"spirals"]           / decision boundary, 2-input models only
```

### Core (`src/defineEquation.q`)

| Function | What it does |
|---|---|
| `.qs.init[nIn;sizes;acts;seed]` | New model. `sizes` gives one entry per layer, the last being the output. `acts` is one per layer, or a single activation for every layer. Xavier initialisation (He for relu), zero biases. A `seed` of `0N` leaves q's random state alone. |
| `.qs.predict[m;X]` | Outputs for a matrix with one row per example, or for a single example vector. |
| `.qs.train[m;data;opts]` | Trains on `data` (`` `X`Y ``, optionally `` `Xval`Yval ``). Returns `` `model`history`bestEpoch ``. |
| `.qs.evaluate[m;X;Y;loss]` | `` `loss`acc`mae`rmse`r2 `` dictionary (`acc` is null for numeric targets). |
| `.qs.confusion[m;X;Y]` | Confusion matrix. |
| `.qs.grad[m;X;Y;loss]` | Gradient of the loss, shaped like the model (`.qs.gradDrop` adds a dropout rate). |
| `.qs.search[d;space;base;n]` | Trains one model per configuration and returns a table ranked by validation loss. `space` maps options (and `hidden`, `act`) to candidate values, `base` fixes the rest, `n` is `0N` for the whole grid or a number to sample. |
| `.qs.crossValidate[d;k;base]` | k-fold cross-validation of one configuration: `` `folds`summary `` (per-fold scores; mean and sd). |
| `.qs.summary m` | One row per layer: inputs, outputs, activation, parameters. |
| `.qs.save[file;m]` / `.qs.load file` | Saves and loads a model (q's native format). |

`opts` is merged over `.qs.defaults`:

| Key | Default | |
|---|---|---|
| `epochs` | 10 | |
| `batch` | 32 | the last batch of an epoch may be smaller |
| `lr` | 0.1 | learning rate |
| `loss` | `` `mse `` | `` `mse`mae`bce`xent`norm `` (`norm` is qSolve's original loss) |
| `optimiser` | `` `sgd `` | `` `sgd`momentum`adam `` |
| `momentum` | 0.9 | for `momentum` |
| `beta1`, `beta2` | 0.9, 0.999 | for `adam` |
| `l2` | 0 | weight decay |
| `dropout` | 0 | dropout rate on hidden layers during training |
| `schedule` | `` `none `` | `` `none`step`cosine `` learning-rate schedule |
| `decay`, `stepEvery` | 0.5, 10 | `step`: multiply `lr` by `decay` every `stepEvery` epochs |
| `minLr` | 0 | `cosine`: final learning rate |
| `patience` | `0N` | stop after this many epochs without a lower validation loss (training loss if no validation data); `0N` never stops |
| `keepBest` | `0b` | return the model from the epoch with the lowest validation loss rather than the last |
| `shuffle` | `1b` | |
| `seed` | 42 | `0N` to leave q's random state alone |
| `verbose` | 1 | log every n epochs, 0 for silent |

A model is a dictionary `` `weight`bias`act!(W;B;acts) ``. `W[l]` is an `n_l × n_(l-1)` matrix, `B[l]` a vector,
and `acts` has one symbol per layer.

`xent` expects one-hot targets. Softmax is only allowed on the output layer, and only with `xent`.

### Data (`src/data.q`)

| Function | Dataset |
|---|---|
| `.qs.data.gate[`` `and`or`xor`nand ``]` | The four rows of a logic gate |
| `.qs.data.fourBit n` | 4 random bits, classified as 1100, 0011 or neither (the original qSolve task) |
| `.qs.data.sine n` | Regression of sin x |
| `.qs.data.circles[n;noise]`, `.moons`, `.spirals` | 2-D two-class problems |
| `.qs.data.houses n` | A house-price table for regression, with categorical and missing values |
| `.qs.data.seasonal[n;period;noise]` | A series with trend, a cycle and autocorrelated noise |
| `.qs.data.iris file` | Iris, standardised, split 80/20 |
| `.qs.data.mnist dir` | MNIST or Fashion-MNIST (IDX files), test set as validation |
| `.qs.data.load[name;n;noise]` | Any of the above by name, already split |

Helpers: `oneHot`, `split`, `standardise`, `stats`/`scale`, `take`, `idx` (reads any unsigned-byte IDX file).
Generated datasets use q's random state, so call `system "S 42"` first for repeatable data.

### Your own data

From a kdb+ table or a CSV file:

```q
d:.qs.data.fromTable[t;`area`rooms`city;`price;0.8]    / or () for every other column
d:.qs.data.csv[`:sales.csv;();`price;0.8]              / header row; numbers and text columns detected
d`enc                                                   / the column encoding, fitted on the training rows
.qs.data.inputNames d`enc                               / `area`rooms`city_east`city_north ...
p:.qs.predict[m;.qs.data.encodeX[d`enc;newRows]]
.qs.data.decodeY[d`enc;p]                               / back to prices, or to class symbols
```

Numeric columns are standardised and their nulls become the mean. Symbol or text columns are one-hot
encoded over the levels seen in training; nulls and unseen levels encode as all zeros. A symbol target
means classification, a numeric one regression (standardised, and decoded back by `decodeY`).

### Time series

```q
s:exec demand from t                                    / any numeric series, in time order
d:.qs.data.windows[s;48;1;0.8]                          / 48 lags, 1 step ahead, first 80% of time trains
r:.qs.train[.qs.init[47;16 1;`tanh`linear;42];d;`loss`optimiser`lr`epochs!(`mse;`adam;0.001;100)]
f:.qs.data.unwindow[d;.qs.predict[r`model;d`Xval];d`levelVal]   / forecasts in the series' units
.qs.data.naive[s;48;1;0.8;24]                           / MAE of last-value and seasonal-naive forecasts
```

Each row holds the window's earlier values and the target, all relative to the window's last value, so a
trend or a shift in level doesn't take validation outside what the model saw in training. R² from
`.qs.evaluate` on these rows is therefore R² of the *change*, a harder bar than R² of the level.

Or build a dictionary `` `X`Y `` yourself, where `X` has one row per example and `Y` is one-hot (or a single
column), then call `.qs.data.split` and `.qs.data.standardise` and pass the result to `.qs.train`.

## Forecasting exchange rates: a real-world test

`fx.q` loads the European Central Bank's daily euro reference rates (since 1999, about 30 currencies;
`scripts/fetch_data.sh fx`), builds any currency pair, and scores forecasts of daily returns against the
naive forecast "no change". Skill is `1 - MSE(model) / MSE(no change)`: above 0 beats it, below 0 is worse.

```bash
q bench/fx/fx.q -s 4          # the whole experiment, about 30 s
```

```q
t:.qs.fx.ecb `:data/fx/eurofxref-hist.csv
in:.qs.fx.inputs[t;`EURUSD`GBPUSD`USDJPY`USDCHF`AUDUSD`USDCAD]   / daily log returns of each pair
d:.qs.data.lagged[in;`EURUSD;5;1;0.8]                            / last 5 days of all six -> tomorrow's EUR/USD
.qs.fx.fit[d;`hidden`act`out`loss`optimiser`lr`epochs!(enlist 16;`tanh;`linear;`mse;`adam;0.001;30)]
```

Results (ECB data to 23 October 2026; tested on the last 20% of days, April 2021 onwards):

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

Walk-forward (retrained every year on all earlier years, EUR/USD from the six majors, 2010–2026): skill was
above zero in 1 of 17 years for both the linear model and the network (mean −0.028), and the up/down
classifier was right 51.1% of the time against 51.6% for always guessing the more common direction.

**What this says:** daily exchange rates behave like a random walk, and nothing here beats "no change"
out of sample. That is the expected answer for liquid markets, and the controls show the pipeline itself
is sound: given the answer it finds it (skill 0.99), and given shuffled targets it finds nothing. The
same tools would show an edge if one existed, so they're ready for data where one might: intraday data,
other markets, or other inputs (rates, news, order flow). `eurusd` and `gbpusd` are also datasets in the
dashboard and `train.q`, with the naive forecast shown alongside.

### Slow trends: months and years instead of days

A second study asks whether slow structure (1-12 month momentum, distance from the 3-year mean, volatility, month of
the year) helps forecast the next week to six months. It is purged walk-forward by year (2010-2026, training rows
whose targets overlap the test year are dropped), over six pairs and horizons of 5, 20, 60 and 120 trading days, with
fixed trend and reversion rules, linear models, small networks and one model pooled over the pairs. Intervals come
from a block bootstrap, because overlapping targets make ordinary standard errors far too narrow.

```bash
q bench/fx/fx_trends.q -s 4      # the whole study, about 25 minutes
q bench/fx/fx_trends_l2.q -s 4    # does more shrinkage change the answer?
```

**Result: no edge.** Of 192 forecast configurations 29 had skill above zero, 2 of those with one-sided p <= 0.05,
against about 10 expected at that level if all were noise, and none survives a correction for the 204 tried. The
average skill was negative for every model and horizon, and worse for flexible models and longer horizons (linear
-0.06 to -0.32, small network -0.05 to -0.61, rules about 0, pooled linear -0.003 to -0.04). The best, a pooled
linear model of EUR/USD at six months, had skill +0.06 (interval -0.02 to +0.13, p = 0.08) and ranged from -0.80 to
+0.43 between years. Mean reversion at six months looks mildly useful for EUR/USD and GBP/USD (correlation about 0.2)
and harmful for USD/JPY and USD/CAD. Holding the forecast's sign for 20 days lost money after costs on average.

The checks behind it: a synthetic series with a hidden drift is found and the same noise without the drift is not;
shuffled targets give no skill and a leaked answer gives +0.95; forecasting volatility (which is predictable) gives skill
up to +0.30 on the same pipeline; and a split that mixes training and test rows shows fake skill (+0.59) on a random
walk, which is what the purging prevents. The limit is the sample: 27 years is about 330 independent months per pair.
`plans/completed/fx_trends_plan.md` has every number and the places the study departed from its plan.

## Finding gravitational waves in LIGO data

A second real-world test, revisiting the owner's honours thesis
([recurrent_grav_detector](https://github.com/Oiubrab/recurrent_grav_detector)): a network that finds binary
black hole mergers in real detector data from LIGO's Hanford (H1) and Livingston (L1) observatories. Everything is
pure q except one small Python converter for GWOSC's HDF5 files.

- **Signal processing** (`src/signal.q`): FFT, Welch power spectra, whitening, band-pass, resampling, spectrograms
  and phase-maximised matched filtering.
- **Waveforms and detectors** (`src/gw.q`): TaylorF2 and IMRPhenomB in the frequency domain, checked against
  LALSuite's conventions; antenna patterns and light-travel delays for H1 and L1. A matched filter with these
  templates finds GW150914 at SNR 19 in H1 and 13 in L1, 7 ms apart.
- **Training data made fresh each epoch**: real whitened noise from the public 4096 s files around six events and
  from O4a bulk data, chirps (10–80 M☉, aligned spins, random sky) injected at a chosen network SNR, simulated
  glitches, real glitches mined from the noise, and chirps in one detector only. These last three are labelled
  noise, which is what made the detector usable on real data.
- **Models**: a 1-D convolutional network (the core gained Conv1D layers for this) on 1 s of both detectors at
  2048 Hz. A GRU (also in the core, gradient-checked) didn't learn the task from raw samples.

Results for the best model, `gw_cnn_4`. Each false-alarm rate is measured by sliding one detector's noise against
the other's, never on the data it's applied to:

| test | result |
|---|---|
| Simulated chirps in held-out O1 noise (83 h of background) | at 1 false alarm per hour: half found at network SNR 10, 95% or more above SNR 16. At 1 per day: half at SNR 12 |
| 35 confident O3 detections, never trained on | 29 found at under 1 false alarm per hour (the first CNN, before the glitch work, found 12) |
| GW150914 and GW231226 | both found clearly; GW231226 (O4a) is louder than all 412 h of its day's background |
| Against a plain matched filter (41 templates) on the same 3000 injections | at 1 false alarm per hour the matched filter is better (half found at SNR 7.1 against 9.8). At 1 per day the CNN is better (12.2 against 16.4): the bare matched filter has no glitch rejection, and the CNN was trained against glitches |

So it finds loud mergers reliably, and finds quieter ones only some of the time. LIGO's own matched-filter searches
add signal-consistency tests against glitches, and found those O3 events at false-alarm rates typically below 1 per
year, so this is far from competitive. The point is a working, honest pipeline in q.

**Looking for missed events.** The search ran over six whole days of public bulk data, about 97 h in all:
- one O4a day (around GW231226);
- four O3a days chosen around events that only the independent IAS pipeline reported (Olsen et al. 2022);
- one O2 day around a GWTC-1 marginal candidate.

What it found:
- **No new candidates.** In every day, the uncatalogued triggers match what the time-slide background predicts.
  The 29 loudest were followed up with the matched filter and the data-quality flags. None holds up: one detector
  is below its own background, or the trigger is a flagged glitch, or it is at noise level.
- No trigger coincides with a public GraceDB alert that isn't in a catalogue.
- **GW190707_083226, which is in no LVK catalogue, was found.** It scored about 1 false alarm per 47 h, the
  second-loudest trigger of its day, and the matched filter confirms it in both detectors (network SNR 7.9).
- Two other IAS-only events at SNR 8 were missed, as expected at that strength.

The full numbers and every decision are in `plans/completed/gw_detection_plan.md` and
`plans/completed/gw_open_search_plan.md`.

```bash
scripts/gw/fetch_gw.sh                     # 4096 s event files for the noise library (a few GB)
scripts/gw/fetch_gw.sh --test-events --duration 32              # 32 s files around the test detections
q bench/gw/gw_cnn.q -s 4 -q -save models/gw_cnn_1.qsm          # train (options in the file's header)
q bench/gw/gw_events.q -s 4 -q -model models/gw_cnn_1.qsm      # the O3 detections, with time-slide background
q bench/gw/gw_eval.q -s 4 -q -model models/gw_cnn_1.qsm        # background, efficiency and glitch checks
q bench/gw/gw_mf_eval.q -s 4 -q -model models/gw_cnn_1.qsm     # the same injections against a matched filter
python3 -m venv .venv && .venv/bin/pip install h5py numpy       # for bulk data only
scripts/gw/search_days.sh models/gw_cnn_1.qsm 10 O4a:1387577000 # fetch, search and time-slide whole days
q bench/gw/gw_vet.q -s 4 -q -run O4a -start 1387577000 -stop 1387663400   # matched-filter follow-up of the top triggers
```

The dashboard's **Waves** tab shows any downloaded event, or a chirp you inject into held-out noise. It plots the
whitened strain, a spectrogram and the network's output over time, and gives the verdict at each measured
false-alarm rate.

**Data.** This uses data obtained from the Gravitational Wave Open Science Center (https://gwosc.org), a service of
the LIGO Scientific Collaboration, the Virgo Collaboration and KAGRA, under the CC BY 4.0 licence. LIGO is funded
by the U.S. National Science Foundation; see https://gwosc.org/acknowledgement/ for the full acknowledgement.

**References.** Abbott et al., *Open data from the first and second observing runs of Advanced LIGO and Advanced
Virgo*, SoftwareX 13 (2021); Ajith et al., *Inspiral-merger-ringdown waveforms for black-hole binaries with
nonprecessing spins*, PRL 106 (2011) (IMRPhenomB); LALSuite (https://git.ligo.org/lscsoft/lalsuite) for waveform
and detector conventions; George and Huerta, *Deep learning for real-time gravitational wave detection and
parameter estimation*, PLB 778 (2018), and Gabbard et al., *Matching matched filtering with deep networks for
gravitational-wave astronomy*, PRL 120 (2018), for CNN detectors on whitened strain; Olsen et al., *New binary
black hole mergers in the LIGO-Virgo O3a data*, PRD 106 (2022), for the IAS events.

## Serving a model

`train.q -save` stores the model, and for table or CSV data the column encoding too. `serve.q` answers
requests for it from other q processes:

```bash
q app/train.q -data csv -file sales.csv -target price -optimiser adam -lr 0.01 -epochs 100 -save price.qsm
q app/serve.q -model price.qsm -p 5050 -s 4
```

```q
h:hopen `::5050
h (`predictTable;([] area:120 80f; rooms:3 2; age:10 0n; city:`north`west))   / adds a prediction column
h (`predict;X)                                                                / raw model outputs
h (`info;::)                                                                  / layers, input columns, target
```

Only `` (`function;argument) `` calls to `predict`, `predictTable` and `info` are accepted; strings of q
code and anything else are refused with an error. For classes, `predictTable` also returns a `confidence`
column (the winning class's probability).

## Tests and benchmarks

```bash
q tests/run.q                    # unit tests; exits 1 on failure; -only <substring> to filter
q bench/benchmarks.q             # trains every benchmark dataset and checks its target; -skipLarge, -only <names>
```

`node tests/ui/check.js [dir]` checks the dashboard in a real browser: it starts a dashboard, uses every view in
headless Chromium, saves screenshots to `dir` (default `/tmp/qsolve-ui`), and fails on any page error. It
needs `chromium` and Node 22 or later, so it isn't part of `tests/run.q`.

The tests compare every gradient with finite differences, across activations, losses and network depths.
Tests and benchmarks that need downloaded data are skipped if it isn't there.

Benchmark results (seed 42):

| Benchmark | Model | Result | Target |
|---|---|---|---|
| and, or, nand | 2→1 sigmoid | 100% | 100% |
| xor | 2→4→1 tanh, sigmoid | 100% | 100% |
| fourBit | 4→8→3 relu, softmax | 100% | 100% |
| sine | 1→16→1 tanh, linear | ½·MSE 1.2e-5 | ≤ 0.001 |
| circles | 2→8→2 tanh, softmax | 99.0% | ≥ 98% |
| moons | 2→16→2 tanh, softmax | 100% | ≥ 97% |
| spirals | 2→32→32→2 tanh, softmax | 99.75% | ≥ 95% |
| houses | 7→32→1 tanh, linear (from a table) | R² 0.983 | ≥ 0.95 |
| seasonal | 47→16→1 tanh, linear (48 lags) | R² of change 0.883 | ≥ 0.85 |
| iris | 4→8→3 tanh, softmax | 96.7% | ≥ 93% |
| mnist | 784→128→10 relu, softmax, 5 epochs | 97.5% | ≥ 96% |
| fashion | 784→128→10 relu, softmax, 5 epochs | 87.4% | ≥ 84% |

## Layout

```
app/                        the programs (run them from the repository root, e.g. q app/dashboard.q -s 4)
  train.q                   command-line training
  serve.q                   IPC model server
  dashboard.q               web dashboard server
  ui/                       the dashboard's page
  kx/qsolve.json            a KX Dashboards connection to the dashboard
src/                        the library, loaded by everything else
  defineEquation.q          core: layers, conv front end, training, search (.qs)
  data.q                    datasets, tables, CSV, time series (.qs.data)
  plot.q                    decision-boundary SVGs (.qs.plot2D)
  fx.q                      exchange rates (ECB), currency pairs, forecast scoring and controls (.qs.fx)
  signal.q                  FFT, PSD, whitening, filtering, matched filters (.qs.sig)
  gw.q                      gravitational waves: catalogue, waveforms, noise, injections, search (.qs.gw)
bench/benchmarks.q          benchmark suite
bench/fx/                   FX experiments: fx.q (forecasting), fx_trends.q (slow trends), fx_trends_l2.q
bench/gw/                   GW experiments: detector training (gw_mlp.q, gw_cnn.q), real events (gw_events.q),
                            open-data search (gw_bulk_scan.q) and vetting (gw_vet.q)
scripts/fetch_data.sh       dataset downloader
scripts/gw/                 GW data: event strain (fetch_gw.sh), bulk HDF5 conversion (gwosc_bulk.py),
                            day-by-day search batches (search_days.sh)
tests/                      test runner, assertion helpers, tests, fixtures, the dashboard's browser check
plans/                      development plans and progress logs (inProgress/, completed/, future/)
data/  models/  .venv/      downloads, saved models, the Python venv for HDF5 (all git-ignored)
```
