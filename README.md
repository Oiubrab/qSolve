# qSolve

A small neural-network library written in pure q/kdb+: dense layers, with an optional Conv1D or GRU front end.
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
- 1-D convolutional layers (with max pooling) and a GRU, both gradient-checked
- Hyperparameter search (grid or random) returning a ranked results table, and k-fold cross-validation
- Serve a trained model to other kdb+ processes over IPC
- A web dashboard for all of the above: train with live charts, compare runs, search, predict

MNIST with one hidden layer of 128 relu nodes reaches 97.5% test accuracy in 5 epochs (under 20 seconds with 4 threads).

## Projects

Real-world tests of the library. Each one lives in its own folder under `projects/` with its library code,
experiments, scripts, tests, data, results and models, and a write-up of what it found.

- **[Finding gravitational waves in LIGO data](projects/gw/README.md).** A 1-D CNN finds binary black hole mergers
  in real LIGO strain:
  - half of all simulated mergers found at network SNR 9.8, at 1 false alarm per hour;
  - 29 of 35 held-out O3 detections found at under 1 false alarm per hour;
  - compared against a matched filter;
  - a search of about 98 h of public data found no new candidates, but recovered GW190707_083226, which is in no
    LIGO-Virgo-KAGRA catalogue.
- **[Forecasting exchange rates](projects/markets/README.md).** Daily ECB rates since 1999, forecast tomorrow and up to
  six months ahead with honest baselines, controls and purged walk-forward:
  - nothing beats "no change" out of sample;
  - the controls show the pipeline would find an edge if there were one.

## Plans

Development is planned in documents: what to build, decisions, acceptance checks and a dated progress log. Each
plan lives next to the code it is about:

- **`plans/`** at the root holds plans for the library itself: the network, training, the dashboard, and anything
  every project uses.
- **`projects/<name>/plans/`** holds plans for one project, for example `projects/gw/plans/` for the
  gravitational-wave search or `projects/markets/plans/` for the market studies.

Each plans folder has the same three subfolders:

| Folder | What goes in it |
|---|---|
| `future/` | written but not started |
| `inProgress/` | being worked on |
| `completed/` | done, kept as the record |

A plan moves from `future/` to `inProgress/` when work starts, and to `completed/` when it is finished.

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
- **Waves:** the [gravitational-wave project](projects/gw/README.md)'s detector on a real LIGO event, or on a chirp
  you inject into held-out noise.

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

scripts/fetch_data.sh                 # downloads MNIST, Fashion-MNIST and Iris into data/, and the ECB rates for the FX project
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
| `.qs.initConv[channels;length;layers;sizes;acts;seed]` | New model with a conv front end: `layers` is a table (or list of dictionaries) of `filters`, `kernel`, `pool` and `act`, then dense layers as `.qs.init`. Each input row holds its channels one after another. |
| `.qs.initGru[channels;length;block;hidden;sizes;acts;seed]` | New model with a GRU front end, then dense layers. |
| `.qs.assemble[inShape;layers;params;seed]` | Any other stack, from a list of `(kind;spec)` pairs (see below); `params` is `()` to initialise them, or one list of arrays per layer. |
| `.qs.fromWeights[W;B;acts]` | A dense model from given weights and biases. |
| `.qs.predict[m;X]` | Outputs for a matrix with one row per example, or for a single example vector. |
| `.qs.train[m;data;opts]` | Trains on `data` (`` `X`Y ``, optionally `` `Xval`Yval ``). Returns `` `model`history`bestEpoch ``. |
| `.qs.evaluate[m;X;Y;loss]` | `` `loss`acc`mae`rmse`r2 `` dictionary (`acc` is null for numeric targets). |
| `.qs.confusion[m;X;Y]` | Confusion matrix. |
| `.qs.grad[m;X;Y;loss]` | Gradient of the loss: one list of arrays per layer, shaped like `m`params` (`.qs.gradDrop` adds a dropout rate). |
| `.qs.search[d;space;base;n]` | Trains one model per configuration and returns a table ranked by validation loss. `space` maps options (and `hidden`, `act`) to candidate values, `base` fixes the rest, `n` is `0N` for the whole grid or a number to sample. |
| `.qs.crossValidate[d;k;base]` | k-fold cross-validation of one configuration: `` `folds`summary `` (per-fold scores; mean and sd). |
| `.qs.summary m` | One row per layer that has parameters: inputs, outputs, activation, parameters. |
| `.qs.check m` | Signals unless the layers, shapes and parameters fit together; returns the model. |
| `.qs.gradCheck[kind;spec;inShape]` | Largest difference between a layer's backward pass and finite differences. |
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

A model is a dictionary `` `layers`params `` (plus any extra keys you add, such as a table encoding). `layers` is a
list of `(kind;spec)` pairs, one per layer, each spec recording the shapes the layer takes and gives; `params` holds
each layer's arrays, in the order of `.qs.layer.names[kind]`. Between layers, data is a batch matrix (shape `,d`) or,
inside a conv stack, a sequence with one vector per channel (shape `(c;t)`).

| Layer kind | What it does | Parameters |
|---|---|---|
| `dense` | fully connected layer with `n` outputs | `W`, `b` |
| `act` | activation `f`: `` `sigmoid`tanh`relu`linear`softmax `` | |
| `dropout` | zeroes each value with probability `rate` while training | |
| `seq` | reads a flat batch as `channels` × `length` | |
| `conv1d` | 1-D convolution (no padding, stride 1) with `filters` and `kernel` | `K`, `b` |
| `maxpool1d` | maximum over blocks of `pool` steps | |
| `flatten` | a sequence to a flat row, step by step | |
| `gru` | gated recurrent unit over a flat input (`channels`, `length`, `block`, `hidden`) | `W`, `U`, `b` |

A new layer type is one call to `.qs.layer.add[kind;parts]`, giving its shape rule, initialiser, forward and backward
passes, parameter check and a few examples. `q tests/run.q` then checks its gradient against finite differences along
with every other type (`.qs.gradCheck`). The plan for the types still to come is in
[plans/inProgress/model_expansion_plan.md](plans/inProgress/model_expansion_plan.md).

Models saved before this layout (keys `weight`, `bias`, `act`, and `conv` or `gru` front ends) are the same network:
`.qs.load` and the public functions convert them, and they give identical outputs.

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

The runner also loads each project's library and runs its tests (`projects/*/tests/test_*.q`).

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
README.md                   this file: the library
src/                        the library, loaded by everything else
  defineEquation.q          core: layers, conv and GRU front ends, training, search (.qs)
  data.q                    datasets, tables, CSV, time series (.qs.data)
  plot.q                    decision-boundary SVGs (.qs.plot2D)
  signal.q                  FFT, PSD, whitening, filtering, matched filters (.qs.sig)
app/                        the programs (run them from the repository root, e.g. q app/dashboard.q -s 4)
  train.q                   command-line training
  serve.q                   IPC model server
  dashboard.q               web dashboard server
  ui/                       the dashboard's page
  kx/qsolve.json            a KX Dashboards connection to the dashboard
bench/benchmarks.q          benchmark suite
scripts/fetch_data.sh       dataset downloader
tests/                      test runner, assertion helpers, the library's tests, the dashboard's browser check
projects/                   real-world tests, one folder each, laid out alike:
  gw/                       gravitational waves in LIGO data (.qs.gw)
  markets/                  market studies: exchange-rate forecasting (.qs.fx)
    README.md               the write-up: method, results, how to reproduce
    src/                    the project's library, on top of src/ (each file x.q is namespace .qs.x)
    experiments/            the scripts that produced the results
    scripts/                downloads and batch jobs
    tests/                  the project's tests, run by tests/run.q
    plans/                  the project's plans (future/, inProgress/, completed/)
    data/ results/ models/  downloads, measured results, trained models (git-ignored)
plans/                      plans for the library itself (future/, inProgress/, completed/); see Plans above
data/  models/  .venv/      the library's datasets and saved models, the Python venv for GW HDF5 (git-ignored)
```
