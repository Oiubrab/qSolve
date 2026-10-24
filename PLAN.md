# qSolve improvement plan

This is the working plan for reviving qSolve, a small feed-forward neural network written in pure q/kdb+.
It is written so that any engineer or LLM session can pick it up cold, see what is done, and continue.

**How to use this file**

- Work top to bottom. Phases depend on earlier phases; items inside a phase are mostly independent.
- Tick an item (`- [x]`) only when its acceptance criterion is met *and* `q tests/run.q` passes.
- When you tick an item, add a dated line to the [Progress log](#progress-log) at the bottom saying what
  changed and anything surprising. Keep the log append-only.
- If you change a design decision, edit the [Design decisions](#design-decisions) section and say why in the log.
- Do not commit or push unless the repository owner asks you to.

**Current status (2026-10-24):** Phases 0-7 are done (79 tests, 14 benchmarks, browser check pass); Phases
0-7 are committed on main. **Phase 8 (a clearer Predict tab) is in
progress**: continue with its first unticked item. Start with `q tests/run.q` and read the Progress log.

---

## 1. Context

### What the project is

A teaching-scale multilayer perceptron (MLP) in q: random weight initialisation, forward pass,
hand-derived backpropagation, mini-batch gradient descent, and an MNIST experiment.
The owner wants it to be "way more robust and way more useful". There is **no specific target problem
yet**: the tool should stay general, and the test-data ladder (section 5) is how we prove it works. When a
real problem turns up, it should be possible to plug it in through `.qs.data` without touching the core.

### State of the repo when this plan was written (commit `32392c7`)

| File | What it was |
|---|---|
| `defineEquation.q` | The whole library (~150 lines, global namespace): `sigmoid`, `mSig` (sigmoid'), `linear`, `mGoras` (1/‖e‖), `useModel` (forward), `grad` + `gradBuild` (backprop), `backPropogation` (training loop), `weightBiasGen` (init), `indexGen` (index list for every weight). |
| `test.q` | Not a test: a training script. `$[0b; mnist-branch; 4-bit-pattern-branch]` picks the experiment by hand-editing. |
| `mnist_data_pull.py3` | Uses TensorFlow to download MNIST and writes 70,000 text files into `eternalnightmare/` and `byinheritance/`, which `test.q` parses back with fixed-width reads. |

Model representation (kept, see decisions): a dictionary `` `weight`bias!(W;B) `` where `W[l]` is an
`n_l × n_(l-1)` float matrix (rows = nodes of layer `l`, columns = inputs from the previous layer) and `B[l]`
is a length-`n_l` vector. A forward step is `sigmoid B[l] + W[l] mmu a`.

### Findings from the review (with evidence)

1. **The backprop maths is correct.** A finite-difference check against `gradBuild` on 2-, 3-, 4- and 5-layer
   nets (which exercise every branch of `grad`) gave max |numeric − analytic| ≈ 1e-7 for weights and biases.
2. **It is ~700× slower than it needs to be.** `grad` computes each weight's partial derivative independently
   with scalar chain-rule expressions over an index list built by `indexGen`. A matrix (vectorised) backprop
   computing the same loss agreed to 5e-17 and timed:
   - 784→32→10, batch 20: ~1,000 ms (old) vs ~1.6 ms (vectorised)
   - 784→784→10, batch 20 (the config in `test.q`): 20 ms vectorised; the old code would take over an hour for
     one pass over 6,000 examples.
3. **`mSig` overflows.** `exp[x]%xexp[1+exp[x];2]` gives `0w%0w = 0n` (NaN) for x ≳ 710. Use `s*(1-s)`.
4. **Initialisation is poor.** `gen:{(0.1 + 0.1*x?10)%x}`: every weight is positive and takes one of only ten
   values. Biases are initialised the same way. Should be symmetric (Xavier/He) with zero biases.
5. **Unusual loss.** The gradient implies loss = ‖all errors in the batch‖ (Euclidean norm, not squared).
   Its gradient `e/‖e‖` always has unit length, so the step size never shrinks as the model improves.
   Standard choices are MSE (regression), softmax + cross-entropy (multi-class), sigmoid + binary
   cross-entropy (binary/multi-label).
6. **Training loop is thin:** one pass only (no epochs, no shuffling); drops the remainder batch; evaluates on a
   different small slice of the test set each step; prints the entire gradient and model every batch (600k+
   numbers for MNIST); no accuracy metric; uses a global `oldTimer`; spelled `backPropogation`; always writes the
   model to `./model`.
7. **No tests.** Nothing asserts anything.
8. **Heavy data pipeline:** TensorFlow dependency, 70k text files. MNIST's IDX format is trivial binary that q
   can read directly with `read1`.
9. **The 4-bit pattern task in `test.q` is imbalanced:** ~87% of examples have target `0 0`, so a model that
   always outputs `0 0` looks good on mean absolute error. Accuracy must be reported per class.

---

## 2. Goals

1. **Correct and verified**: every gradient is checked against finite differences in an automated test.
2. **Fast enough to be useful**: MNIST 784→128→10 trains an epoch in well under a minute on a laptop.
3. **General**: choice of activations per layer, loss functions, optimiser, batch size, epochs, seed.
4. **Usable**: one namespace, a clean API (`init`, `predict`, `train`, `evaluate`, `save`, `load`), a CLI
   training script, and a README with examples.
5. **Good test data**: a ladder of datasets from trivial to MNIST, each with a target result, all runnable
   from q with no Python.

Non-goals: GPUs, convolutions, autodiff, or competing with real ML frameworks.

---

## 3. Design decisions

These are settled unless the log says otherwise.

| # | Decision | Reason |
|---|---|---|
| D1 | The library stays in `defineEquation.q` and everything lives in the `.qs` namespace. Sub-namespaces: `.qs.act` (activations), `.qs.loss` (losses), `.qs.data` (datasets). | Keeps the file the owner knows; namespaces stop global pollution. |
| D2 | Model = dictionary `` `weight`bias`act!(W;B;acts) ``. `W[l]` is `n_l × n_(l-1)`, `B[l]` length `n_l`, `acts` a symbol per layer (e.g. `` `relu`softmax ``). Gradients are `` `weight`bias!(gW;gB) `` with the same shapes. | Same layout as the original, plus the activation names. Weights and biases can be updated with dictionary arithmetic on `` model`weight`bias ``. |
| D3 | Batches are **matrices**: `X` is `n × nIn` (one row per example), `Y` is `n × nOut`. Forward: `Z[l] = (A[l-1] mmu flip W[l]) +\: B[l]`. | Vectorised maths; one `mmu` per layer per batch. |
| D4 | Backprop: `δ_L = ∂L/∂A_L · f'(Z_L)` (with a shortcut `δ_L = (A−Y)/n` for softmax+xent and sigmoid+bce); `δ_l = (δ_(l+1) mmu W[l+1]) · f'(Z_l)`; `gW[l] = flip[δ_l] mmu A[l-1]`; `gB[l] = sum δ_l`. | Standard; matches the original to 5e-17. |
| D5 | Losses average over the batch (`mse` = mean over examples of ½Σ(a−y)²). The original loss is kept as `norm` for comparison. | Learning rate then doesn't depend on batch size. |
| D6 | Activations: `sigmoid`, `tanh`, `relu`, `linear`, `softmax` (softmax only on the output layer and only with `xent`). Each activation has a forward function and a derivative written in terms of `(z;a)`. | Covers the classic MLP set without needing a full Jacobian. |
| D7 | Init: Xavier-uniform for sigmoid/tanh/linear/softmax layers, He-uniform for relu layers, biases zero. Randomness is reproducible through a `seed` option (`system "S ",string seed`). | Standard, symmetric, reproducible. |
| D8 | Training returns `` `model`history `` where `history` is a table (`epoch`, `trainLoss`, `valLoss`, `valAcc`, `seconds`). Logging is one line per epoch, controlled by `verbose`. | Tables are natural in q; quiet by default. |
| D9 | Options are a dictionary merged over defaults: `.qs.defaults ^ opts` style (implemented with `,`). | Callers pass only what they change. |
| D10 | Tests live in `tests/`, use a small assertion helper (`tests/assert.q`), and run with `q tests/run.q` which exits 0 on pass, 1 on failure. Tests must not need network access or downloaded data; tests that need MNIST/Iris skip themselves if the files are absent. | CI-friendly; quick to run. |
| D11 | Downloaded data goes in `data/` (git-ignored), fetched by `scripts/fetch_data.sh` (curl + gunzip). | No Python/TensorFlow needed. |
| D12 | Legacy files (`test.q`, `mnist_data_pull.py3`) are removed once their replacements work. The old `grad`/`indexGen` are removed once the vectorised version passes the gradient check; git history keeps them. | One way to do each thing. |

### q gotchas met so far

- The script must be q's first argument: `q tests/run.q -s 4`, not `q -s 4 tests/run.q` (that loads nothing
  and waits on stdin, which looks like a hang).
- A fully applied lambda `{[a;b] ...}[x;y]` runs immediately; to pass it as a callback give it a spare
  argument so it stays a projection.
- `system "q ..."` runs q's own `\q` (quit), not a shell command; likewise any string whose first word is a q
  system command (`cd`, `c`, `e`, `l` ...). Use `system "env q ..."`.
- Single-letter command-line flags belong to q (`-n 2000` fails with `n invalid`); script options need
  longer names (`-rows`).
- Symbol literals can't contain `-`: `` `fill-opacity `` is `` `fill `` minus `opacity`. Use `` `$"fill-opacity" ``.
- `in` and `~` compare types strictly: `0f in 0,1.5` is false because `0,1.5` is a mixed list; write `0f,1.5`.
- `sum` of booleans is an int (`6401i`), not a long; cast with `"j"$` before `.t.eq` against a long.
- `'x 1` signals `x` (then fails), not `x 1`: assign the message first, `e:x 1; 'e`.
- In this shell `cp` and `rm` are aliased to interactive/trash versions: use `command cp -f`, `command rm`.
- `.j.k` turns a JSON list of one-character strings into one string: `["t","v"]` -> `"tv"`. The dashboard's
  `.dash.syms` handles it.
- `.j.j` writes infinities as `inf` (invalid JSON): `.dash.clean` replaces them with nulls before sending.
- `.Q.trp`'s error handler takes two arguments (message, backtrace); a one-argument one fails with `rank`.
- `get`, `distinct` and the like are keywords even as local names or table column names (`'assign`).
- `pkill -f pattern` from this shell can match the shell's own command line and kill it (exit 144): find
  q processes with `pgrep -x q` and check their arguments instead.
- Multi-line lambdas don't work when q code is piped in on stdin; put experiments in a script file.
- This licence allows at most 4 secondary threads (`-s 4`).

- `exp` is a reserved word: don't use it as a variable name (it gave `'assign`). Same for `neg`, `type`, `max` …
- q evaluates right to left: `(count w) - 2 + i` means `count[w] - (2+i)`.
- `x?1f` gives uniform floats in [0,1); `x?10` gives longs 0–9.

---

## 4. Target layout

```
defineEquation.q        core library (.qs namespace)
data.q                  dataset generators and loaders (.qs.data)
plot.q                  decision-boundary SVGs (.qs.plot2D)
train.q                 command-line training script
serve.q                 IPC model server (predict, predictTable, info)
dashboard.q             web dashboard: HTTP for ui/, JSON over WebSocket, q IPC for KX Dashboards
ui/                     the dashboard page: index.html, app.js, style.css (no external libraries)
dashboards/qsolve.json  KX Dashboards connection for a dashboard on port 8080
tests/ui/check.js       browser check of the dashboard (headless Chromium over DevTools; not in run.q)
models/                 models saved from the dashboard (git-ignored)
tests/assert.q          assertion helpers (.t namespace)
tests/run.q             test runner: loads library, runs every tests/test_*.q, exits non-zero on failure
tests/test_*.q          test files
bench/benchmarks.q      trains each benchmark dataset and checks it hits its target
scripts/fetch_data.sh   downloads MNIST / Fashion-MNIST / Iris into data/
data/                   downloaded data (git-ignored)
PLAN.md                 this file
README.md               user documentation
```

### Public API

```q
m:.qs.init[nIn; layerSizes; acts; seed]     / e.g. .qs.init[784; 128 10; `relu`softmax; 42]
a:.qs.predict[m; X]                          / X: n×nIn matrix (or a single vector) -> outputs
g:.qs.grad[m; X; Y; lossName]                / `weight`bias!(gW;gB)
g:.qs.gradDrop[m; X; Y; lossName; p]         / the same with dropout rate p on hidden layers
r:.qs.train[m; data; opts]                   / data: `X`Y`Xval`Yval dict; returns `model`history`bestEpoch
.qs.evaluate[m; X; Y; lossName]              / `loss`acc dict (acc only for one-hot Y)
.qs.confusion[m; X; Y]                       / rows true class, columns predicted
.qs.save[`:file; m]   m:.qs.load `:file
.qs.plot2D[m; data; `:file.svg; title]       / decision boundary of a 2-input model (plot.q)
```

`opts` keys (defaults in brackets; the source of truth is `.qs.defaults`): `epochs` [10], `batch` [32],
`lr` [0.1], `loss` [`mse] (`mse`bce`xent`norm), `optimiser` [`sgd] (`sgd`momentum`adam), `momentum` [0.9],
`beta1` [0.9], `beta2` [0.999], `l2` [0], `dropout` [0], `schedule` [`none] (`none`step`cosine), `decay`
[0.5], `stepEvery` [10], `minLr` [0], `patience` [0N = never stop], `keepBest` [0b], `shuffle` [1b], `seed`
[42], `verbose` [1 = log every epoch; 0 silent; n every n epochs].

---

## 5. Test data ladder

Each dataset has a target that the benchmark script checks. Targets are deliberately achievable, not
state of the art.

| # | Dataset | Source | Checks | Suggested config | Target |
|---|---|---|---|---|---|
| T1 | Random tiny nets + finite differences | generated | gradient maths for every activation × loss | any | max abs error < 1e-5 |
| T2 | Hand-computed 2-2-1 net | fixed weights | forward pass | — | exact to 1e-12 |
| T3 | AND, OR | 4 rows | single layer learns linearly separable | `2→1 sigmoid`, bce | 100% |
| T4 | XOR | 4 rows | hidden layer needed | `2→4→1 tanh,sigmoid`, bce | 100% |
| T5 | 4-bit pattern (legacy task) | generated: 1100→class A, 0011→class B, else neither | imbalanced classes | `4→8→3 relu,softmax`, xent | 100% per class |
| T6 | sin(x) | generated, x∈[0,2π], y scaled to [0,1] | regression | `1→16→1 tanh,linear`, mse | MSE < 0.002 |
| T7 | Two concentric circles | generated, 2-D | curved boundary | `2→8→2 tanh,softmax` | ≥ 98% |
| T8 | Two moons | generated, 2-D | curved boundary | `2→16→2 tanh,softmax` | ≥ 97% |
| T9 | Two spirals | generated, 2-D | hard non-linear boundary | `2→32→32→2 tanh,tanh,softmax` | ≥ 95% |
| T9b | houses | generated table: numeric, categorical and missing columns | regression through `fromTable` | `7→32→1 tanh,linear`, mse | R² ≥ 0.95 |
| T9c | seasonal | generated series: trend + cycle (period 24) + AR(1) noise, 48 lags, 1 step ahead | forecasting, time-ordered split | `47→16→1 tanh,linear`, mse | R² of the change ≥ 0.85; MAE beats seasonal naive |
| T10 | Iris | UCI CSV, 150 rows, 4 features, 3 classes | first real dataset | `4→8→3 tanh,softmax` | ≥ 93% val |
| T11 | MNIST | IDX files | full scale | `784→128→10 relu,softmax`, xent | ≥ 96% test |
| T12 | Fashion-MNIST | IDX files (same format) | harder drop-in | same as T11 | ≥ 84% test |

---

## 6. Phased checklist

### Phase 0 — Safety net

- [x] **0.1 Test harness.** `tests/assert.q` (helpers: `.t.ok`, `.t.eq`, `.t.near[a;b;tol]`, `.t.throws`)
  and `tests/run.q` (loads the library, loads each `tests/test_*.q`, runs every registered test inside a
  trap, prints a pass/fail table, `exit 1` on any failure).
  *Accept:* `q tests/run.q` runs and exits 0 with a trivial test; exits 1 if a test is made to fail.
- [x] **0.2 Gradient check test** (T1) against the *original* implementation, proving the harness catches
  real things. *Accept:* passes on the original `gradBuild`.

### Phase 1 — Core rewrite

- [x] **1.1 Namespace + model format** (D1, D2). Move to `.qs`, model carries `act`.
- [x] **1.2 Activations** (D6) in `.qs.act`, numerically safe (`sigmoid'` = `a*1-a`; softmax subtracts
  row max before `exp`). *Accept:* unit tests for values and derivatives (derivative vs finite difference).
- [x] **1.3 Losses** (D5) in `.qs.loss`: `mse`, `bce`, `xent`, `norm`. Each has value and ∂L/∂A; clip
  probabilities to [1e-12, 1−1e-12] in log losses. *Accept:* unit tests incl. finite-difference on ∂L/∂A.
- [x] **1.4 Vectorised forward + backprop** (D3, D4): `.qs.forward`, `.qs.predict`, `.qs.grad`.
  *Accept:* T1 passes for every activation × loss combination on 2-, 3- and 4-layer nets; T2 passes.
- [x] **1.5 Initialisation** (D7): `.qs.init`, Xavier/He, zero biases, seeded. *Accept:* tests for shapes,
  zero-mean-ish weights, same seed → same model.
- [x] **1.6 Remove old code**: `grad`, `gradBuild`, `indexGen`, `useModel`, `weightBiasGen`, `mSig`,
  `mGoras`, `linear`, `backPropogation`. Point the T1 test at the new code.

### Phase 2 — Training loop

- [x] **2.1 `.qs.train`** (D8, D9): epochs, shuffling, mini-batches including the remainder batch,
  validation after each epoch, history table, one log line per epoch.
- [x] **2.2 Metrics**: `.qs.evaluate` returns loss and accuracy (argmax; for single-output models,
  threshold 0.5). `.qs.confusion[m;X;Y]` returns a confusion matrix.
- [x] **2.3 Optimisers**: `sgd`, `momentum`, `adam`; optional L2 weight decay.
- [x] **2.4 Save / load**: `.qs.save`, `.qs.load` (q `set`/`get`). *Accept:* round-trip test.
- [x] **2.5 Learning tests**: T3 and T4 as automated tests (small, fast, seeded).

### Phase 3 — Data

- [x] **3.1 Synthetic generators** in `data.q`: `gate[`and`or`xor`nand]`, `fourBit`, `sine`, `circles`,
  `moons`, `spirals`, plus helpers `oneHot`, `split` (train/val), `standardise`.
  *Accept:* shape/range tests for each.
- [x] **3.2 Fetch script** `scripts/fetch_data.sh` for MNIST, Fashion-MNIST, Iris into `data/`; add
  `data/` to `.gitignore`.
- [x] **3.3 IDX reader** `.qs.data.idx` and `.qs.data.mnist[dir]` (returns `X`Y`Xval`Yval, pixels /255,
  one-hot labels). *Accept:* test on a small IDX file written by the test itself; real-data check if present.
- [x] **3.4 Iris loader** `.qs.data.iris[file]` (standardised features, one-hot labels, seeded split).
- [x] **3.5 Remove** `mnist_data_pull.py3` and the old text-file loaders.

### Phase 4 — Usability

- [x] **4.1 `train.q` CLI**: e.g.
  `q train.q -data moons -hidden 16 -act tanh -epochs 200 -lr 0.1 -save model.qsm` (output size, output
  activation and loss are inferred from the data unless given with `-out` / `-loss`; see the header of `train.q`).
  Replaces `test.q` (deleted).
- [x] **4.2 Benchmarks** `bench/benchmarks.q`: runs T3–T12 with the suggested configs (skipping datasets not
  downloaded), prints a results table, marks each pass/fail against its target. Record the results in
  README.
- [x] **4.3 README**: what it is, quick start, API, datasets, how to run tests and benchmarks.

### Phase 5 — Later / optional

- [x] 5.1 Early stopping on validation loss; keep best model.
- [x] 5.2 Learning-rate schedule (step or cosine).
- [x] 5.3 Use `peach` over batch shards for large batches (measure first: `mmu` may already be multi-threaded).
- [x] 5.4 Dropout.
- [x] 5.5 Plotting the decision boundary of 2-D datasets (done as SVG rather than PGM/PPM: plain text, opens
  in any browser).

### Phase 6 — Next directions (owner chose to do all of them, in order)

There is still no specific target problem, so these are ranked by how much more *generally useful* each
makes qSolve. Recommended order: 6.1 first (it lets any real question in), then 6.2 (it plays to kdb+'s
strength), then the rest as wanted.

- [x] **6.1 Train from any kdb+ table or CSV.** `.qs.data.fromTable[t;features;target]`: numeric columns
  standardised, symbol columns one-hot encoded, nulls filled (and reported), target one-hot for symbols or
  left numeric for regression; returns the usual `` `X`Y `` plus the encoding, so new rows can be encoded the
  same way at prediction time (`.qs.data.encode[enc;t]`). `.qs.data.csv[file;types;target]` on top. Add
  regression metrics (MAE, RMSE, R²) to `.qs.evaluate` so regression gets something better than accuracy.
  *Accept:* Iris and a regression CSV (e.g. a generated one) train through it; encode/train/predict
  round-trip test.
- [x] **6.2 Time-series windows.** `.qs.data.windows[series;lags;horizon]` turns a series (or table columns)
  into lagged-feature rows, with a time-ordered (not shuffled) train/validation split. Benchmark on a
  generated noisy seasonal series against a naive "last value" baseline.
- [x] **6.3 Hyperparameter search.** Grid or random search over options and layer sizes, scored on
  validation loss, returning a results table (q is good at this); optionally `peach` across configs.
- [x] **6.4 Serve a model over IPC.** A small script that loads a saved model and answers `predict`
  requests on a port (`.z.pg`), with input checks. kdb+'s natural deployment path.
- [x] **6.5 k-fold cross-validation** for small datasets, reporting mean and spread of validation scores.
- [x] **6.6 Model summary**: layer table (inputs, outputs, activation, parameter count) for logs and the CLI.


### Phase 7 — Dashboard (owner asked for a usable UI)

The owner asked for better usability and "some kind of dashboard". KX Dashboards is installed at
`~/.kx/dashboards` (a drag-and-drop builder that queries q processes over IPC). Decision: build qSolve's
own web dashboard, served by q itself (HTTP + WebSocket on one port, no other dependencies), because it can
drive training interactively and stream progress, lives in the repo and can be tested; and let the same
process act as a KX Dashboards data source for anyone who wants to build their own views.

Design: `dashboard.q` loads the library, serves files from `ui/` over HTTP (`.z.ph`; only files in `ui/`, no
path traversal), and speaks JSON over a WebSocket (`.z.ws`): each message is `{"cmd":...}` and the reply
messages carry a `type`. Training runs inside the WebSocket handler (q is single-threaded, so the UI waits,
but each epoch is pushed to the browser as it finishes through the `onEpoch` callback). Runs, their
histories and models are kept in memory for the session; models can be saved to `models/` (git-ignored).
Binds to 127.0.0.1 by default.

- [x] **7.1 Progress callback.** `onEpoch` training option: a function called with each history row (default
  does nothing). *Accept:* test that it is called once per epoch with the row.
- [x] **7.2 `dashboard.q` server.** HTTP static files from `ui/`; WebSocket JSON API with commands `hello`
  (datasets with availability, saved models, runs, defaults), `upload` (CSV text -> table kept in memory;
  returns columns, types, preview), `train` (any dataset including uploads, as table or time series; streams
  `epoch` messages then `done` with metrics, history, confusion matrix, decision-boundary SVG for 2-D data,
  layer summary), `search` (streams one `trial` per configuration), `predict` (rows for table models; points
  for 2-D models), `save` / `load` / `models`, `forget`. Errors come back as `{"type":"error"}`, never kill
  the server. *Accept:* API tests call the handler directly with a stub sender; HTTP tests via a spawned
  process and curl, including a refused traversal path.
- [x] **7.3 UI** (`ui/index.html`, `ui/app.js`, `ui/style.css`; no external libraries, charts are
  hand-written SVG): Train view (dataset picker incl. CSV upload with target/series selection, layers,
  options with sensible defaults, live loss/accuracy charts, metric tiles, confusion heatmap, decision
  boundary, layer table), Runs view (compare runs' validation curves and metrics), Search view (grid input,
  streamed results table, train-the-best button), Predict view (form or pasted CSV for table models), Models
  view (save/load). Light and dark themes. *Accept:* checked in a real browser (headless Chromium
  screenshots) at desktop and narrow widths.
- [x] **7.4 KX Dashboards data source.** The dashboard process also answers q IPC queries from localhost
  (tables `.dash.runs`, `.dash.history[]`), plus a connection file and README steps for adding it in KX
  Dashboards.
- [x] **7.5 Docs and launcher.** README section, `q dashboard.q -port 8080` printing the URL.

### Phase 8 — A Predict tab that explains itself (owner feedback)

The owner trained spirals, opened Predict, and got "2 input values, comma-separated" and "class 0, 100%"
with no link to what was trained. Each model needs a description of its inputs and outputs in the training
data's own terms, an input method that fits the data, and real examples to compare against.

- [ ] **8.1 Model schema.** For every run (and saved model) the server describes the model: dataset and
  description, task, input kind (`table`, `points` for 2-D, `image`, `series`, `numbers`), input names with
  training ranges, class names (built-ins named, e.g. moons "upper moon"/"lower moon", Fashion-MNIST item
  names; tables and uploads from the encoding), target name, validation score. Saved models carry it.
- [ ] **8.2 Examples.** `example` command: a random validation row in raw terms (table row, point, image
  pixels, or the window of recent series values) with its true answer. Needs the raw validation rows: table
  data keeps them (`Tval`), windows keep the series and validation times, and the column scaling stats.
- [ ] **8.3 Series forecasting from recent values.** Predict takes `recent` raw values for series models and
  returns the forecast in the series' units (`.qs.data.windowRow`).
- [ ] **8.4 UI.** "About this model" card; inputs by kind (named fields with ranges; clickable boundary map
  for 2-D; 28x28 drawing pad for images, centred like MNIST; recent values for series, prefilled from the
  data; the table form); results show the answer by name, probability bars per class, the true answer for
  examples, and a warning for inputs outside the training range. *Accept:* browser check covers spirals,
  iris, houses, MNIST (when downloaded) and a series upload.

---

## 7. How to run

```bash
q tests/run.q                       # all tests, exits non-zero on failure
q bench/benchmarks.q                # train every benchmark dataset and check targets
scripts/fetch_data.sh               # download MNIST, Fashion-MNIST, Iris into data/
q train.q -data xor -hidden 4 -epochs 300 -batch 4 -lr 0.5
q train.q -s 4 -data mnist -hidden 128 -act relu -optimiser adam -lr 0.001 -epochs 5 -batch 64 -plot x.svg
```

Environment used so far: q 5.0 (2026.05.01), Linux, 16 cores.

---

## Progress log

- 2026-10-24: Plan written after reviewing commit `32392c7`. Findings 1–2 were measured with scratch
  scripts (finite-difference check; vectorised backprop prototype), not yet in the repo.
- 2026-10-24: Owner confirmed there is no specific target problem; the tool is meant to be general. Added to Context.
- 2026-10-24: 0.1 done. `tests/assert.q` (`.t.add/.t.ok/.t.eq/.t.near/.t.throws`), `tests/run.q` (supports
  `-only substring`), `tests/test_harness.q`. Verified the runner exits 1 when a test fails.
- 2026-10-24: 0.2 done. `tests/test_grad.q` checks the original `gradBuild` against central finite differences
  on 2/3/4-layer nets (error < 1e-6). Passes.
- 2026-10-24: Phase 1 done (1.1-1.6) in one rewrite of `defineEquation.q`. Tests: `tests/test_act.q`,
  `tests/test_loss.q`, `tests/test_model.q` (init, `.qs.check`, T2 hand-computed forward, save/load), and
  `tests/test_grad.q` now checks `.qs.grad` against finite differences for 84 cases (3 depths x 4 hidden
  activations x 7 output-activation/loss pairs), all < 1e-6. A deliberately broken tanh derivative is caught
  (error 0.046). Before deleting the old code, `.qs.grad` with all-sigmoid + `norm` loss matched the original
  `gradBuild` to 5.6e-17 on a 4-layer net. Notes:
  - q has no `tanh` builtin; it is `{1-2%1+exp 2*x}`.
  - `xent` assumes each target row sums to 1 (one-hot); the softmax+xent shortcut `(A-Y)/n` relies on it.
  - `cross` flattens nested items: build pairs with `{(x;y)}/:\:` instead.
  - `.qs.save`/`.qs.load` were written at the same time, so 2.4 is ticked too.
  - `test.q` still calls the deleted functions and is broken until 4.1 replaces it.
- 2026-10-24: Phase 2 done. `tests/test_train.q`: AND/OR with one sigmoid node, XOR with each of
  sgd/momentum/adam, a control showing a single layer *cannot* solve XOR, history columns, seeded determinism,
  remainder batch is used (adam step counter = 2 for batch 3 over 4 rows), l2 shrinks weights, input
  validation, metrics/confusion. Bug found and fixed on the way: `.qs.step` rebuilt the state dict and dropped
  `history`. `tests/run.q` now sets `\c 500 200` so the results table isn't truncated.
- 2026-10-24: Phase 3 done. `data.q` (`.qs.data`): `gate`, `fourBit` (now 3 one-hot classes: 1100, 0011,
  neither), `sine`, `circles`, `moons`, `spirals`, `oneHot`, `split`, `stats`/`scale`/`standardise`, `take`,
  `idx`, `mnist` (also reads Fashion-MNIST), `iris`. Logic gates are `gate[`xor]` etc. because `and`/`or` are
  q keywords. `scripts/fetch_data.sh` downloads into `data/` (git-ignored); MNIST from the Google
  `cvdf-datasets` mirror (yann.lecun.com often refuses), Fashion from Zalando's S3, Iris from UCI. All fetched
  fine. MNIST loads in ~0.4 s; first labels 5 0 4 1 9 2 1 3 1 4 as expected. `tests/test_data.q` covers every
  function, including an IDX round-trip with truncated and wrong-type files. Runner now has a `skip` status
  (`.t.skip msg`), used by the MNIST/Iris tests when `data/` is missing (checked by hiding `data/`).
  `mnist_data_pull.py3` deleted.
- 2026-10-24: 4.1 done. `train.q` CLI (options documented in its header) plus `.qs.data.load[name;n;noise]`
  and `.qs.data.names` in `data.q`, shared with the benchmarks. `-hidden none` gives a single-layer model.
  `tests/test_cli.q` runs `train.q` as a subprocess and reloads the saved model. `test.q` deleted.
  First full MNIST run: 784-128-10 relu/softmax, adam lr 0.001, batch 64, **1 epoch: 95.15% val in 5.4 s**.
  Gotcha found while writing the CLI test: q's `system` treats a command whose first word is a q system
  command as that command, so `system "q train.q ..."` runs `\q` (quits!) and `system "cd /x && ..."` runs
  `\cd`. Prefix with `env`. Also give the child `</dev/null` so it doesn't read the parent's stdin, and
  `hdel` needs a `:`-prefixed handle (`hsym`).
- 2026-10-24: 4.2 done. `bench/benchmarks.q` (`-only name ...`, `-skipLarge`) runs T3-T12; the case table is
  at the top of the file. First run, all 12 pass (seed 42, q 5.0, 16 cores):

  | benchmark | metric | result | target | epochs | seconds |
  |---|---|---|---|---|---|
  | and / or / nand / xor | val acc | 1.0 | 1.0 | 300 | <0.1 |
  | fourBit | val acc | 1.0 | 1.0 | 30 | 0.6 |
  | sine | val loss (½·MSE) | 1.2e-5 | 0.001 | 300 | 2.7 |
  | circles | val acc | 0.99 | 0.98 | 100 | 0.8 |
  | moons | val acc | 1.0 | 0.97 | 100 | 0.9 |
  | spirals | val acc | 0.9975 | 0.95 | 300 | 13.1 |
  | iris | val acc | 0.967 | 0.93 | 100 | 0.2 |
  | mnist | val (test-set) acc | 0.9754 | 0.96 | 5 | 62 |
  | fashion | val (test-set) acc | 0.8739 | 0.84 | 5 | 63 |

  Open observation for 5.3: MNIST took ~12 s/epoch here but 5.4 s for the single epoch run through `train.q`
  earlier. Not yet investigated (could be machine load; profile before optimising).
- 2026-10-24: 4.3 done. README rewritten: quick start, q API, options, data functions, how to bring your own
  data, tests/benchmarks, results table, layout. The README's q example was run end to end.
  Phases 0-4 complete. Final check: `q tests/run.q` 39 pass / 0 fail / 0 skip; benchmarks 12/12 pass.
- 2026-10-24: 5.3 done. Profiled first: the ~12 s/epoch MNIST figure from the first benchmark run did not
  reproduce (5.4 s/epoch serial: ~3.8 s training steps, ~1.6 s evaluating train+val sets), so it was machine
  load. `mmu` does not use secondary threads by itself (this licence allows at most 4: `-s 4`). Measured
  `peach` over row shards: predict on 60k rows 3.3x faster, batch-64 grad 1.9x, batch-512 grad 3x. Added
  `.qs.shards` (one shard per thread, each >= `.qs.minShard` = 32 rows) used by `.qs.predict` and `.qs.grad`;
  shard gradients are weighted by their share of rows; `norm` is never sharded (its gradient needs the whole
  batch). Old single-pass gradient is `.qs.grad1`. Test `grad_sharded` fakes 4 threads via `.qs.threads` and
  checks sharded == serial to 1e-15. MNIST benchmark: 27 s -> 18 s with `-s 4`, same 97.54%.
  Threaded runs can differ from serial ones in the last bits (summation order), still deterministic per
  thread count.
- 2026-10-24: Owner asked to commit Phases 0-4 to main (`20608e8`) and carry on with Phase 5.
- 2026-10-24: 5.4 done. `.qs.forwardMasked[m;X;M]` (returns Z, unmasked U, masked A), `.qs.dropMasks`
  (inverted dropout; output layer never masked), `.qs.gradDrop[m;X;Y;loss;p]`; `.qs.grad` is `gradDrop` with
  p=0, and `.qs.grad1` now takes the masks. Masks are drawn on the main thread before sharding, so a seeded
  dropout gradient is the same for any thread count (test `grad_dropout_sharded`). Test `grad_dropout`
  (finite differences with fixed masks) caught a real bug on its first run: an edit to the backward pass
  had silently not applied, so masks were ignored.
- 2026-10-24: 5.1 and 5.2 done. Options `schedule` (`none`step`cosine via `.qs.schedule`), `decay`,
  `stepEvery`, `minLr`, `patience`, `keepBest`, `dropout`; option checks moved to `.qs.checkTrain`.
  History gains an `lr` column; `.qs.train` returns `bestEpoch`. The monitored loss is validation loss when
  there is validation data, else training loss. Tests: schedule values, early stopping (validation labels
  are flipped so validation loss must rise), keepBest returns the best epoch's model, dropout still learns
  moons, bad options rejected. `train.q` takes all of these; `-n` renamed `-rows` (q owns `-n`); the final
  line now evaluates the *returned* model (it showed the last epoch's numbers even with `-keepBest`).
- 2026-10-24: 5.5 done. `plot.q`: `.qs.plot2D[m;d;file;title]` writes an SVG: 80x80 grid coloured by predicted
  class, paler where less certain; training points filled, validation hollow. Works with softmax or a single
  sigmoid output. Checked visually on spirals, moons and circles (rendered with rsvg-convert). `train.q
  -plot file.svg`. Tests check the SVG's structure.
- 2026-10-24: Phase 5 complete. 49 tests pass serial and with `-s 4`. Full benchmark with `-s 4`: 12/12
  pass in 46 s total, same numbers as before (MNIST 97.54% and Fashion 87.39% in ~19.6 s each).
  Added Phase 6 (proposed directions, not started; owner to choose).
- 2026-10-24: Owner asked to commit Phase 5 to main and carry on with all of Phase 6.
- 2026-10-24: 6.1 done. `data.q` tables section: `.qs.data.fit[t;features;target]` returns the encoding as
  a keyed table (`col`role`kind`mu`sd`levels`; `num` columns standardised with nulls -> mean, `cat` one-hot
  over training levels with nulls/unseen -> zeros; strings treated as symbols); `encodeX`, `encodeY`,
  `decodeY` (class symbols, or numbers in target units), `inputNames`; `fromTable[t;features;target;frac]`
  shuffles, splits, fits the encoding on the training rows only, and returns `X`Y`Xval`Yval`enc;
  `readCsv` (header, `.Q.id` column names, float if every non-empty value parses else symbol) and
  `csv[file;features;target;frac]`. `iris` now goes through `fromTable`. New generated regression table
  `.qs.data.houses n`. Core: `.qs.isClass`, `.qs.regression` (mae, rmse, r2); `.qs.evaluate` now returns
  `loss`acc`mae`rmse`r2` with acc null for regression targets; log lines omit null accuracy.
  `train.q -data csv -file f -target c [-features ...] [-frac]`; output activation now follows the targets
  (linear for regression) instead of special-casing `sine`; regression summary shows R2 and MAE in target
  units; `-save` stores the encoding in the model under `enc` (extra keys pass `.qs.check`). Benchmarks now
  score the *returned* model via `.qs.evaluate` (metric `acc`, `loss` or `r2`), plus new `houses`
  benchmark (R2 0.983 vs target 0.95). Tests: `tests/test_table.q` (fit, encode, unseen/missing levels,
  decode, fit-on-train-only, CSV inference, regression metrics, houses regression) and `cli_csv`. 57 pass.
  Gotcha: indexing a keyed table by a column name (`enc`role`) does a key lookup; use `(0!enc)`role` or
  `exec`.
- 2026-10-24: 6.2 done. `.qs.data.windows[s;lags;horizon;frac]`: rows are the lags-1 earlier values of
  each window and the value horizon steps on, all *relative to the window's last value* (first version used
  raw levels and lost to seasonal naive, MAE 0.158 vs 0.110, because the trend put validation outside the
  training range). Inputs standardised per column, target by its own mean/sd, all from training rows; split
  by time. Returns `level`/`levelVal` so `.qs.data.unwindow[d;A;levels]` gives predictions in series units.
  `.qs.data.naive[s;lags;horizon;frac;period]` gives last-value and seasonal-naive MAE in series units.
  Generator `.qs.data.seasonal[n;period;noise]`; dataset name `seasonal`. Result on 3000 points: model MAE
  0.068 vs seasonal naive 0.114 vs last value 0.209 (noise floor ~0.064). `train.q -data csv -file f
  -series col -lags k -horizon h` forecasts a CSV column; summary prints the last-value baseline.
  Benchmark `seasonal` (R² of the change, 0.883 vs target 0.85). Tests in `tests/test_series.q`
  (alignment, horizon, train-only scaling, naive, beats naive). 61 pass.
- 2026-10-24: 6.3 done. `.qs.search[d;space;base;n]` in `defineEquation.q`: `space` maps option names (plus
  `hidden` layer-size lists and hidden `act`) to candidates; `base` fixes the rest (out/loss inferred from the
  targets as in the CLI); `n` 0N = full grid, else that many distinct configurations sampled without
  replacement. Returns a table sorted by `valLoss` with `valAcc` (classification) or `valR2` (regression),
  `bestEpoch`, `seconds`. Configurations run serially: each training run already uses threads, and `\S`
  (seeding) from inside `peach` threads isn't something to rely on. `train.q -search k=v1,v2 ... [-trials n]`
  (layers joined by `x`, e.g. `hidden=8,32x32`) prints the table and trains the best configuration.
  Tests `tests/test_search.q` (grid, random without duplicates, best row reproducible, regression) and
  `cli_search`. Gotchas: a general (mixed-type) dictionary returns `::` for a missing key, so `42^c`seed`
  fails: use `(.qs.defaults,c)`seed`; and `n?m` samples *with* replacement, `neg[n]?m` without.
- 2026-10-24: 6.4 and 6.6 done. `serve.q -model f -p port`: `.z.pg` accepts only (`function;argument) for
  `predict` (shape-checked), `predictTable` (needs a model saved with `enc` by `train.q`; returns the table
  plus `prediction`, and `confidence` for classes; missing columns refused) and `info` (layers, input names,
  target, port, pid). Strings of q code are refused; `.z.ps` refuses async. `tests/test_serve.q` starts a
  real server process on a pid-derived port, polls until it opens, checks predictions match local ones and
  that bad calls are refused, then kills it by pid, also on failure (checked by breaking an assertion: no
  process left). `.qs.summary m` gives the layer table; `train.q` prints the parameter count.
  Bug hunt notes: the test first "failed" with `type` because `.Q.trp[{[h;m;data] ...}[h;m;data];...]`
  ran the fully applied lambda immediately (the projection gotcha again), and the error was hidden
  because `'res 1` signals `res`.
- 2026-10-24: 6.5 done. Factored `.qs.spec` (fill model keys, output activation and loss from the targets)
  and `.qs.fitScore` (train one configuration, score on validation) out of `.qs.search`; both used by
  `.qs.crossValidate[d;k;base]`, which pools any Xval/Yval into the rows, shuffles into k folds, and returns
  `folds` (per-fold scores) and `summary` (mean, sd). Iris, 5 folds: 97-98% +/- 2%. `train.q -cv K`.
  Tests `cross_validate` (fold partition incl. uneven folds, bad k) and `model_summary`. 69 tests pass.
- 2026-10-24: Phase 6 complete. Full benchmark with `-s 4`: 14/14 pass (adds houses R2 0.983 and seasonal
  R2-of-change 0.883 to the earlier 12, which are unchanged).
- 2026-10-24: Owner asked for better usability and a dashboard UI. Found KX Dashboards installed at
  `~/.kx/dashboards`. Added Phase 7 (own q-served web dashboard, plus KX Dashboards data-source hook).
- 2026-10-24: Owner asked for Phase 6 to be committed before Phase 7: done as `5cb7890` (Phase 7 edits were
  set aside and restored around the commit).
- 2026-10-24: 7.1 done. `onEpoch` option (default `{[r]}`) is called with each history row after it is
  appended; test `train_on_epoch`. Checked that q pushes WebSocket frames immediately from inside a busy
  `.z.ws` handler (`neg[h] msg; neg[h][]`): a client got 4 messages 400 ms apart while the handler worked,
  so live progress works without threads or a second process.
- 2026-10-24: 7.2 done. Refactors first: `.qs.grid` (search configurations, shared by `.qs.search` and the
  dashboard's streamed search), `.qs.plot.svg` / `.qs.plot.grid` (plot data without writing a file),
  `.qs.data.parseCsv` (CSV from text). `dashboard.q`: HTTP for files directly inside `ui/` (anything else
  404s, including `../` and encoded traversal), WebSocket JSON API (`hello datasets upload train search
  predict save load report forget runs`), replies carry the request's `id`; `train` streams `start`, `cv`,
  `epoch`..., then `done` with the report (scores, history, confusion, plot grid, fit, layers, input schema);
  `search` streams `trial`s. Errors return `{"type":"error"}` and never stop the server. Binds 127.0.0.1.
  Bugs met and fixed: `forget` used `where id=id` (local shadowed the column), `enlist[i]_` doesn't drop
  from a keyed table (`delete ... where`), `$[...]` condition was a vector for strings in `.dash.f`, a
  column named `distinct`, `.Q.trp` handler arity, one-letter column lists from `.j.k`.
  `tests/test_dashboard.q`: 9 tests through the handler with a stub sender, plus a real process checked
  over HTTP (curl) and q IPC.
- 2026-10-24: 7.3 done. `ui/`: Train, Runs, Search, Predict and Models views; hand-written SVG line charts
  with crosshair tooltips, canvas decision boundary (grid painted at n x n then scaled, so no seams),
  heatmap confusion matrix with text colour chosen by luminance, predicted-vs-actual scatter and
  forecast-vs-actual line for regression; light and dark tokens from the dataviz reference palette; phone
  layout. Checked in headless Chromium driven over the DevTools protocol; problems found that way and fixed:
  the browser's own form validation silently blocked Train (`step=100` on rows), the placeholder card ignored
  `hidden`, a resize redraw replaced the live run with the previous report, a queued live redraw overwrote
  the final tiles, dark-mode heatmap text was unreadable, the phone header overflowed. The driver is now
  `tests/ui/check.js` and passes.
- 2026-10-24: 7.4 and 7.5 done. The dashboard port answers q IPC (tested: `.dash.runs`, `.dash.history[]`);
  `dashboards/qsolve.json` is a KX Dashboards connection in the format of the installed samples. Not done:
  adding it to `~/.kx/dashboards` or building a Dashboards layout, since that changes the owner's install.
  README has a Dashboard section. 79 tests pass with and without threads.
- 2026-10-24: Owner found the Predict tab confusing (spirals: two unnamed numbers, "class 0"). Added
  Phase 8.
- 2026-10-24: Owner asked for Phase 7 to be committed before any Phase 8 changes.
