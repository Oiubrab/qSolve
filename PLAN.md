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

**Current status (2026-10-24):** Phases 0-4 are done; all tests and all 12 benchmarks pass. Phase 5
(optional extras) has not been started. A new session should begin with `q tests/run.q`, read the
Progress log, then pick from Phase 5 or from whatever real problem the owner brings. None of this work has
been committed yet: check `git status`.

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

- `system "q ..."` runs q's own `\q` (quit), not a shell command; likewise any string whose first word is a q
  system command (`cd`, `c`, `e`, `l` ...). Use `system "env q ..."`.

- `exp` is a reserved word: don't use it as a variable name (it gave `'assign`). Same for `neg`, `type`, `max` …
- q evaluates right to left: `(count w) - 2 + i` means `count[w] - (2+i)`.
- `x?1f` gives uniform floats in [0,1); `x?10` gives longs 0–9.

---

## 4. Target layout

```
defineEquation.q        core library (.qs namespace)
data.q                  dataset generators and loaders (.qs.data)
train.q                 command-line training script
tests/assert.q          assertion helpers (.t namespace)
tests/run.q             test runner: loads library, runs every tests/test_*.q, exits non-zero on failure
tests/test_*.q          test files
bench/benchmarks.q      trains each benchmark dataset and checks it hits its target
scripts/fetch_data.sh   downloads MNIST / Fashion-MNIST / Iris into data/
data/                   downloaded data (git-ignored)
PLAN.md                 this file
README.md               user documentation
```

### Public API (target)

```q
m:.qs.init[nIn; layerSizes; acts; seed]     / e.g. .qs.init[784; 128 10; `relu`softmax; 42]
a:.qs.predict[m; X]                          / X: n×nIn matrix (or a single vector) -> outputs
g:.qs.grad[m; X; Y; lossName]                / `weight`bias!(gW;gB)
r:.qs.train[m; data; opts]                   / data: `X`Y`Xval`Yval dict; returns `model`history
.qs.evaluate[m; X; Y; lossName]              / `loss`acc dict (acc only for one-hot Y)
.qs.save[`:file; m]   m:.qs.load `:file
```

`opts` keys (defaults in brackets): `epochs` [10], `batch` [32], `lr` [0.1], `loss` [`mse], `optimiser`
[`sgd] (`sgd`momentum`adam), `momentum` [0.9], `l2` [0], `shuffle` [1b], `seed` [42], `verbose` [1b].

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

- [ ] 5.1 Early stopping on validation loss; keep best model.
- [ ] 5.2 Learning-rate schedule (step or cosine).
- [ ] 5.3 Use `peach` over batch shards for large batches (measure first: `mmu` may already be multi-threaded).
- [ ] 5.4 Dropout.
- [ ] 5.5 Plotting the decision boundary of 2-D datasets (e.g. write a PGM/PPM image from q).

---

## 7. How to run

```bash
q tests/run.q                       # all tests, exits non-zero on failure
q bench/benchmarks.q                # train every benchmark dataset and check targets
scripts/fetch_data.sh               # download MNIST, Fashion-MNIST, Iris into data/
q train.q -data xor -layers 4 1 -act tanh sigmoid -loss bce -epochs 2000 -batch 4 -lr 0.5
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
