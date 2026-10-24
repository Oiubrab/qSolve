# qSolve

A small multilayer perceptron (feed-forward neural network) written in pure q/kdb+.
No Python and no dependencies beyond q itself (plus `curl` and `gunzip` to download the datasets).

- Layers with `sigmoid`, `tanh`, `relu`, `linear` or `softmax` activation
- Losses: mean squared error, binary and categorical cross-entropy
- Optimisers: SGD, momentum, Adam; optional L2 weight decay
- Mini-batches, shuffling, validation each epoch, history table, confusion matrix, save/load
- Vectorised backpropagation, checked against finite differences in the test suite
- Built-in datasets, from logic gates up to MNIST and Fashion-MNIST

MNIST with one hidden layer of 128 relu nodes reaches 97.5% test accuracy in 5 epochs (about a minute).

## Quick start

```bash
q train.q -data moons -hidden 16 -epochs 100 -optimiser adam -lr 0.01

scripts/fetch_data.sh                 # downloads MNIST, Fashion-MNIST and Iris into data/
q train.q -data mnist -hidden 128 -act relu -optimiser adam -lr 0.001 -epochs 5 -batch 64 -save mnist.qsm
```

`train.q` infers the output layer size from the data. It picks softmax with cross-entropy for several
outputs, sigmoid with binary cross-entropy for one output, and linear with MSE for `sine`. Its header
lists every option.

## Using it from q

```q
\l defineEquation.q
\l data.q

d:.qs.data.split[.qs.data.spirals[2000;0.02];0.8]      / `X`Y`Xval`Yval
m:.qs.init[2;32 32 2;`tanh`tanh`softmax;42]             / 2 inputs, two hidden layers, 2 outputs
r:.qs.train[m;d;`loss`optimiser`lr`epochs`verbose!(`xent;`adam;0.005;300;50)]
r`history                                               / one row per epoch
.qs.predict[r`model;0.1 0.2]                            / one example, or a matrix of rows
.qs.evaluate[r`model;d`Xval;d`Yval;`xent]               / `loss`acc
.qs.confusion[r`model;d`Xval;d`Yval]                    / rows true class, columns predicted
.qs.save[`:spirals.qsm;r`model]
```

### Core (`defineEquation.q`)

| Function | What it does |
|---|---|
| `.qs.init[nIn;sizes;acts;seed]` | New model. `sizes` gives one entry per layer, the last being the output. `acts` is one per layer, or a single activation for every layer. Xavier initialisation (He for relu), zero biases. A `seed` of `0N` leaves q's random state alone. |
| `.qs.predict[m;X]` | Outputs for a matrix with one row per example, or for a single example vector. |
| `.qs.train[m;data;opts]` | Trains on `data` (`` `X`Y ``, optionally `` `Xval`Yval ``). Returns `` `model`history ``. |
| `.qs.evaluate[m;X;Y;loss]` | `` `loss`acc `` dictionary. |
| `.qs.confusion[m;X;Y]` | Confusion matrix. |
| `.qs.grad[m;X;Y;loss]` | Gradient of the loss, shaped like the model. |
| `.qs.save[file;m]` / `.qs.load file` | Saves and loads a model (q's native format). |

`opts` is merged over `.qs.defaults`:

| Key | Default | |
|---|---|---|
| `epochs` | 10 | |
| `batch` | 32 | the last batch of an epoch may be smaller |
| `lr` | 0.1 | learning rate |
| `loss` | `` `mse `` | `` `mse`bce`xent`norm `` (`norm` is qSolve's original loss) |
| `optimiser` | `` `sgd `` | `` `sgd`momentum`adam `` |
| `momentum` | 0.9 | for `momentum` |
| `beta1`, `beta2` | 0.9, 0.999 | for `adam` |
| `l2` | 0 | weight decay |
| `shuffle` | `1b` | |
| `seed` | 42 | `0N` to leave q's random state alone |
| `verbose` | 1 | log every n epochs, 0 for silent |

A model is a dictionary `` `weight`bias`act!(W;B;acts) ``. `W[l]` is an `n_l × n_(l-1)` matrix, `B[l]` a vector,
and `acts` has one symbol per layer.

`xent` expects one-hot targets. Softmax is only allowed on the output layer, and only with `xent`.

### Data (`data.q`)

| Function | Dataset |
|---|---|
| `.qs.data.gate[`` `and`or`xor`nand ``]` | The four rows of a logic gate |
| `.qs.data.fourBit n` | 4 random bits, classified as 1100, 0011 or neither (the original qSolve task) |
| `.qs.data.sine n` | Regression of sin x |
| `.qs.data.circles[n;noise]`, `.moons`, `.spirals` | 2-D two-class problems |
| `.qs.data.iris file` | Iris, standardised, split 80/20 |
| `.qs.data.mnist dir` | MNIST or Fashion-MNIST (IDX files), test set as validation |
| `.qs.data.load[name;n;noise]` | Any of the above by name, already split |

Helpers: `oneHot`, `split`, `standardise`, `stats`/`scale`, `take`, `idx` (reads any unsigned-byte IDX file).
Generated datasets use q's random state, so call `system "S 42"` first for repeatable data.

To use your own data, build a dictionary `` `X`Y `` where `X` has one row per example and `Y` is one-hot (or a
single column), then call `.qs.data.split` and `.qs.data.standardise` and pass the result to `.qs.train`.

## Tests and benchmarks

```bash
q tests/run.q                    # unit tests; exits 1 on failure; -only <substring> to filter
q bench/benchmarks.q             # trains every benchmark dataset and checks its target; -skipLarge, -only <names>
```

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
| iris | 4→8→3 tanh, softmax | 96.7% | ≥ 93% |
| mnist | 784→128→10 relu, softmax, 5 epochs | 97.5% | ≥ 96% |
| fashion | 784→128→10 relu, softmax, 5 epochs | 87.4% | ≥ 84% |

## Layout

```
defineEquation.q        core library (.qs)
data.q                  datasets (.qs.data)
train.q                 command-line training
tests/                  test runner, assertion helpers, tests
bench/benchmarks.q      benchmark suite
scripts/fetch_data.sh   dataset downloader
PLAN.md                 development plan and progress log
```
