# Model expansion plan: from a fixed CNN/MLP to a general network toolkit

The owner, 2026-11-02: the network behind the GW work is a small CNN with about 35k parameters (three Conv1D layers
and two dense ones). The training system needs to grow, so qSolve can build modern architectures (deeper and residual
CNNs, sequence models, attention), and generative ones (autoencoders, GANs), still in pure q.

Read the "q gotchas" in [../completed/improvemet_plan.md](../completed/improvemet_plan.md) first.

**How to use this file:** as the other plans. Tick an item only when its check passes and `q tests/run.q`
passes; add a dated line to the Progress log; commit each finished stage to main (the owner's rule).

**Current status (2026-11-03):** in progress. **M1 (the general layer core) is done.** M2 onwards is next. The owner said
"get a start on the model expansion plan" without answering M-Q1 to M-Q3, so the recommended answers are assumed:
(a) better CNNs first after the core, a new model format with old files converted on load, and pure q. M1 doesn't depend on
M-Q1, and M-Q2 and M-Q3 are built in. M-Q1 only sets the order of M2, so the owner can still change it.

---

## 1. Where we are

What `src/defineEquation.q` can do now:

| Area | Have |
|---|---|
| Layers | dense; Conv1D (valid padding, stride 1, ReLU or other activation, then max pool) as a front end; one GRU as a front end |
| Activations | sigmoid, tanh, relu, linear, softmax |
| Losses | mse, mae, bce, xent, norm |
| Training | SGD, momentum, Adam; step and cosine schedules; L2; dropout (dense only); early stopping and keep-best; per-epoch data generators; threads over batch shards |
| Tools | save/load, summary, grid search, cross-validation, finite-difference gradient checks in the tests |

**The limit is structural, not just missing layers.** A model is a dict with one fixed shape: an optional front
end (conv *or* GRU, never both) followed by a dense stack. Each layer type is special-cased in about eight places
(`.qs.check`, `.qs.inputSize`, `.qs.features`, `.qs.predict`/`predictRaw`, `.qs.grad1`, `.qs.params`, L2,
`.qs.summary`). Adding batch norm or a skip connection means touching all of them. A GRU over conv features
(the follow-up after G5.3 failed) can't be expressed at all. Training assumes one model, one input and one
supervised loss, so a GAN or an autoencoder with a custom objective has no place to go.

So the first stage is a general layer core. Everything else builds on it.

## 2. Open questions for the owner

1. **M-Q1, what to grow first after the core.** This is the main choice. Recommended **(a)**: it is measurable
   on the GW problem straight away.
   - (a) *Better CNNs*: batch norm, residual blocks, dilated and strided convs, global pooling. These are what
     published GW detectors use. Test: does a deeper residual network beat gw_cnn_4 at SNR 8-10, where it finds
     only 38%?
   - (b) *Sequence models*: an LSTM, GRUs that return whole sequences (stackable after conv layers), then
     self-attention and a transformer encoder block.
   - (c) *Generative models*: autoencoders (a denoising one for GW strain), VAEs, then GANs (for example,
     generating realistic glitches to train against).
2. **M-Q2, saved models.** Recommended: the new core uses a new model format, and `.qs.load` converts old
   `.qsm` files (`gw_cnn_1`-`4`, `spirals_1`, the FX models) on load. The alternative is to keep both code paths.
3. **M-Q3, speed.** Training gw_cnn_4 already takes hours on this shared machine, and deeper networks cost more.
   Recommended: stay pure q (the point of the project), but measure where the time goes before the bigger models
   (stage M4), and accept 2-3× slower per parameter if necessary.

## 3. Stages

### M1 — General layer core (done 2026-11-03)

How it differs from the sketch below, and why:
- **A layer is a `(kind;spec)` pair, and a model's `params` is a list of arrays per layer, not a table or dictionaries.**
  q silently turns a list of dictionaries with the same keys into a table, and `enlist` of one dictionary into a one-row
  table, so a model that held dictionaries would change shape depending on its contents (two dense layers, say). Layers
  are `(kind;spec)` pairs, parameters and caches are lists, and each layer function still receives its parameters as a
  dictionary built from `.qs.layer.names[kind]`.
- **Dense and its activation are separate layers** (`dense`, `act`), as planned, and the dropout *training option* keeps
  working: it inserts dropout layers after the hidden dense activations for one gradient (`.qs.withDropout`), and the
  masks are drawn for the whole batch before it is split over threads, so seeded runs repeat whatever the thread count.
- **The registry is a set of dictionaries keyed by kind** (`.qs.layer.shape`, `.init`, `.forward`, `.backward`, `.check`,
  `.names`, `.l2`, `.inGrad`, `.example`), like `.qs.act` and `.qs.opt`. `.qs.layer.add[kind;parts]` registers a type.
- **`seq` is a layer** (flat rows to the channel-major layout), so no special case remains at the model's input.
- **Inference skips what only backpropagation needs**, and the first layer skips its input gradient. Both made things
  faster, not slower (M1.5).

- [x] **M1.1 Layer registry.** `.qs.layer.*`, `.qs.assemble` (shapes worked out layer by layer and recorded in each spec),
  `.qs.check`. `.qs.predict`, `.qs.predictRaw`, `.qs.grad`, `.qs.gradDrop`, L2 (`.qs.addL2`, from `.qs.layer.l2`),
  `.qs.params` and `.qs.summary` are generic loops over the layers.
- [x] **M1.2 Port the existing layers.** `dense`, `act`, `conv1d`, `maxpool1d`, `gru`, `dropout`, `flatten` and `seq`.
  `.qs.init`, `.qs.initConv` and `.qs.initGru` build the same networks with the same random draws in the same order.
  *Checked against the old library:* `tests/fixtures/old_format_cases` holds seven models (MLPs with relu, tanh, softmax
  and linear outputs; two conv models, one with pool 1; a GRU; one with extra keys) and the numbers the old library gave
  for them. `tests/test_core_compat.q` reproduces every prediction, logit and gradient to 1e-12, and whole training runs
  (adam, momentum, l2, dropout, conv, GRU) to 1e-9. The dashboard's browser check gives the same runs (best epoch 93 on
  moons, same decision boundary). The dashboard, `serve.q`, `train.q` and `plot.q` use new accessors (`.qs.inputSize`,
  `.qs.outputSize`, `.qs.sizes`, `.qs.acts`). The tests that read the old layout directly were rewritten for the new one.
- [x] **M1.3 Generic gradient check.** `.qs.gradCheck[kind;spec;inShape]`, run by `tests/test_layers.q` over every
  registered type with the examples each registers. Every type agrees with finite differences to about 1e-10 (the limit is
  1e-6), including the input gradient where a layer has one. A test shows the check catches a gradient that is 10% wrong,
  and another registers a new layer type from outside the library and trains through it.
- [x] **M1.4 Old models load.** `.qs.load` and every public function convert the old layout (`.qs.fromOld`, extra keys
  kept). *Checked:* `gw_cnn_4`, `gw_gru_1` and `spirals_1` give **identical** logits to the old library (difference exactly
  0) on 300 random inputs, with threads off and with `-s 4`. One time slide of the real G6 test noise (GW150914's held-out
  noise) scored by `gw_cnn_4` gives 1,790 trigger logits with a maximum difference of 0 between the two libraries, so
  the G6 thresholds, which are computed from those logits, reproduce. (The full 76-slide run wasn't repeated; it would
  have overwritten the saved thresholds.)
- [x] **M1.5 Speed parity.** The machine was saturated by other jobs (load average 17.6 on 16 cores), so wall-clock times
  swung by 3×; the comparison uses CPU time, old and new alternating in fresh processes.
  - `gw_cnn_4`'s architecture, 128-row gradient: 1.20 s old, 1.19 s new. 512-row prediction: 2.33 s old, **1.93 s new**
    (17% faster, from not working out which position won each pooling block when only predicting).
  - GRU gradient: equal. MNIST-shaped MLP (784-128-10) training epoch: 0.60 s old, 0.60 s new with threads off; with 4
    threads 0.59 to 0.70 s old, 0.40 to 0.46 s new.
  - A step's parts match too: gradient 7.0 ms, Adam update 1.4 ms, whole step 8.8 ms, both libraries.
  - The benchmark suite (`bench/benchmarks.q`) meets every target.

### M2 — New layers (order set by M-Q1; each needs a gradient check and a small learning test)

- [ ] **M2.1 Normalisation.** Batch norm (running statistics for inference) and layer norm.
- [ ] **M2.2 Residual blocks.** A `residual` layer wrapping a sub-list of layers, with a 1×1 projection when
  shapes differ. This is the one non-sequential structure needed for ResNets and transformers.
- [ ] **M2.3 Conv options.** Same padding, stride, dilation; average and global average pooling; Conv2D (for
  spectrograms and MNIST).
- [ ] **M2.4 Recurrent.** An LSTM; GRU and LSTM with a `sequence` option returning every step, so they stack
  and can follow conv layers.
- [ ] **M2.5 Attention.** Multi-head self-attention with positional encoding, then a transformer encoder block
  (attention + layer norm + feed-forward, residual).
- [ ] **M2.6 More activations.** GELU, leaky ReLU, ELU.

### M3 — Training system

- [ ] **M3.1 Custom objectives.** `.qs.train` accepts a loss function returning the value and its gradient at the
  output, so new objectives need no core edits (a weighted bce, a reconstruction loss plus a KL term, focal loss).
- [ ] **M3.2 Optimiser extras.** AdamW (decoupled weight decay), gradient clipping by norm, learning-rate warmup,
  and per-layer frozen parameters (for fine-tuning a trained network).
- [ ] **M3.3 Checkpoints.** Save the model and optimiser state every n epochs, and resume a run from them (long GW
  runs have been lost to shared-machine trouble).
- [ ] **M3.4 Several networks in one loop.** A training step over a set of networks with a caller-supplied
  step function: the basis for GANs (alternating generator and discriminator updates) and for
  encoder/decoder pairs.

### M4 — Speed

- [ ] **M4.1 Profile.** Measure time per layer type in forward and backward for gw_cnn_4 and a deeper candidate,
  and record it in the log.
- [ ] **M4.2 Fix the top costs found.** Likely candidates: the gathers that build conv patch matrices, the max
  pool's scatter in backward, and threading inside a layer rather than only across batch shards. *Accept:*
  measured speed-up on the M4.1 cases, with outputs unchanged.

### M5 — Generative models

- [ ] **M5.1 Autoencoder.** Dense and conv encoder/decoder (needs transposed conv or upsampling layers).
  *Accept:* reconstructs MNIST; a denoising version trained on GW injections recovers the clean whitened chirp
  better than the noisy input does (overlap measured).
- [ ] **M5.2 VAE.** Reparameterised sampling and the KL term (on M3.1). *Accept:* samples of MNIST digits that
  look like digits; the latent space interpolates.
- [ ] **M5.3 GAN.** On M3.4. *Accept:* on a 2-D toy distribution the generator matches it (measured, not by
  eye); then a 1-D GAN for simulated glitches (blips, sine-Gaussians). Whether GAN glitches help detector
  training is the real test. GANs are notoriously unstable, so this stage may take more than one attempt.

### M6 — Proof on real problems

- [ ] **M6.1 A better GW detector.** A residual CNN with batch norm, trained as gw_cnn_4 was, then evaluated
  with `projects/gw/experiments/gw_eval.q` on the same test noise. *Accept:* a reported efficiency curve against gw_cnn_4's,
  whichever way it comes out; if better, rerun the O4a day and GW190707.
- [ ] **M6.2 GRU or attention over conv features** on the GW task (the follow-up after G5.3 failed).
- [ ] **M6.3 The benchmarks:** MNIST with Conv2D in `bench/benchmarks.q`; FX with an LSTM against the current
  results.

## Progress log

- 2026-11-02: Plan written after reviewing `src/defineEquation.q`. Not started.
- 2026-11-03: Started on the owner's go-ahead; moved to `inProgress/`, with the recommended answers to M-Q1 to M-Q3 assumed.
  **M1 done**: layer registry and the eight ported layer types, old models converted on load, generic gradient check, speed
  at parity or better. Before changing anything, fixtures and timings were recorded from the old library (see M1.2 and M1.5).
  q gotchas met: a list of dictionaries with the same keys becomes a table, and `enlist` of a dictionary is a table (so
  layers are pairs and parameters lists); a dictionary whose values are all symbols can't take a vector by amending (join a
  new dictionary instead); `desc` and `ij` are keywords (`'assign`, `'ij`); `.[x;i;f;v]` with a mixed nested list gave
  `'nyi` (L2 is now a multiplication by 1 or 0 per parameter); `read0` of a `/proc` file is empty (use `cat`); and wall-clock
  timings on this shared machine are unusable, so compare CPU time (`/proc/<pid>/stat`).
