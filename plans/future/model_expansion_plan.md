# Model expansion plan: from a fixed CNN/MLP to a general network toolkit

The owner, 2026-11-02: the network behind the GW work is a small CNN with about 35k parameters (three Conv1D layers
and two dense ones). The training system needs to grow, so qSolve can build modern architectures (deeper and residual
CNNs, sequence models, attention), and generative ones (autoencoders, GANs), still in pure q.

Read the "q gotchas" in [../completed/improvemet_plan.md](../completed/improvemet_plan.md) first.

**How to use this file:** as the other plans. Tick an item only when its check passes and `q tests/run.q`
passes; add a dated line to the Progress log; commit each finished stage to main (the owner's rule).

**Current status (2026-11-02):** design only; open questions M-Q1 to M-Q3 below need the owner.

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

### M1 — General layer core

- [ ] **M1.1 Layer registry.** A model is `` `layers`meta ``. Layers is a list of dicts
  (`` `type`spec`params ``). Each type registers `init`, `forward[spec;params;A;train]` → output and cache,
  `backward[spec;params;cache;dOut]` → parameter gradients and input gradient, `shape[spec;inShape]`,
  and which of its parameters take L2. `.qs.predict`, `.qs.grad`, `.qs.step`, L2, `.qs.params` and
  `.qs.summary` become generic loops over the list.
- [ ] **M1.2 Port the existing layers** as separate types: `dense`, `act`, `conv1d`, `maxpool1d`, `gru`,
  `dropout` (now usable anywhere), `flatten`. *Accept:* all current tests pass unchanged through the public
  functions; the existing training scripts and dashboard work unchanged.
- [ ] **M1.3 Generic gradient check.** `.qs.gradCheck[layer;inShape]` compares backward with finite differences
  for parameters and inputs. It runs over every registered type in the tests, so each new layer gets checked for
  free.
- [ ] **M1.4 Old models load.** `.qs.load` converts the old format (M-Q2). *Accept:* the converted gw_cnn_4 gives
  the same logits as before (within 1e-9) on 1,000 windows, and its G6 thresholds reproduce.
- [ ] **M1.5 Speed parity.** *Accept:* training and prediction for gw_cnn_4 and the MNIST benchmark are no more
  than 10% slower than before.

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
