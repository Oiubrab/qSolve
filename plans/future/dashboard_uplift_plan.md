# Dashboard uplift plan: easy on top, everything underneath

The owner, 2026-11-03: the dashboard still feels clunky. Someone should be able to come in and run whatever they
want. Review the setup, and find what to bring forward and what to hide, so it's easier to use.

**The owner's philosophy.** There are two classic ways to build a tool:
- **Inch wide, mile deep:** thousands of focused, advanced options.
- **Mile wide, inch deep:** lots of basic options and obvious visuals.

The owner wants both, layered:
- **On top:** lots of good visuals of the data and the model, and very easy data upload, training and
  visualisation.
- **Underneath:** one click opens the advanced options. Every one of them is reachable there, and none of them is
  in the way until then.

Read the "q gotchas" in [../completed/improvemet_plan.md](../completed/improvemet_plan.md) first.

**How to use this file:** the same as the other plans.
- Tick an item only when its check passes and `q tests/run.q` passes.
- Add a dated line to the Progress log.
- Commit each finished stage to main.
- Don't push without asking.

The browser check `node tests/ui/check.js` must pass after every stage that changes the page, and it grows with
the stages.

**Current status (2026-11-03):** design only, not started. The owner answered D-Q1 to D-Q5 (section 4):
- jobs run in separate q worker processes;
- the page may run the known download scripts;
- a bold redesign;
- advanced options go fully deep, but with no q console;
- runs are kept on disk.

Start with D0, the design, which needs the owner to pick a direction.

---

## 1. Review: the dashboard today

This review is based on reading `app/dashboard.q` (420 lines), `app/ui/app.js` (1,305 lines), `index.html` and
`style.css`. It also uses the screenshots of every view from the browser check, taken 2026-11-03.

**What works and should stay:**
- It's one q process with no build step and no dependencies, and it works offline. It's served by q and talks over
  one WebSocket.
- Live loss and accuracy charts, updated every epoch.
- The decision boundary for 2-D data, the confusion matrix, and the forecast-against-actual chart.
- The Predict view speaks the data's own terms: a form with training ranges, a clickable map, a drawing pad for
  digits, recent values for a series, and "try a validation example".
- Presets per dataset, light and dark themes, and a layout that works on a phone.
- The KX Dashboards connection (`app/kx/qsolve.json`; `.dash.runs`, `.dash.history[]`).
- The Waves view, the best-looking page in the app.

**What's clunky,** most important first:

1. **Everything is shown at once, at the same weight.** The Train form opens with about 20 controls: dataset,
   rows, noise, upload, hidden layers, activation, output, epochs, batch, learning rate, optimiser, schedule,
   dropout, L2, seed, patience, cross-validation and keep-best. A newcomer sees seed and L2 before seeing the data.
2. **Training locks the whole dashboard.** `.dash.api.train` runs `.qs.train` inside the WebSocket handler. Until
   a run ends, nothing else answers, and there's no Stop. A 5-epoch MNIST run takes about 25 s. "Run whatever they
   want" isn't possible while one thing blocks everything.
3. **You can't see the data before training.** The first screen is an empty card that says "Pick a dataset and
   press Train". Moons, an uploaded table, a series and MNIST could all be drawn the moment they're picked.
4. **Uploading is an afterthought.**
   - It's a small "Upload CSV…" button under the dataset list, with no drag-and-drop.
   - The server already returns a preview and column kinds, but the page shows only "1500 rows, 2 columns".
   - There's no column profile: type, gaps, range, distribution.
   - There's no suggested target, and no hint whether a column is numeric or a category, beyond its name.
5. **Search is a separate tab that depends on another tab.** It "uses the Train tab's data and settings". Its
   inputs are syntax, for example `8; 16; 16, 16`, and its results are only a table.
6. **Runs and Models split one idea in two.** Runs are this session's models, and Models are saved files. That
   difference is an implementation detail. Also:
   - Saved models can't be deleted, inspected or downloaded from the page.
   - Runs are lost when the dashboard restarts.
   - Compare puts different datasets' losses on one axis: in the screenshot, moons, houses and an uploaded series
     share a chart.
7. **The hidden layers field is free text** (`32, 16`), and the network itself is never drawn.
8. **Part of the library is out of reach:**
   - from the UI entirely: the loss; momentum and the Adam betas; the step-decay settings and the cosine floor;
     shuffling; the validation share (the server reads `frac` but the page never sends it); the Conv1D and GRU
     front ends;
   - from the dashboard: anything from [model_expansion_plan.md](model_expansion_plan.md) as it arrives.
9. **Nothing explains itself, and nothing says what went wrong.**
   - There's no help on terms like L2, dropout or patience.
   - There's no hint after a bad run, for example "the loss diverged: try a smaller learning rate", or "training
     is much better than validation: it's overfitting".
10. **Datasets that need downloading are dead ends.** They show "(download needed)" and point to a shell script.
11. **Project views are hard-coded into the core page.** The Waves tab lives in `index.html` and `dashboard.q`.
    Markets and geopolitics would each have to edit core files to add a view.
12. **Smaller things:**
    - Toasts sit over the charts, for example on the Waves spectrogram.
    - Predicted numbers show raw precision, like `441.107` and `164.7183`.
    - The address bar doesn't record the view, so the Back button and reloading lose your place.
    - `app.js` is one 1,300-line file.

## 2. The design: three layers

| Layer | Who it's for | What it shows |
|---|---|---|
| **1. Just go** | anyone, first visit | A dataset gallery with pictures, or drop a file anywhere. The data is drawn straight away. One **Train** button, sensible defaults from presets, and live visuals. Then **Use it**: predict, save, compare. |
| **2. Simple knobs** | someone who wants to play | About four controls in plain words, each mapped onto real settings: **model size** (small, medium, large, custom), **training length**, **learning speed**, **guard against overfitting** (off, some, strong). Plus **Find good settings for me**, which is the search with a time budget. |
| **3. Advanced** | someone who knows what an optimiser is | A drawer opened by one button, holding every library option under its library name, with the range, default and a one-line explanation. Plus the raw config as JSON, "copy as q code" (the exact `.qs.init` and `.qs.train` calls that reproduce the run), architectures (dense, Conv1D or GRU front ends, and new layer types as the model expansion plan adds them), and the search grid. |

**Rules that make the layering honest:**
- **Hidden isn't lost.** Closing the drawer keeps every value. A badge says "3 advanced settings changed", with a
  reset.
- **The layers agree.** The simple knobs write real settings, and the drawer shows exactly what they wrote. Changing
  an advanced value that a knob controls turns that knob to "custom".
- **Everything comes from one schema.** One server-side table describes every option: its name, level, label, help,
  type, range, default and when it applies. The form, the drawer, the search grid and "copy as q" are all built
  from it. A test checks that every key of `.qs.defaults` is in the schema, or is listed as deliberately not
  shown (`verbose`, `onEpoch`).

**New navigation:** **Data → Train → Models**, plus **Projects**.
- **Data:** the gallery and uploads.
- **Train:** training, with Tune folded in as a mode.
- **Models:** everything trained, this session or saved, each with its report, predictions and comparison.
- **Projects:** Waves now, and each project's own views later.

Predict becomes the **Use it** panel of a model. It keeps everything it does now, opened from Train's results or
from Models.

## 3. Decisions

| # | Decision | Why |
|---|---|---|
| D-D1 | **Stay dependency-free:** vanilla JS with SVG and canvas, served by q, no build step, works offline. Split `app.js` into ES modules served from `app/ui/` and its subfolders. Fonts are bundled in `app/ui/`, never fetched from a CDN. | That's the current strength. Module scripts only need the JavaScript MIME type q already sends, and the static-file rule widens to subfolders of `ui/` only. |
| D-D2 | **Long work becomes jobs, each in its own q worker process** (D-Q1): training, search, cross-validation, permutation importance and downloads. The dashboard process stays free for the page and quick commands like predictions and the Waves views. Each job has an id, progress, a result, and **Stop**. The header shows running jobs. | Point 2 of the review is the root cause of "clunky". Workers also let a search run its trials in parallel. |
| D-D7 | **Bold redesign** (D-Q3), still with light and dark themes, still at phone width, and still following D-D5. The owner picks the direction from mockups (D0.1) before any building. | A bold look is a matter of taste, so it's shown, not described. |
| D-D8 | **Runs live on disk** (D-Q5) in `runs/<id>/`, git-ignored: the config, history, report, model, and the uploaded data it trained on. Workers write there, so the disk is also how results come back. The dashboard reloads it on start. | One mechanism for both persistence and returning results from workers. |
| D-D3 | **The options schema lives on the server** as `.dash.options`, a q table, sent in `hello`. | One source of truth. q decides what's valid, and the page only draws it. |
| D-D4 | **The KX Dashboards interface stays compatible:** `.dash.runs` and `.dash.history[]` keep their columns. New columns are only added. | Others may have connections built on it. |
| D-D5 | **Visuals follow the dataviz rules used so far:** one axis per chart, fixed series colours per run, a hover tooltip on every chart, legends for two or more series, and each colour scheme checked in light and dark. | The current charts already mostly do this. New ones must too. |
| D-D6 | **Every stage keeps the current features working.** The browser check's existing scenarios must pass throughout, rewritten only where the layout moves things. | This is an uplift, not a rewrite. |

## 4. The owner's answers (2026-11-03)

1. **D-Q1, how jobs run: separate q worker processes.**
   - The dashboard starts one q process per job. The worker connects back over IPC, trains, streams epochs as
     async messages, and writes its result to `runs/<id>/` (D-D8).
   - The single-process timer design was the alternative, and isn't used.
   - **To check at the start of D2:** how many q processes, and how many threads each, the licence allows
     alongside the dashboard. We run at most `-s 4` now. That sets the default number of workers running at once.
     Further jobs queue.
2. **D-Q2, downloads from the page: yes.**
   - Only the known scripts run: `scripts/fetch_data.sh`, and project fetch scripts registered by name. A request
     never supplies a command or path.
   - Each download runs as a job, with its output streamed to the page.
   - The dashboard still listens on localhost only by default.
3. **D-Q3, look: a bold redesign.** Stage D0 shows directions as mockups for the owner to choose from.
4. **D-Q4, depth: fully advanced, without a q console.** That means:
   - the full options drawer;
   - the architecture editor;
   - the raw JSON config;
   - "copy as q code";
   - the full search grid.

   There's no box in the page for running q. The port's existing q interface for KX Dashboards is unchanged.
5. **D-Q5, runs on disk: yes** (D-D8). A "clear session" button, and deleting a run, remove its folder.

## 5. Stages

### D0 — The bold redesign: pick a direction, then build the look

- [ ] **D0.1 Three directions as mockups.** Static HTML pages showing the same two screens in each direction:
  - the Train screen with data drawn and a run in progress;
  - a finished run's results.

  Each is shown in light and dark, and at phone width, using real data from the browser check's runs. Starting
  ideas for the three:
  - a **lab instrument**: dark-first, dense, glowing traces, like the Waves page pushed further;
  - an **editorial** style: big type, generous space, charts as the heroes;
  - a **playful canvas**: the data and the network fill the screen, with controls floating over them.

  *Accept:* the owner picks one, or a mix. The choice and the reasons go in the Progress log.
- [ ] **D0.2 Design tokens and components.** CSS variables for colour, type, spacing, radius and motion. The
  chart palettes pass the dataviz validator in both themes. Build one set of components: card, button, field,
  drawer, tabs, toast, job pill, tooltip and empty state. Toasts never cover a chart, and they stack and fade.
  Numbers are shown to sensible significant figures (`441.1`, not `441.107`).
- [ ] **D0.3 The current views in the new look,** with behaviour unchanged.
  *Accept:* the browser check passes, and its screenshots are shown to the owner.

### D1 — Foundations

- [ ] **D1.1 Split `app.js` into modules:** connection, state, charts, data, train, models, predict and projects.
  No behaviour changes. *Accept:* the browser check passes unchanged.
- [ ] **D1.2 The options schema.** `.dash.options` has one row per option: name, level (`basic`, `simple` or
  `advanced`), group, label, help, type, range, default and "applies when". Options include data options like
  rows, noise, limit and validation share.
  *Accept:* a q test checks it covers every `.qs.defaults` key except `verbose` and `onEpoch`. A browser test
  checks every row is drawn somewhere.
- [ ] **D1.3 A form built from the schema.** The Train settings and the search grid are both drawn from
  `.dash.options`, replacing the hand-written HTML.
  *Accept:* the same requests as before, compared by recording the payloads in the browser check.
- [ ] **D1.4 The address holds the view and the run** (`#train`, `#models/3`), so Back and reload work.

### D2 — Jobs in worker processes

- [ ] **D2.0 Licence check.** Start the dashboard plus 1, 2 and 4 workers, each with `-s 1` and `-s 2`, and record
  what the licence allows.
  *Accept:* a default number of workers and threads per worker, written into the README.
- [ ] **D2.1 Stopping from outside.** A worker is busy inside `.qs.train`, so it can't read messages until it
  finishes. To fix that, the library lets `onEpoch` return `` `stop `` to end training after that epoch. The
  result is the model so far, or the best one with `keepBest`.
  - The worker's `onEpoch` checks for a `runs/<id>/stop` file, which the dashboard writes when Stop is pressed.
  - If a worker hasn't stopped within 10 s of its epoch time, Kill ends the process by its PID.

  *Accept:* library tests for early stopping. `.qs.train` without it gives identical results to now.
- [ ] **D2.2 The worker.** `app/worker.q` loads the library and the same data and config code as the dashboard,
  which moves into shared files under `app/`.
  - It is started with the job id and the dashboard's port.
  - It connects back, receives the job: the request, plus the table for an upload.
  - It runs it, sends `start`, `epoch` and `cv` messages as async IPC, writes `runs/<id>/`, sends `done`, and
    exits.
  - A worker that dies is marked failed, with the tail of its log.
- [ ] **D2.3 Dashboard jobs.** `.dash.jobs` is a table of id, kind, state, progress, PID and started time. It
  starts workers up to the D2.0 limit and queues the rest. It forwards worker messages to the browser, and loads
  results from `runs/<id>/` when they finish.
  - Train, cross-validation and search become jobs.
  - A search spreads its trials over the free workers.

  *Accept:* while MNIST trains:
  - the page switches views, predicts with an older model and shows a GW event;
  - a second training job runs alongside, or queues if the limit is 1;
  - Stop ends a job within one epoch and keeps the model so far;
  - killing a worker by hand marks its job failed without disturbing the dashboard.
- [ ] **D2.4 Downloads as jobs** (D-Q2). A registry of named fetch scripts, `scripts/fetch_data.sh` and each
  project's. The page asks for one by name, and the dashboard runs it in a worker and streams its output.
  *Accept:* an unknown name is refused. A dry run streams in the browser check.
- [ ] **D2.5 The jobs indicator** in the header: what's running and queued, progress, Stop and Kill. It works from
  any view.

### D3 — Layer 1: data and "just go"

- [ ] **D3.1 Dataset gallery.** Cards with a small picture drawn from a sample of the data: dots for 2-D, a
  sparkline for series, four digits for MNIST, a mini table for tables. Each card has a one-line purpose. A card
  that needs downloading has a Download button that starts the D2.4 job, and the card comes alive when it
  finishes.
- [ ] **D3.2 Drop a file anywhere.** CSV via drag-and-drop or the file picker.
  - Show a column profile: kind, gaps, range or levels, and a small histogram per column.
  - Suggest a target: the last column, or the only categorical one.
  - Suggest a task: classes, a number, or a forecast if there's a time-like column.
  - Show a plain sentence: "Predict *price* from *area, rooms, age, city*: 1,600 rows for training, 400 to check
    it".
- [ ] **D3.3 See the data first.** As soon as data is picked, the results area draws it:
  - a scatter for 2-D data;
  - the series;
  - an image grid;
  - a table preview with the column profile.

  The Train button sits under the picture.
- [ ] **D3.4 The first-visit path.** Open the dashboard, pick a card, press Train, see live results.
  *Accept:* the browser check does it in 2 clicks from load. At most 6 controls are visible on the first screen,
  against about 20 now.

### D4 — Layer 2: simple knobs and "find good settings"

- [ ] **D4.1 Four knobs:** model size, training length, learning speed and overfitting guard. Each maps onto
  schema options per data kind and size. For example, "medium" might be one hidden layer of 32 for a table and 128
  for images. The mapping lives in the schema, not in page code.
  *Accept:* a q test of the mapping for each data kind.
- [ ] **D4.2 Find good settings for me.** A search over a small space chosen by data kind, with a time budget (1,
  5 or 15 minutes) instead of a grid. It shows results as a chart (validation loss against each setting) as well
  as a table. "Use the best" fills in the knobs.
- [ ] **D4.3 After-run coach.** Simple checks on the history produce one-line diagnoses, each with a one-click fix:
  - the loss went up or to NaN: lower the learning rate;
  - training is far better than validation: raise the overfitting guard;
  - both are still falling at the end: train longer;
  - no change for many epochs: the learning rate is too low, or the model too small.

  *Accept:* each diagnosis triggered by a deliberately bad config in the browser check.

### D5 — Visuals

All charts follow D-D5. Each is checked in light and dark, and at phone width.

- [ ] **D5.1 The network, drawn.** Layers as columns of nodes; for large layers, a bar with the count. Weights are
  shown as lines whose thickness and colour give size and sign. It updates during training, and works for
  Conv1D and GRU front ends.
- [ ] **D5.2 Decision boundary over time.** For 2-D data, the boundary redrawn every few epochs during training,
  with a scrubber to replay it afterwards.
- [ ] **D5.3 Inside the model:** a histogram of weights and gradients per layer by epoch, to show learning or
  stalling.
- [ ] **D5.4 Mistakes.**
  - For images: a gallery of misclassified validation examples, with what the model said.
  - For tables: the rows with the largest errors.
  - For regression: residuals against the predicted value.
- [ ] **D5.5 What matters.** For table models, permutation importance per input column, on validation data, run as
  a job.
- [ ] **D5.6 Compare, grouped.** Runs are grouped by dataset, so different datasets never share an axis, and an
  accuracy or R² chart sits beside the loss.
- [ ] **D5.7 Search, drawn.** Parallel coordinates, or one small chart per setting, showing which settings gave
  low validation loss.

### D6 — Layer 3: advanced

- [ ] **D6.1 The drawer.** One "Advanced" button opens every schema option, grouped (data, architecture,
  optimiser, schedule, regularisation, stopping, reproducibility), with help on each, the changed badge and reset.
- [ ] **D6.2 Architecture editor.** Add and remove layers and pick each one's type, size and activation. It
  includes Conv1D and GRU front ends when the data's shape allows, and layer types from the model expansion plan as
  they land, through the schema. The D5.1 drawing doubles as its preview.
- [ ] **D6.3 Config as text.** Show, edit, import and export the full run config as JSON.
  *Accept:* exporting a run's config and importing it reproduces the run exactly.
- [ ] **D6.4 Copy as q code.** The exact q that reproduces the run outside the dashboard: data load, `.qs.init`
  and `.qs.train` with every option.
  *Accept:* the browser check copies the snippet for a run, runs it in q, and gets the same validation loss.
- [ ] **D6.5 Full search grid.** The current Search tab's power, built from the schema, with every option
  searchable, as an advanced mode of D4.2.

### D7 — Models, saving, export

- [ ] **D7.1 One Models view.** Session runs and saved files appear together, each marked with its status. Each
  row opens the full report and Use it. You can delete, rename and compare runs.
- [ ] **D7.2 Runs across restarts** (D-D8).
  - On start, the dashboard reloads every `runs/<id>/`, including the uploads they trained on, so examples and
    predictions still work.
  - "Clear session" removes the folders after a confirmation built into the page.
  - Runs from before a format change still load, or are listed as old with the reason.

  *Accept:* the browser check trains, restarts the dashboard, and finds the run with its charts and predictions.
- [ ] **D7.3 Downloads:** the model file (`.qsm`), the report, prediction results as CSV, and any chart as PNG or
  SVG. Uploading a `.qsm` adds it to Models.

### D8 — Projects as plugins

- [ ] **D8.1 A project panel interface.** A project can add a view by providing `projects/<name>/ui/<name>.js` and
  dashboard commands in `projects/<name>/src/<name>_dash.q`. Both are found at start, the way `tests/run.q` finds
  project code.
- [ ] **D8.2 Waves moves out.** The Waves view and the `gw*` commands move into `projects/gw/` through that
  interface. *Accept:* the Waves part of the browser check passes unchanged.
- [ ] **D8.3 Markets gets a small panel.** EUR/USD and GBP/USD with the FX study's forecasts against the actual
  rates. This shows a second project can be added without touching core files. Geopolitics adds its own later,
  through P7 of its plan.

## Progress log

- 2026-11-03: Plan written after a review of the dashboard: code, and screenshots of every view from the browser
  check. Not started.
- 2026-11-03: The owner answered D-Q1 to D-Q5: worker processes, downloads allowed, a bold redesign, fully
  advanced with no q console, runs on disk. The decisions and stages were updated to match. The old D0 quick wins
  are folded into D0.2, D1.4 and D5.6, and the upload preview into D3.2. Not started.
