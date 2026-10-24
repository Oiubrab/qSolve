// qSolve dashboard client: talks to dashboard.q over a WebSocket and draws everything with plain DOM, SVG and canvas
"use strict";

// ---------- small helpers ----------

const $ = (s) => document.querySelector(s);

// element builder: h("td", {class: "num"}, "1.5"); strings become text nodes, never HTML
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "style") e.style.cssText = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}

const NS = "http://www.w3.org/2000/svg";
function s(tag, attrs, ...kids) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v !== null && v !== undefined) e.setAttribute(k, v);
  for (const k of kids.flat()) if (k !== null && k !== undefined) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const series = (i) => css(`--s${(i % 8) + 1}`);

const isNum = (x) => typeof x === "number" && isFinite(x);
function fmt(x, digits = 3) {
  if (!isNum(x)) return "–";
  const a = Math.abs(x);
  if (a !== 0 && (a < 0.001 || a >= 1e6)) return x.toExponential(2);
  if (a >= 1000) return x.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return x.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}
const pct = (x) => (isNum(x) ? +(100 * x).toFixed(1) + "%" : "–");
function secs(x) {
  if (!isNum(x)) return "–";
  if (x < 1) return Math.round(x * 1000) + " ms";
  if (x < 60) return x.toFixed(1) + " s";
  return Math.floor(x / 60) + " min " + Math.round(x % 60) + " s";
}

function toast(msg, kind) {
  const t = h("div", { class: "toast" + (kind ? " " + kind : "") }, msg);
  $("#toasts").append(t);
  setTimeout(() => t.remove(), kind === "error" ? 8000 : 3500);
}

// ---------- connection ----------

const conn = { ws: null, next: 1, pending: new Map(), busy: 0, retry: 500, queue: [] };

function setStatus(state, text) {
  $("#conn-dot").className = "dot " + state;
  $("#conn-text").textContent = text;
}

function connect() {
  setStatus("", "connecting");
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/");
  conn.ws = ws;
  ws.onopen = () => {
    conn.retry = 500;
    setStatus("on", "connected");
    for (const m of conn.queue.splice(0)) ws.send(m);
    hello();
  };
  ws.onclose = () => {
    setStatus("off", "disconnected, retrying");
    for (const [, p] of conn.pending) p.reject(new Error("connection lost"));
    conn.pending.clear();
    conn.busy = 0;
    updateBusy();
    setTimeout(connect, conn.retry);
    conn.retry = Math.min(conn.retry * 2, 8000);
  };
  ws.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    const p = conn.pending.get(m.id);
    if (!p) { if (m.type === "error") toast(m.message, "error"); return; }
    if (m.type === "error") { conn.pending.delete(m.id); p.reject(new Error(m.message)); return; }
    if (p.final.includes(m.type)) { conn.pending.delete(m.id); p.resolve(m); }
    else if (p.onMsg) p.onMsg(m);
  };
}

// send a command; resolves with the final reply, onMsg gets streamed messages before it
function call(cmd, payload = {}, onMsg, long = false) {
  const id = conn.next++;
  const final = { hello: ["hello"], upload: ["uploaded"], train: ["done"], search: ["searched"], predict: ["predicted"],
    save: ["saved"], load: ["loaded"], report: ["done"], forget: ["runs"], runs: ["runs"], datasets: ["datasets"] }[cmd];
  const msg = JSON.stringify({ ...payload, cmd, id });
  return new Promise((resolve, reject) => {
    if (long) { conn.busy++; updateBusy(); }
    const done = (f) => (x) => { if (long) { conn.busy--; updateBusy(); } f(x); };
    conn.pending.set(id, { final, onMsg, resolve: done(resolve), reject: done(reject) });
    if (conn.ws && conn.ws.readyState === 1) conn.ws.send(msg); else conn.queue.push(msg);
  });
}

function updateBusy() {
  const b = conn.busy > 0;
  for (const id of ["#train-btn", "#search-btn", "#predict-btn", "#predict-csv"]) $(id).disabled = b;
  if (conn.ws && conn.ws.readyState === 1) setStatus(b ? "busy" : "on", b ? "working" : "connected");
}

// ---------- state ----------

const state = {
  datasets: [], runs: [], models: [], uploads: {},
  reports: {},          // run id -> full report
  current: null,        // run being shown in Train
  live: null,           // streaming history of the run in progress
  compare: [],          // run ids ticked in Runs, in tick order
  colours: {},          // run id -> colour slot, fixed once given
  searchRows: [], searchBest: null,
};

// suggested settings per dataset (the benchmark configurations)
const PRESETS = {
  and: { hidden: "", epochs: 300, batch: 4, lr: 0.1, out: "sigmoid" },
  or: { hidden: "", epochs: 300, batch: 4, lr: 0.1, out: "sigmoid" },
  nand: { hidden: "", epochs: 300, batch: 4, lr: 0.1, out: "sigmoid" },
  xor: { hidden: "4", epochs: 300, batch: 4, lr: 0.05, act: "tanh" },
  fourBit: { hidden: "8", act: "relu", epochs: 30, rows: 2000, lr: 0.01 },
  sine: { hidden: "16", epochs: 300, rows: 1000, lr: 0.01 },
  circles: { hidden: "8", epochs: 100, rows: 1000, noise: 0.1, lr: 0.01 },
  moons: { hidden: "16", epochs: 100, rows: 1000, noise: 0.1, lr: 0.01 },
  spirals: { hidden: "32, 32", epochs: 300, rows: 2000, noise: 0.02, lr: 0.005 },
  houses: { hidden: "32", epochs: 100, rows: 2000, lr: 0.01 },
  seasonal: { hidden: "16", epochs: 100, rows: 3000, noise: 0.1, lr: 0.001 },
  iris: { hidden: "8", epochs: 100, batch: 16, lr: 0.01 },
  mnist: { hidden: "128", act: "relu", epochs: 5, batch: 64, lr: 0.001, limit: "" },
  fashion: { hidden: "128", act: "relu", epochs: 5, batch: 64, lr: 0.001, limit: "" },
};

const GROUPS = [
  ["logic", "Logic gates"], ["2d", "2-D shapes"], ["classes", "Classes"], ["regression", "Regression"],
  ["table", "Tables"], ["series", "Time series"], ["images", "Images"], ["upload", "Your uploads"],
];

// ---------- hello / datasets ----------

async function hello() {
  try {
    const m = await call("hello");
    state.datasets = m.datasets || [];
    state.models = m.models || [];
    state.runs = m.runs || [];
    renderDatasets();
    renderRuns();
    renderModels();
    renderPredictRuns();
  } catch (e) { toast(e.message, "error"); }
}

function renderDatasets(select) {
  const sel = $("#f-data");
  const cur = select || sel.value || "moons";
  sel.replaceChildren();
  for (const [kind, label] of GROUPS) {
    const ds = state.datasets.filter((d) => d.kind === kind);
    if (!ds.length) continue;
    const g = h("optgroup", { label });
    for (const d of ds) g.append(h("option", { value: d.name, disabled: !d.available }, d.name + (d.available ? "" : " (download needed)")));
    sel.append(g);
  }
  sel.value = state.datasets.some((d) => d.name === cur && d.available) ? cur : "moons";
  onDataChange(false);
}

function dataset() { return state.datasets.find((d) => d.name === $("#f-data").value) || {}; }

function onDataChange(applyPreset = true) {
  const d = dataset();
  const about = $("#data-about");
  about.textContent = d.about || "";
  about.className = "hint";
  if (d.kind === "images") about.textContent += ". All 60,000 rows take about 5 s per epoch; set a limit to go faster.";
  if (!d.available && d.name) { about.textContent = "Run scripts/fetch_data.sh to download this dataset."; about.className = "hint warn"; }
  $("#gen-opts").classList.toggle("hidden", !d.generated);
  $("#f-noise").closest(".field").classList.toggle("hidden", !(d.kind === "2d" || d.name === "seasonal"));
  $("#limit-opt").classList.toggle("hidden", d.kind !== "images");
  const up = d.kind === "upload";
  $("#upload-opts").classList.toggle("hidden", !up);
  if (up) renderUploadOpts(d.name);
  if (applyPreset && PRESETS[d.name]) applySettings(PRESETS[d.name]);
  $("#search-data").textContent = "Data: " + (d.name || "–") + ". Epochs, optimiser and the rest come from Train.";
}

function applySettings(p) {
  const map = { hidden: "#f-hidden", act: "#f-act", out: "#f-out", epochs: "#f-epochs", batch: "#f-batch", lr: "#f-lr",
    rows: "#f-rows", noise: "#f-noise", limit: "#f-limit", optimiser: "#f-optimiser", dropout: "#f-dropout" };
  for (const [k, sel] of Object.entries(map)) if (k in p) $(sel).value = p[k];
  if (!("out" in p)) $("#f-out").value = "auto";
}

// ---------- uploads ----------

$("#upload-btn").addEventListener("click", () => $("#upload-file").click());
$("#upload-file").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  ev.target.value = "";
  if (!f) return;
  if (f.size > 50e6) return toast("That file is over 50 MB: train it with train.q instead.", "error");
  try {
    const text = await f.text();
    const m = await call("upload", { name: f.name.replace(/\.[^.]*$/, ""), csv: text });
    state.uploads[m.name] = m;
    state.datasets = m.datasets;
    renderDatasets(m.name);
    toast(`Uploaded ${m.name}: ${m.rows} rows, ${m.columns.length} columns`);
  } catch (e) { toast("Upload failed: " + e.message, "error"); }
});

function renderUploadOpts(name) {
  const u = state.uploads[name];
  if (!u) return;
  const cols = u.columns;
  const tgt = $("#f-target");
  const prev = tgt.value;
  tgt.replaceChildren(...cols.map((c) => h("option", { value: c.name }, `${c.name} (${c.kind === "num" ? "number" : c.levels + " categories"})`)));
  tgt.value = cols.some((c) => c.name === prev) ? prev : cols[cols.length - 1].name;
  const drawFeatures = () => {
    const box = $("#f-features");
    box.replaceChildren(...cols.filter((c) => c.name !== tgt.value).map((c) =>
      h("label", { class: "check" }, h("input", { type: "checkbox", value: c.name, checked: true }), c.name)));
  };
  tgt.onchange = drawFeatures;
  drawFeatures();
  const mode = $("#f-mode");
  const sync = () => {
    const series = mode.value === "series";
    $("#features-field").classList.toggle("hidden", series);
    $("#series-fields").classList.toggle("hidden", !series);
    $("#target-field label").textContent = series ? "Column to forecast" : "Column to predict";
  };
  mode.onchange = sync;
  sync();
}

// ---------- train ----------

function parseLayers(t) {
  return t.split(/[,\sx]+/).filter(Boolean).map((x) => parseInt(x, 10)).filter((x) => x > 0);
}

function trainPayload() {
  const v = (id) => $(id).value;
  const num = (id) => (v(id) === "" ? null : Number(v(id)));
  const d = dataset();
  const p = {
    data: d.name, hidden: parseLayers(v("#f-hidden")), act: v("#f-act"), out: v("#f-out"),
    epochs: num("#f-epochs"), batch: num("#f-batch"), lr: num("#f-lr"), optimiser: v("#f-optimiser"),
    schedule: v("#f-schedule"), dropout: num("#f-dropout"), l2: num("#f-l2"), seed: num("#f-seed"),
    patience: num("#f-patience"), keepBest: $("#f-keepBest").checked, cv: num("#f-cv"),
  };
  if (d.generated) { p.rows = num("#f-rows"); p.noise = num("#f-noise"); }
  if (d.kind === "images" && v("#f-limit")) p.limit = num("#f-limit");
  if (d.kind === "upload") {
    if (v("#f-mode") === "series") { p.series = v("#f-target"); p.lags = num("#f-lags"); p.horizon = num("#f-horizon"); }
    else { p.target = v("#f-target"); p.features = [...document.querySelectorAll("#f-features input:checked")].map((x) => x.value); }
  }
  return p;
}

$("#train-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const d = dataset();
  if (!d.available) return toast("Download this dataset first: scripts/fetch_data.sh", "error");
  const p = trainPayload();
  if (d.kind === "upload" && p.features && !p.features.length) return toast("Pick at least one input column.", "error");
  const bad = [["epochs", 1], ["batch", 1], ["lr", 0]].find(([k, lo]) => !isNum(p[k]) || p[k] < lo || (k === "lr" && p[k] === 0));
  if (bad) return toast(`Set ${bad[0] === "lr" ? "a learning rate above 0" : bad[0] + " to at least " + bad[1]}.`, "error");
  if (p.rows !== undefined && !(p.rows >= 20)) return toast("Use at least 20 rows.", "error");
  startRun(p);
  try {
    const rep = await call("train", p, onTrainMessage, true);
    finishRun(rep);
  } catch (e) {
    state.live.done = true;
    toast("Training failed: " + e.message, "error");
    $("#run-meta").textContent = "failed: " + e.message;
  }
});

function startRun(p) {
  state.live = { history: [], epochs: p.epochs, task: null, run: null, cv: null, done: false };
  state.current = null;
  $("#train-empty").hidden = true;
  $("#train-out").hidden = false;
  $("#run-title").textContent = "Training on " + p.data;
  $("#run-meta").textContent = "starting";
  $("#run-progress").style.width = "0%";
  for (const id of ["#boundary-card", "#confusion-card", "#cv-card", "#layers-card"]) $(id).hidden = true;
  $("#tiles").replaceChildren();
  drawHistory([], "classes");
}

function onTrainMessage(m) {
  const L = state.live;
  if (m.type === "start") {
    L.task = m.task; L.run = m.run; L.epochs = m.epochs;
    $("#run-title").textContent = `Run ${m.run}: ${m.data}`;
    $("#run-meta").textContent = `${m.arch} · ${m.rows.toLocaleString()} training rows, ${(m.valRows || 0).toLocaleString()} validation`;
  } else if (m.type === "cv") {
    L.cv = m;
    renderCV(m);
    $("#run-meta").textContent += " · cross-validated";
  } else if (m.type === "epoch") {
    L.history.push(m.row);
    $("#run-progress").style.width = (100 * L.history.length / L.epochs).toFixed(1) + "%";
    scheduleDraw();
  }
}

let drawPending = false;
function scheduleDraw() {
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(() => {
    drawPending = false;
    const L = state.live;
    if (L.done) return;
    drawHistory(L.history, L.task);
    renderTiles(liveTiles(L));
  });
}

function liveTiles(L) {
  const r = L.history[L.history.length - 1] || {};
  const t = L.history.reduce((a, x) => a + x.seconds, 0);
  if (L.task === "regression") return [
    ["Validation loss", fmt(r.valLoss), "epoch " + (r.epoch || 0)],
    ["Training loss", fmt(r.trainLoss), ""],
    ["Epochs", `${L.history.length} / ${L.epochs}`, ""],
    ["Time", secs(t), ""]];
  return [
    ["Validation accuracy", pct(r.valAcc), "epoch " + (r.epoch || 0)],
    ["Validation loss", fmt(r.valLoss), ""],
    ["Epochs", `${L.history.length} / ${L.epochs}`, ""],
    ["Time", secs(t), ""]];
}

function renderTiles(tiles) {
  $("#tiles").replaceChildren(...tiles.map(([label, value, note, cls]) =>
    h("div", { class: "tile" }, h("div", { class: "label" }, label), h("div", { class: "value", title: value }, value),
      h("div", { class: "note" + (cls ? " " + cls : "") }, note || " "))));
}

function finishRun(rep) {
  state.live.done = true;
  state.reports[rep.run] = rep;
  state.current = rep.run;
  showReport(rep);
  call("runs").then((m) => { state.runs = m.runs; renderRuns(); renderPredictRuns(); });
}

function showReport(rep) {
  $("#train-empty").hidden = true;
  $("#train-out").hidden = false;
  const row = rep.row || {};
  $("#run-title").textContent = `Run ${rep.run}: ${row.data}`;
  $("#run-meta").textContent = `${row.arch} · ${row.epochs} epochs` + (isNum(rep.bestEpoch) ? ` · best epoch ${rep.bestEpoch}` : "");
  $("#run-progress").style.width = "100%";
  const task = row.task;
  drawHistory(rep.history, task);
  if (task === "regression" && rep.fit && rep.fit.actual) drawFit(rep.fit);
  const va = rep.val || {}, tr = rep.train || {};
  const total = (rep.history || []).reduce((a, x) => a + x.seconds, 0);
  if (task === "regression") {
    const scale = isNum(rep.unitScale) ? rep.unitScale : 1;
    const naive = isNum(rep.naiveMae) ? rep.naiveMae : null;
    const mae = isNum(va.mae) ? va.mae * scale : null;
    renderTiles([
      ["Validation R²", fmt(va.r2, 3), "training " + fmt(tr.r2, 3)],
      ["Validation MAE", fmt(mae), naive ? `naive forecast ${fmt(naive)}` : "in target units",
        naive && mae < naive ? "good" : ""],
      ["Validation loss", fmt(va.loss), "training " + fmt(tr.loss)],
      ["Time", secs(total), row.epochs + " epochs"]]);
  } else {
    renderTiles([
      ["Validation accuracy", pct(va.acc), "training " + pct(tr.acc)],
      ["Validation loss", fmt(va.loss), "training " + fmt(tr.loss)],
      ["Epochs", String(row.epochs), isNum(rep.bestEpoch) ? "best " + rep.bestEpoch : ""],
      ["Time", secs(total), ""]]);
  }
  const hasPlot = rep.plot && rep.plot.n;
  $("#boundary-card").hidden = !hasPlot;
  if (hasPlot) drawBoundary(rep.plot);
  const hasConf = rep.confusion && rep.confusion.length;
  $("#confusion-card").hidden = !hasConf;
  if (hasConf) drawConfusion(rep.confusion, rep.labels);
  $("#layers-card").hidden = false;
  renderTable($("#layers"), rep.layers, [["layer", "Layer"], ["inputs", "Inputs", "num"], ["outputs", "Outputs", "num"], ["act", "Activation"], ["params", "Parameters", "num"]]);
  $("#save-name").value = (row.data || "model") + "_" + rep.run;
  if (state.live && state.live.cv && state.live.run === rep.run) renderCV(state.live.cv); else $("#cv-card").hidden = true;
}

function renderCV(m) {
  $("#cv-card").hidden = false;
  const box = $("#cv");
  box.replaceChildren();
  const sum = m.summary.filter((r) => isNum(r.mean));
  const t = h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Score"), h("th", { class: "num" }, "Mean"), h("th", { class: "num" }, "Spread (sd)"))),
    h("tbody", {}, sum.map((r) => h("tr", {}, h("td", {}, { valLoss: "Validation loss", valAcc: "Validation accuracy", valR2: "Validation R²" }[r.metric] || r.metric),
      h("td", { class: "num" }, r.metric === "valAcc" ? pct(r.mean) : fmt(r.mean)), h("td", { class: "num" }, r.metric === "valAcc" ? pct(r.sd) : fmt(r.sd))))));
  box.append(t, h("p", { class: "hint" }, `${m.folds.length} folds, each validated on the rows the others trained on.`));
}

// save from the layers card
$("#save-run").addEventListener("click", () => saveRun(state.current, $("#save-name").value));
$("#predict-run").addEventListener("click", () => { $("#p-run").value = state.current; showView("predict"); renderPredictForm(); });

async function saveRun(id, name) {
  if (!id) return;
  try {
    const m = await call("save", { run: id, name });
    state.models = m.models;
    renderModels();
    toast(`Saved as models/${m.name}.qsm`);
  } catch (e) { toast("Save failed: " + e.message, "error"); }
}

// ---------- charts ----------

const tip = h("div", { class: "tooltip" });
document.body.append(tip);
function showTip(x, y, title, rows) {
  tip.replaceChildren(h("div", { class: "t" }, title), ...rows.map(([c, name, v]) =>
    h("div", { class: "r" }, c ? h("i", { style: `background:${c}` }) : null, h("b", {}, v), h("span", {}, name))));
  tip.style.display = "block";
  const r = tip.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > innerHeight - 8) top = y - r.height - 14;
  tip.style.left = left + scrollX + "px";
  tip.style.top = top + scrollY + "px";
}
const hideTip = () => { tip.style.display = "none"; };

// round axis ticks
function ticks(lo, hi, n = 4) {
  if (!(hi > lo)) { hi = lo + 1; }
  const step0 = (hi - lo) / n;
  const p = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((x) => x >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

// line chart: lines = [{name, colour, points: [[x, y]]}]; opts: yMin, yMax, yFmt, xLabel
function lineChart(el, lines, opts = {}) {
  el.replaceChildren();
  const W = Math.max(el.clientWidth, 280), H = 220, m = { l: 48, r: 56, t: 8, b: 24 };
  const pts = lines.flatMap((l) => l.points.filter((p) => isNum(p[1])));
  const yf = opts.yFmt || ((v) => fmt(v, 3));
  if (lines.length > 1) el.append(h("div", { class: "legend" }, lines.map((l) => h("span", {}, h("i", { style: `background:${l.colour}` }), l.name))));
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || "chart" });
  el.append(svg);
  if (!pts.length) {
    svg.append(s("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "tick" }, "waiting for the first epoch"));
    return;
  }
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  let y0 = opts.yMin ?? Math.min(...ys), y1 = opts.yMax ?? Math.max(...ys);
  if (opts.yMin === undefined && y0 > 0 && y0 < (y1 - y0) * 0.5) y0 = 0;
  if (y1 === y0) y1 = y0 + 1;
  const yt = ticks(y0, y1);
  y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]);
  const X = (v) => m.l + (W - m.l - m.r) * (v - x0) / (x1 - x0);
  const Y = (v) => H - m.b - (H - m.t - m.b) * (v - y0) / (y1 - y0);
  for (const t of yt) {
    svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: t === yt[0] ? "baseline" : "gridline" }));
    svg.append(s("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end", class: "tick" }, yf(t)));
  }
  const xt = ticks(x0, x1, 5).filter((t) => Number.isInteger(t) && t >= x0);
  for (const t of xt) svg.append(s("text", { x: X(t), y: H - 6, "text-anchor": "middle", class: "tick" }, t));
  for (const l of lines) {
    const p = l.points.filter((q) => isNum(q[1]));
    if (!p.length) continue;
    svg.append(s("path", { d: p.map((q, i) => (i ? "L" : "M") + X(q[0]).toFixed(1) + "," + Y(q[1]).toFixed(1)).join(""), class: "line", stroke: l.colour }));
    const e = p[p.length - 1];
    svg.append(s("circle", { cx: X(e[0]), cy: Y(e[1]), r: 4, fill: l.colour, class: "end" }));
  }
  // end labels, unless they would collide
  const ends = lines.map((l) => { const p = l.points.filter((q) => isNum(q[1])); return p.length ? { l, y: Y(p[p.length - 1][1]), v: p[p.length - 1][1] } : null; }).filter(Boolean);
  const apart = ends.every((a, i) => ends.every((b, j) => i === j || Math.abs(a.y - b.y) > 13));
  if (apart && ends.length <= 4) for (const e of ends) svg.append(s("text", { x: W - m.r + 8, y: e.y + 4, class: "endlabel" }, yf(e.v)));
  // crosshair and tooltip
  const cross = s("line", { y1: m.t, y2: H - m.b, class: "cross", visibility: "hidden" });
  svg.append(cross);
  const hit = s("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
  svg.append(hit);
  const allX = [...new Set(xs)].sort((a, b) => a - b);
  hit.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect();
    const vx = x0 + (x1 - x0) * ((ev.clientX - r.left) * W / r.width - m.l) / (W - m.l - m.r);
    const nx = allX.reduce((a, b) => (Math.abs(b - vx) < Math.abs(a - vx) ? b : a), allX[0]);
    cross.setAttribute("x1", X(nx)); cross.setAttribute("x2", X(nx)); cross.setAttribute("visibility", "visible");
    showTip(ev.clientX, ev.clientY, (opts.xLabel || "epoch") + " " + nx,
      lines.map((l) => { const q = l.points.find((p) => p[0] === nx); return [l.colour, l.name, q && isNum(q[1]) ? yf(q[1]) : "–"]; }));
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
}

function drawHistory(hist, task) {
  const ser = (k) => hist.map((r) => [r.epoch, r[k]]);
  const hasVal = hist.some((r) => isNum(r.valLoss));
  lineChart($("#chart-loss"), [{ name: "training", colour: series(0), points: ser("trainLoss") }]
    .concat(hasVal ? [{ name: "validation", colour: series(1), points: ser("valLoss") }] : []), { label: "loss per epoch" });
  const cls = task !== "regression";
  $("#acc-card").hidden = !cls;
  if (cls) $("#acc-title").replaceChildren("Accuracy ", h("span", { class: "sub" }, "per epoch"));
  if (cls) lineChart($("#chart-acc"), [{ name: "training", colour: series(0), points: ser("trainAcc") }]
    .concat(hasVal ? [{ name: "validation", colour: series(1), points: ser("valAcc") }] : []), { yMax: 1, yFmt: pct, label: "accuracy per epoch" });
}

// regression fit: forecast and actual over time for a series, else predicted against actual per row
function drawFit(f) {
  $("#acc-card").hidden = false;
  const el = $("#chart-acc");
  if (f.series) {
    $("#acc-title").replaceChildren("Forecast and actual ", h("span", { class: "sub" }, `last ${f.actual.length} validation steps`));
    lineChart(el, [{ name: "actual", colour: series(0), points: f.actual.map((v, i) => [i + 1, v]) },
      { name: "forecast", colour: series(1), points: f.predicted.map((v, i) => [i + 1, v]) }], { xLabel: "step", label: "forecast and actual" });
    return;
  }
  $("#acc-title").replaceChildren("Predicted and actual ", h("span", { class: "sub" }, `${f.actual.length} validation rows; on the line is perfect`));
  scatterChart(el, f.actual.map((a, i) => [a, f.predicted[i]]));
}

function scatterChart(el, pts) {
  el.replaceChildren();
  const W = Math.max(el.clientWidth, 280), H = 260, m = { l: 52, r: 16, t: 8, b: 28 };
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "predicted against actual" });
  el.append(svg);
  const all = pts.flat().filter(isNum);
  let lo = Math.min(...all), hi = Math.max(...all);
  const t = ticks(lo, hi);
  lo = Math.min(lo, t[0]); hi = Math.max(hi, t[t.length - 1]);
  const X = (v) => m.l + (W - m.l - m.r) * (v - lo) / (hi - lo), Y = (v) => H - m.b - (H - m.t - m.b) * (v - lo) / (hi - lo);
  for (const v of t) {
    svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: v === t[0] ? "baseline" : "gridline" }));
    svg.append(s("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "tick" }, fmt(v)));
    svg.append(s("text", { x: X(v), y: H - 8, "text-anchor": "middle", class: "tick" }, fmt(v)));
  }
  svg.append(s("line", { x1: X(lo), y1: Y(lo), x2: X(hi), y2: Y(hi), stroke: css("--axis"), "stroke-width": 1 }));
  const c = series(0), ring = css("--surface");
  for (const [a, p] of pts) svg.append(s("circle", { cx: X(a), cy: Y(p), r: 4, fill: c, "fill-opacity": 0.75, stroke: ring, "stroke-width": 1 }));
  svg.append(s("text", { x: W - m.r, y: H - m.b - 6, "text-anchor": "end", class: "tick" }, "actual →"));
  svg.append(s("text", { x: m.l + 6, y: m.t + 10, class: "tick" }, "↑ predicted"));
  // nearest point under the pointer
  const hit = s("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
  svg.append(hit);
  hit.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width, py = (ev.clientY - r.top) * H / r.height;
    let best = null, bd = Infinity;
    for (const q of pts) { const d = (X(q[0]) - px) ** 2 + (Y(q[1]) - py) ** 2; if (d < bd) { bd = d; best = q; } }
    if (best && bd < 900) showTip(ev.clientX, ev.clientY, "validation row", [[null, "actual", fmt(best[0])], [null, "predicted", fmt(best[1])]]);
    else hideTip();
  });
  hit.addEventListener("pointerleave", hideTip);
}

// decision boundary from the server's grid: cells tinted by predicted class, points on top
function drawBoundary(p) {
  const box = $("#boundary");
  box.replaceChildren();
  const size = Math.min(Math.max(box.clientWidth, 240), 520);
  const dpr = window.devicePixelRatio || 1;
  const c = h("canvas", { width: size * dpr, height: size * dpr, style: `max-width:${size}px`, role: "img", "aria-label": "decision boundary" });
  box.append(c);
  const g = c.getContext("2d");
  g.scale(dpr, dpr);
  const n = p.n, k = Math.max(2, ...p.classes.map((x) => x + 1));
  const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const bg = rgb(css("--surface")), cols = [...Array(k).keys()].map((i) => rgb(series(i)));
  const small = document.createElement("canvas");
  small.width = small.height = n;
  const img = small.getContext("2d").createImageData(n, n);
  for (let i = 0; i < n * n; i++) {
    const a = 0.12 + 0.5 * Math.max(0, (p.prob[i] - 1 / k) / (1 - 1 / k)), c = cols[p.classes[i]];
    for (let j = 0; j < 3; j++) img.data[4 * i + j] = Math.round(bg[j] + a * (c[j] - bg[j]));
    img.data[4 * i + 3] = 255;
  }
  small.getContext("2d").putImageData(img, 0, 0);
  g.imageSmoothingEnabled = true;
  g.drawImage(small, 0, 0, size, size);
  const px = (x, y) => [size * (x - p.lo[0]) / (p.hi[0] - p.lo[0]), size * (1 - (y - p.lo[1]) / (p.hi[1] - p.lo[1]))];
  const ring = css("--surface");
  for (const [x, y, cl] of p.train || []) {
    const [a, b] = px(x, y);
    g.beginPath(); g.arc(a, b, 3.5, 0, 2 * Math.PI); g.fillStyle = series(cl); g.fill();
    g.lineWidth = 1.5; g.strokeStyle = ring; g.stroke();
  }
  for (const [x, y, cl] of p.val || []) {
    const [a, b] = px(x, y);
    g.beginPath(); g.arc(a, b, 3.5, 0, 2 * Math.PI); g.lineWidth = 2; g.strokeStyle = series(cl); g.stroke();
  }
  c.addEventListener("pointermove", (ev) => {
    const r = c.getBoundingClientRect();
    const u = (ev.clientX - r.left) / r.width, v = (ev.clientY - r.top) / r.height;
    const i = Math.min(n - 1, Math.floor(v * n)) * n + Math.min(n - 1, Math.floor(u * n));
    const x = p.lo[0] + u * (p.hi[0] - p.lo[0]), y = p.hi[1] - v * (p.hi[1] - p.lo[1]);
    showTip(ev.clientX, ev.clientY, `x ${fmt(x, 2)}, y ${fmt(y, 2)}`, [[series(p.classes[i]), "class " + p.classes[i], pct(p.prob[i])]]);
  });
  c.addEventListener("pointerleave", hideTip);
  box.append(h("div", { class: "legend", style: "margin-top:8px" }, [...Array(k).keys()].map((i) => h("span", {}, h("i", { class: "box", style: `background:${series(i)}` }), "class " + i))));
}

// confusion matrix as a one-hue heatmap
function drawConfusion(M, labels) {
  const box = $("#confusion");
  box.replaceChildren();
  const k = M.length, max = Math.max(1, ...M.flat());
  // one-hue ramp from --seq-1 to --seq-9, with text dark or white by the cell's luminance
  const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const lo = rgb(css("--seq-1")), hi = rgb(css("--seq-9")), zero = rgb(css("--seq-0"));
  const fill = (t) => (t <= 0 ? zero : lo.map((v, i) => Math.round(v + t * (hi[i] - v))));
  const lum = (c) => c.map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }).reduce((a, v, i) => a + v * [0.2126, 0.7152, 0.0722][i], 0);
  const grid = h("div", { class: "heat", style: `grid-template-columns: 56px repeat(${k}, minmax(0, 1fr)); max-width:${Math.min(560, 60 + k * 64)}px` });
  grid.append(h("div", { class: "h" }, ""));
  for (let j = 0; j < k; j++) grid.append(h("div", { class: "h", title: labels[j] }, labels[j]));
  for (let i = 0; i < k; i++) {
    grid.append(h("div", { class: "h", title: labels[i] }, labels[i]));
    const total = M[i].reduce((a, b) => a + b, 0);
    for (let j = 0; j < k; j++) {
      const c = fill(M[i][j] / max);
      const cell = h("div", { class: "c", style: `background:rgb(${c});color:${lum(c) > 0.3 ? "#0b0b0b" : "#ffffff"}` }, k <= 12 ? M[i][j] : "");
      cell.addEventListener("pointermove", (ev) => showTip(ev.clientX, ev.clientY, `true ${labels[i]}, predicted ${labels[j]}`,
        [[null, total ? pct(M[i][j] / total) + " of this class" : "", M[i][j] + " rows"]]));
      cell.addEventListener("pointerleave", hideTip);
      grid.append(cell);
    }
  }
  box.append(grid, h("div", { class: "heat-axis", style: `max-width:${Math.min(560, 60 + k * 64)}px` }, h("span", {}, "rows: true class"), h("span", {}, "columns: predicted")));
}

// ---------- tables ----------

// cols: [key, header, class?, formatter?]
function renderTable(box, rows, cols, rowAttrs) {
  box.replaceChildren();
  if (!rows || !rows.length) { box.append(h("p", { class: "hint" }, "Nothing yet.")); return; }
  box.append(h("table", {}, h("thead", {}, h("tr", {}, cols.map(([, t, c]) => h("th", { class: c || null }, t)))),
    h("tbody", {}, rows.map((r, i) => h("tr", rowAttrs ? rowAttrs(r, i) : {}, cols.map(([k, , c, f]) => h("td", { class: c || null }, f ? f(r[k], r) : (isNum(r[k]) ? fmt(r[k]) : r[k] ?? "–"))))))));
}

// ---------- runs ----------

function colourOf(id) {
  if (!(id in state.colours)) {
    const used = new Set(Object.values(state.colours));
    let slot = 0;
    while (used.has(slot) && slot < 7) slot++;
    state.colours[id] = slot;
  }
  return series(state.colours[id]);
}

function renderRuns() {
  const box = $("#runs-table");
  const rows = [...state.runs].sort((a, b) => b.id - a.id);
  renderTable(box, rows, [
    ["id", "", null, (v, r) => {
      const cb = h("input", { type: "checkbox", checked: state.compare.includes(v), disabled: r.source !== "trained", "aria-label": "compare run " + v });
      cb.addEventListener("change", () => toggleCompare(v, cb.checked));
      return h("span", {}, cb, state.compare.includes(v) ? h("span", { class: "swatch", style: `background:${colourOf(v)};margin-left:6px` }) : null);
    }],
    ["id", "Run", "num"],
    ["data", "Data"],
    ["arch", "Model"],
    ["task", "Task", null, (v) => h("span", { class: "pill" }, v)],
    ["epochs", "Epochs", "num"],
    ["valLoss", "Val loss", "num"],
    ["valAcc", "Val accuracy", "num", (v) => pct(v)],
    ["valR2", "Val R²", "num", (v) => fmt(v)],
    ["seconds", "Time", "num", (v) => secs(v)],
    ["id", "", null, (v, r) => h("span", { class: "btns" },
      r.source === "trained" ? h("button", { class: "btn small", onclick: () => openRun(v) }, "Open") : null,
      h("button", { class: "btn small", onclick: () => { renderPredictRuns(); $("#p-run").value = v; showView("predict"); renderPredictForm(); } }, "Predict"),
      h("button", { class: "btn small", onclick: () => forget(v) }, "Remove"))],
  ]);
  drawCompare();
}

async function getReport(id) {
  if (!state.reports[id]) state.reports[id] = await call("report", { run: id });
  return state.reports[id];
}

async function openRun(id) {
  try { const rep = await getReport(id); state.current = id; showView("train"); showReport(rep); }
  catch (e) { toast(e.message, "error"); }
}

async function toggleCompare(id, on) {
  state.compare = state.compare.filter((x) => x !== id);
  if (on) {
    if (state.compare.length >= 8) { toast("Compare at most 8 runs at a time.", "error"); renderRuns(); return; }
    state.compare.push(id);
    try { await getReport(id); } catch (e) { toast(e.message, "error"); }
  } else delete state.colours[id];
  renderRuns();
}

function drawCompare() {
  const ids = state.compare.filter((id) => state.reports[id]);
  $("#compare-card").hidden = !ids.length;
  if (!ids.length) return;
  lineChart($("#chart-compare"), ids.map((id) => ({
    name: `run ${id} · ${state.reports[id].row.data} ${state.reports[id].row.arch}`, colour: colourOf(id),
    points: state.reports[id].history.map((r) => [r.epoch, r.valLoss]),
  })), { label: "validation loss per epoch" });
}

async function forget(id) {
  try {
    const m = await call("forget", { run: id });
    state.runs = m.runs;
    delete state.reports[id];
    state.compare = state.compare.filter((x) => x !== id);
    delete state.colours[id];
    renderRuns(); renderPredictRuns();
  } catch (e) { toast(e.message, "error"); }
}

// ---------- search ----------

function searchSpace() {
  const list = (id, f) => $(id).value.split(",").map((x) => x.trim()).filter(Boolean).map(f);
  const sp = {};
  const lr = list("#s-lr", Number).filter(isNum); if (lr.length) sp.lr = lr;
  const hid = $("#s-hidden").value.split(";").map((x) => parseLayers(x)).filter((x, i, a) => a.findIndex((y) => y.join() === x.join()) === i);
  if ($("#s-hidden").value.trim()) sp.hidden = hid;
  const act = list("#s-act", String); if (act.length) sp.act = act;
  const b = list("#s-batch", Number).filter(isNum); if (b.length) sp.batch = b;
  const d = list("#s-dropout", Number).filter(isNum); if (d.length) sp.dropout = d;
  return sp;
}

function updateSearchCount() {
  const sp = searchSpace();
  const n = Object.values(sp).reduce((a, v) => a * v.length, 1);
  const t = Number($("#s-trials").value) || n;
  $("#search-count").textContent = Object.keys(sp).length ? `${Math.min(n, t)} of ${n} configurations` : "Give at least one list of values.";
}
for (const id of ["#s-lr", "#s-hidden", "#s-act", "#s-batch", "#s-dropout", "#s-trials"]) $(id).addEventListener("input", updateSearchCount);

$("#search-btn").addEventListener("click", async () => {
  const sp = searchSpace();
  if (!Object.keys(sp).length) return toast("Give at least one list of values to try.", "error");
  const d = dataset();
  if (!d.available) return toast("Download this dataset first: scripts/fetch_data.sh", "error");
  const p = { ...trainPayload(), space: sp, trials: $("#s-trials").value ? Number($("#s-trials").value) : null };
  delete p.cv;
  state.searchRows = []; state.searchBest = null;
  $("#search-empty").hidden = true; $("#search-out").hidden = false; $("#train-best").disabled = true;
  $("#search-progress").style.width = "0%";
  let total = 1, keys = [];
  const draw = (final) => {
    const rows = final ? state.searchRows : state.searchRows;
    const cls = rows.some((r) => isNum(r.valAcc));
    renderTable($("#search-table"), rows, keys.map((k) => [k, k, k === "hidden" || k === "act" ? null : "num", (v) => Array.isArray(v) ? (v.length ? v.join("-") : "none") : (isNum(v) ? fmt(v, 4) : v)])
      .concat([["valLoss", "Val loss", "num"], cls ? ["valAcc", "Val accuracy", "num", pct] : ["valR2", "Val R²", "num"], ["bestEpoch", "Best epoch", "num"], ["seconds", "Time", "num", secs]]),
      (r, i) => (final && i === 0 ? { class: "best" } : {}));
  };
  try {
    const res = await call("search", p, (m) => {
      if (m.type === "searching") { total = m.count; keys = m.keys; $("#search-title").textContent = `Results: ${m.count} configurations on ${d.name}`; }
      if (m.type === "trial") { state.searchRows.push(m.row); $("#search-progress").style.width = (100 * state.searchRows.length / total) + "%"; draw(false); }
    }, true);
    state.searchRows = res.results;
    state.searchBest = res.results[0];
    draw(true);
    $("#search-title").textContent = `Results: ${res.results.length} configurations on ${d.name}, best first`;
    $("#train-best").disabled = false;
  } catch (e) { toast("Search failed: " + e.message, "error"); }
});

$("#train-best").addEventListener("click", () => {
  const b = state.searchBest;
  if (!b) return;
  const p = {};
  if ("hidden" in b) p.hidden = b.hidden.join(", ");
  for (const k of ["act", "lr", "batch", "dropout"]) if (k in b) p[k] = b[k];
  p.out = $("#f-out").value;
  applySettings(p);
  showView("train");
  toast("The best settings are in the Train form: press Train.");
});

// ---------- predict ----------

function renderPredictRuns() {
  const sel = $("#p-run");
  const prev = sel.value;
  sel.replaceChildren(...[...state.runs].sort((a, b) => b.id - a.id).map((r) => h("option", { value: r.id }, `run ${r.id} · ${r.data} · ${r.arch}`)));
  if (state.runs.some((r) => String(r.id) === prev)) sel.value = prev;
  renderPredictForm();
}
$("#p-run").addEventListener("change", () => renderPredictForm());

function runInfo(id) {
  const rep = state.reports[id];
  if (rep) return { features: rep.features || [], nIn: rep.layers[0].inputs, target: null };
  return state.loaded && state.loaded[id] ? state.loaded[id] : null;
}

async function renderPredictForm() {
  const form = $("#predict-form");
  const id = Number($("#p-run").value);
  form.replaceChildren();
  $("#predict-out").replaceChildren(h("div", {}, "Fill in the inputs and press Predict."));
  if (!id) { form.append(h("p", { class: "hint" }, "Train or load a model first.")); return; }
  let info = runInfo(id);
  if (!info) { try { await getReport(id); info = runInfo(id); } catch { info = null; } }
  if (!info) { form.append(h("p", { class: "hint" }, "No details for this run.")); return; }
  $("#batch-card").hidden = !info.features.length;
  if (info.features.length) {
    for (const f of info.features) {
      const fid = "p-in-" + f.col;
      const input = f.kind === "cat"
        ? h("select", { id: fid, "data-col": f.col, "data-kind": "cat" }, h("option", { value: "" }, "(missing)"), f.levels.map((l) => h("option", { value: l }, l)))
        : h("input", { type: "number", step: "any", id: fid, "data-col": f.col, "data-kind": "num", placeholder: "average " + fmt(f.mu) });
      form.append(h("div", { class: "field" }, h("label", { for: fid }, f.col), input));
    }
    $("#p-csv").placeholder = info.features.map((f) => f.col).join(",") + "\n...";
  } else {
    form.append(h("div", { class: "field" }, h("label", { for: "p-inputs" }, `${info.nIn} input value${info.nIn > 1 ? "s" : ""}, comma-separated`),
      h("input", { type: "text", id: "p-inputs", placeholder: Array(info.nIn).fill("0").join(", ") })));
    if (info.nIn > 20) form.append(h("p", { class: "hint" }, "This model takes many inputs (an image?): predict from q instead, or paste them all."));
  }
}

$("#predict-btn").addEventListener("click", async () => {
  const id = Number($("#p-run").value);
  if (!id) return;
  const fields = [...document.querySelectorAll("#predict-form [data-col]")];
  let p;
  if (fields.length) {
    const row = {};
    for (const f of fields) row[f.dataset.col] = f.dataset.kind === "num" ? (f.value === "" ? null : Number(f.value)) : f.value;
    p = { run: id, rows: [row] };
  } else {
    const xs = $("#p-inputs").value.split(/[,\s]+/).filter(Boolean).map(Number);
    if (!xs.length || xs.some((x) => !isNum(x))) return toast("Enter numbers separated by commas.", "error");
    p = { run: id, inputs: [xs] };
  }
  try {
    const m = await call("predict", p);
    const out = $("#predict-out");
    out.replaceChildren();
    out.classList.remove("empty");
    if (m.rows) {
      const r = m.rows[0];
      out.append(h("div", { class: "big" }, isNum(r.prediction) ? fmt(r.prediction) : r.prediction),
        isNum(r.confidence) ? h("p", { class: "hint" }, `confidence ${pct(r.confidence)}`) : h("p", { class: "hint" }, "predicted value"));
    } else {
      const o = m.outputs[0];
      if (o.length === 1) out.append(h("div", { class: "big" }, fmt(o[0])), h("p", { class: "hint" }, "model output"));
      else {
        const best = o.indexOf(Math.max(...o));
        out.append(h("div", { class: "big" }, "class " + best), h("p", { class: "hint" }, `probability ${pct(o[best])}`),
          h("div", { class: "result-grid" }, o.map((v, i) => h("div", {}, h("span", { class: "swatch", style: `background:${series(i)}` }), `class ${i}: ${pct(v)}`))));
      }
    }
  } catch (e) { toast("Prediction failed: " + e.message, "error"); }
});

// a small CSV reader for pasted rows: commas, optional double quotes, header row first
function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  const split = (l) => { const out = []; let cur = "", q = false; for (const ch of l) { if (ch === '"') q = !q; else if (ch === "," && !q) { out.push(cur); cur = ""; } else cur += ch; } out.push(cur); return out.map((x) => x.trim()); };
  const head = split(lines[0] || "");
  return lines.slice(1).map((l) => Object.fromEntries(split(l).map((v, i) => [head[i], v])));
}

$("#predict-csv").addEventListener("click", async () => {
  const id = Number($("#p-run").value);
  const info = runInfo(id);
  if (!info || !info.features.length) return;
  const raw = parseCsv($("#p-csv").value);
  if (!raw.length) return toast("Paste a header row and at least one row.", "error");
  const missing = info.features.map((f) => f.col).filter((c) => !(c in raw[0]));
  if (missing.length) return toast("Missing columns: " + missing.join(", "), "error");
  const rows = raw.map((r) => Object.fromEntries(info.features.map((f) => [f.col, f.kind === "num" ? (r[f.col] === "" ? null : Number(r[f.col])) : r[f.col]])));
  try {
    const m = await call("predict", { run: id, rows });
    const cols = info.features.map((f) => [f.col, f.col, f.kind === "num" ? "num" : null]);
    cols.push(["prediction", "Prediction", "num", (v) => (isNum(v) ? fmt(v) : v)]);
    if (m.rows.some((r) => isNum(r.confidence))) cols.push(["confidence", "Confidence", "num", pct]);
    renderTable($("#predict-batch"), m.rows, cols);
  } catch (e) { toast("Prediction failed: " + e.message, "error"); }
});

// ---------- models ----------

function renderModels() {
  renderTable($("#models-table"), state.models, [
    ["name", "Name"], ["arch", "Layers"], ["params", "Parameters", "num", (v) => fmt(v)], ["target", "Predicts", null, (v) => v || "–"],
    ["name", "", null, (v) => h("button", { class: "btn small", onclick: () => loadModel(v) }, "Load")],
  ]);
}

async function loadModel(name) {
  try {
    const m = await call("load", { name });
    state.runs = m.runs;
    state.loaded = state.loaded || {};
    state.loaded[m.run] = { features: m.features || [], nIn: m.nIn, target: m.target };
    renderRuns(); renderPredictRuns();
    $("#p-run").value = m.run;
    await renderPredictForm();
    showView("predict");
    toast(`Loaded ${name} as run ${m.run}`);
  } catch (e) { toast("Load failed: " + e.message, "error"); }
}

// ---------- views and theme ----------

function showView(v) {
  for (const b of document.querySelectorAll("nav button")) b.setAttribute("aria-selected", b.dataset.view === v);
  for (const s of document.querySelectorAll(".view")) s.classList.toggle("active", s.id === "view-" + v);
  try { localStorage.setItem("qsolve.view", v); } catch {}
  if (v === "runs") drawCompare();
  if (v === "search") updateSearchCount();
}
for (const b of document.querySelectorAll("nav button")) b.addEventListener("click", () => showView(b.dataset.view));

function redraw() {
  if (state.live && !state.live.done) { drawHistory(state.live.history, state.live.task); }
  else if (state.current && state.reports[state.current]) showReport(state.reports[state.current]);
  drawCompare();
}

$("#theme").addEventListener("click", () => {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const cur = document.documentElement.dataset.theme || (dark ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("qsolve.theme", next); } catch {}
  redraw();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
let resizeTimer;
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(redraw, 150); });

$("#f-data").addEventListener("change", () => onDataChange(true));

try {
  const t = localStorage.getItem("qsolve.theme"); if (t) document.documentElement.dataset.theme = t;
  const v = localStorage.getItem("qsolve.view"); if (v) showView(v);
} catch {}

connect();
