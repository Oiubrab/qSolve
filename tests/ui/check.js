// Browser check for the dashboard: starts dashboard.q, drives headless Chromium through every view over
// the DevTools protocol, saves screenshots, and exits 1 on a page error or a step that doesn't finish.
//   node tests/ui/check.js [screenshot-dir]        (needs chromium and node 22+; default dir /tmp/qsolve-ui)
"use strict";
const { spawn, execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "../..");
const OUT = process.argv[2] || "/tmp/qsolve-ui";
const PORT = 20000 + (process.pid % 20000);
const DEBUG = PORT + 1;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
fs.mkdirSync(OUT, { recursive: true });

// a seasonal series to upload, as the user would from a file
function seriesCsv() {
  let s = "hour,demand\n", e = 0;
  for (let t = 0; t < 1500; t++) {
    e = 0.6 * e + 0.08 * (Math.random() * 2 - 1);
    const w = 2 * Math.PI * t / 24;
    s += `${t},${(100 + 20 * (0.001 * t + Math.sin(w) + 0.4 * Math.sin(2 * w) + e)).toFixed(3)}\n`;
  }
  return s;
}

async function scenario({ ev, until, shot, size, sleep }) {
  const pick = (name) => ev(`(document.querySelector('#f-data').value=${JSON.stringify(name)}, document.querySelector('#f-data').dispatchEvent(new Event('change')), true)`);
  const train = async () => { await ev(`document.querySelector('#train-btn').click(), true`); await sleep(300); await until(`state.live && state.live.done`, 120000); await sleep(300); };
  await shot("01-start");
  await pick("moons"); await train();
  await until(`!document.querySelector('#boundary-card').hidden && document.querySelector('#tiles').textContent.includes('Validation accuracy')`);
  await shot("02-moons");
  await pick("houses"); await ev(`document.querySelector('#f-cv').value='3', true`); await train();
  await ev(`document.querySelector('#f-cv').value='0', true`);
  await until(`document.querySelector('#tiles').textContent.includes('Validation R')`);
  await shot("03-houses");
  const csv = seriesCsv();
  await ev(`(() => { const dt = new DataTransfer(); dt.items.add(new File([${JSON.stringify(csv)}], "load.csv", {type: "text/csv"})); const i = document.querySelector('#upload-file'); i.files = dt.files; i.dispatchEvent(new Event('change')); return true; })()`);
  await until(`document.querySelector('#f-data').value === 'load'`);
  await ev(`(document.querySelector('#f-mode').value='series', document.querySelector('#f-mode').dispatchEvent(new Event('change')), document.querySelector('#f-target').value='demand', document.querySelector('#f-lags').value='48', document.querySelector('#f-hidden').value='', document.querySelector('#f-lr').value='0.003', document.querySelector('#f-epochs').value='40', true)`);
  await train();
  await until(`document.querySelector('#tiles').textContent.includes('naive forecast')`);
  await shot("04-series");
  await ev(`document.querySelector('nav button[data-view=runs]').click(), true`);
  await ev(`(document.querySelectorAll('#runs-table input[type=checkbox]:not([disabled])').forEach((c, i) => { if (i < 3) { c.checked = true; c.dispatchEvent(new Event('change')); } }), true)`);
  await until(`document.querySelectorAll('#chart-compare path.line').length === 3`);
  await shot("05-runs");
  await ev(`document.querySelector('nav button[data-view=train]').click(), true`);
  await pick("moons"); await ev(`document.querySelector('#f-epochs').value='20', true`);
  await ev(`(document.querySelector('nav button[data-view=search]').click(), document.querySelector('#search-btn').click(), true)`);
  await until(`!document.querySelector('#train-best').disabled`, 120000);
  await shot("06-search");
  await ev(`(document.querySelector('nav button[data-view=predict]').click(), document.querySelector('#p-run').value='2', document.querySelector('#p-run').dispatchEvent(new Event('change')), true)`);
  await until(`document.querySelector('#p-in-area')`);
  await ev(`(document.querySelector('#p-in-area').value='150', document.querySelector('#p-in-rooms').value='4', document.querySelector('#p-in-city').value='north', document.querySelector('#predict-btn').click(), true)`);
  await until(`document.querySelector('#predict-out .big')`);
  await ev(`(document.querySelector('#p-csv').value='area,rooms,age,city\\n100,2,30,south\\n180,5,,west\\n60,1,70,zzz', document.querySelector('#predict-csv').click(), true)`);
  await until(`document.querySelectorAll('#predict-batch tbody tr').length === 3`);
  await shot("07-predict");
  const name = "uicheck_" + process.pid;
  await ev(`(document.querySelector('nav button[data-view=train]').click(), openRun(2), true)`);
  await until(`document.querySelector('#run-title').textContent.startsWith('Run 2')`);
  await ev(`(document.querySelector('#save-name').value=${JSON.stringify(name)}, document.querySelector('#save-run').click(), true)`);
  await until(`state.models.some(m => m.name === ${JSON.stringify(name)})`);
  await ev(`document.querySelector('nav button[data-view=models]').click(), true`);
  await ev(`[...document.querySelectorAll('#models-table button')].find(b => b.closest('tr').textContent.includes(${JSON.stringify(name)})).click(), true`);
  await until(`document.querySelector('#view-predict').classList.contains('active') && document.querySelector('#p-in-area')`);
  fs.rmSync(path.join(ROOT, "models", name + ".qsm"), { force: true });
  await ev(`(document.documentElement.dataset.theme='dark', document.querySelector('nav button[data-view=train]').click(), openRun(1), true)`);
  await until(`!document.querySelector('#boundary-card').hidden`);
  await sleep(300);
  await shot("08-dark");
  await size(390, 844); await sleep(600);
  await shot("09-phone");
}

(async () => {
  const q = spawn("q", [path.join(ROOT, "dashboard.q"), "-q", "-port", String(PORT)], { stdio: "ignore" });
  const chrome = spawn("chromium", ["--headless=new", `--remote-debugging-port=${DEBUG}`, "--no-first-run", "--no-default-browser-check",
    "--user-data-dir=" + path.join(OUT, ".profile"), "--window-size=1400,1000", "--hide-scrollbars", "about:blank"], { stdio: "ignore" });
  let failed = false;
  const errors = [];
  try {
    let tabs = [];
    for (let i = 0; i < 50 && !tabs.length; i++) { try { tabs = (await (await fetch(`http://127.0.0.1:${DEBUG}/json`)).json()).filter((t) => t.type === "page"); } catch {} await sleep(200); }
    const ws = new WebSocket(tabs[0].webSocketDebuggerUrl);
    await new Promise((r) => (ws.onopen = r));
    let n = 0; const waits = new Map();
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.id && waits.has(m.id)) { waits.get(m.id)(m); waits.delete(m.id); }
      if (m.method === "Runtime.exceptionThrown") errors.push("exception: " + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
      if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") errors.push("console.error: " + m.params.args.map((a) => a.value ?? a.description).join(" "));
    };
    const cmd = (method, params = {}) => new Promise((r) => { const id = ++n; waits.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
    const ev = async (expr) => { const r = await cmd("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); if (r.result.exceptionDetails) throw new Error(expr.slice(0, 80) + ": " + r.result.exceptionDetails.exception?.description); return r.result.result.value; };
    const until = async (expr, ms = 30000) => { const t = Date.now(); while (Date.now() - t < ms) { if (await ev(expr)) return; await sleep(200); } throw new Error("timed out waiting for " + expr.slice(0, 100)); };
    const size = (w, h) => cmd("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: 1, mobile: w < 600 });
    const shot = async (name) => {
      const m = await cmd("Page.getLayoutMetrics");
      const { width, height } = m.result.cssContentSize;
      const r = await cmd("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: { x: 0, y: 0, width, height: Math.min(height, 4000), scale: 1 } });
      fs.writeFileSync(path.join(OUT, name + ".png"), Buffer.from(r.result.data, "base64"));
      console.log("ok  " + name);
    };
    await cmd("Runtime.enable"); await cmd("Page.enable"); await size(1400, 1000);
    for (let i = 0; i < 50; i++) { try { await fetch(`http://127.0.0.1:${PORT}/`); break; } catch { await sleep(200); } }
    await cmd("Page.navigate", { url: `http://127.0.0.1:${PORT}/` });
    await until(`document.querySelector('#conn-text') && document.querySelector('#conn-text').textContent === 'connected' && document.querySelectorAll('#f-data option').length > 5`);
    await scenario({ ev, until, shot, size, sleep });
    ws.close();
  } catch (e) { failed = true; console.log("FAIL " + e.message); }
  chrome.kill(); q.kill();
  if (errors.length) { failed = true; console.log("page errors:\n  " + errors.join("\n  ")); }
  console.log(failed ? "UI check failed" : `UI check passed; screenshots in ${OUT}`);
  process.exit(failed ? 1 : 0);
})();
