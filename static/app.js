// app.js — VitalSense live monitor (Flask-SocketIO + REST)
const socket = io();
const $ = (s, r = document) => r.querySelector(s);

const COLORS = { IR: "#8b5cf6", RED: "#ff7a45", HR: "#ff5c7a", SpO2: "#4cc9ff" };
let RANGE = {
  lo: { IR: 46575, RED: 67676, HR: 40, SpO2: 90 },
  hi: { IR: 1124224, RED: 89898, HR: 89, SpO2: 100 },
};
const ALERT_WINDOW = 3;

// ---- i18n helpers (T() comes from base.html) ---------------------------------
const NAMES = () => ({ IR: "IR", RED: "RED", HR: T("heart_beat"), SpO2: T("oxygen_level") });
const STATUS_WORD = w => ({ ok: T("st_ok"), LOW: T("st_low"), HIGH: T("st_high") }[w] || w);
// "ok/ok/LOW/HIGH" -> localised words (same slash-separated shape as before)
const localStatus = s => String(s).split("/").map(STATUS_WORD).join("/");
// "IR: OK · RED: OK · Heart beat: LOW · Oxygen level: OK" style text for banners
const channelStatus = s => String(s).split("/").map((w, i) => `${NAMES()[["IR", "RED", "HR", "SpO2"][i]]}: ${STATUS_WORD(w)}`).join(" · ");
const reasonText = d => d.reason_code === "missing" ? T("reason_missing") : d.reason_code === "tiny" ? T("reason_tiny") : T("reason_ok");
const judgedText = d => /^GAN prediction/.test(d.judged_on)
  ? T("judged_gan", { model: (d.judged_on.match(/\((.*)\)/) || [, ""])[1].replace("_", " ") })
  : T("judged_raw");

// ---- rolling buffers -------------------------------------------------------
const MAXPTS = 90;
const B = { ts: [], IR: [], RED: [], HR: [], SpO2: [], pIR: [], pRED: [], pHR: [], pSpO2: [] };
// GAN / NoGAN family averages per reading, for the comparison chart
const G = {}, N = {};
for (const c of ["IR", "RED", "HR", "SpO2"]) { G[c] = []; N[c] = []; }
const trim = () => [...Object.values(B), ...Object.values(G), ...Object.values(N)].forEach(a => { while (a.length > MAXPTS) a.shift(); });

// ---- charts ---------------------------------------------------------------
Chart.defaults.color = "#8e91a8";
Chart.defaults.font.family = "Inter, system-ui, sans-serif";
Chart.defaults.borderColor = "rgba(255,255,255,.06)";

function ds(label, data, color, dashed = false) {
  return {
    label, data, borderColor: color, borderWidth: dashed ? 2 : 2.5, tension: .35, pointRadius: 0,
    borderDash: dashed ? [6, 5] : [], spanGaps: false,
    backgroundColor: color + "22", fill: false,
  };
}
function makeChart(id, datasets) {
  const ctx = document.getElementById(id).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: { labels: B.ts, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { usePointStyle: true, boxWidth: 8, color: "#b9bbd0" } },
        tooltip: { backgroundColor: "#15151f", borderColor: "rgba(255,255,255,.12)", borderWidth: 1, padding: 12 },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 7 }, grid: { display: false } },
        y: { grid: { color: "rgba(255,255,255,.05)" } },
      },
    },
  });
}
const vitalChart = makeChart("vitalChart", [
  ds(T("heart_beat"), B.HR, COLORS.HR), ds(T("oxygen_level"), B.SpO2, COLORS.SpO2),
  ds(`${T("heart_beat")} (${T("ai_suffix")})`, B.pHR, COLORS.HR, true), ds(`${T("oxygen_level")} (${T("ai_suffix")})`, B.pSpO2, COLORS.SpO2, true),
]);
const sigChart = makeChart("sigChart", [
  ds("IR", B.IR, COLORS.IR), ds("RED", B.RED, COLORS.RED),
  ds(`IR (${T("ai_suffix")})`, B.pIR, COLORS.IR, true), ds(`RED (${T("ai_suffix")})`, B.pRED, COLORS.RED, true),
]);

// ---- GAN vs No-GAN comparison ---------------------------------------------
let cmpMetric = "HR";
const GANC = "#34f5c5", NOGANC = "#ff7a45";
const cmpChart = makeChart("cmpChart", [
  ds(T("sensor_actual_short"), B.HR, "#ffffff"),
  ds(T("gan_models"), G.HR, GANC),
  ds(T("nogan_models"), N.HR, NOGANC),
]);
cmpChart.data.datasets[0].borderWidth = 2; cmpChart.data.datasets[0].borderDash = [2, 4];
const cmpBar = new Chart(document.getElementById("cmpBar").getContext("2d"), {
  type: "bar",
  data: { labels: [`${T("heart_beat")} (${T("unit_bpm")})`, `${T("oxygen_level")} (%)`], datasets: [
    { label: "GAN", data: [0, 0], backgroundColor: GANC, borderRadius: 8 },
    { label: "No-GAN", data: [0, 0], backgroundColor: NOGANC, borderRadius: 8 },
  ]},
  options: { responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { usePointStyle: true, boxWidth: 8 } } },
    scales: { x: { grid: { display: false } }, y: { beginAtZero: true, title: { display: true, text: T("avg_error") } } } },
});
function setCmpMetric(c) {
  cmpMetric = c;
  document.querySelectorAll("#cmpTabs button").forEach(b => b.classList.toggle("on", b.dataset.c === c));
  const lab = NAMES()[c];
  cmpChart.data.datasets[0].data = B[c];
  cmpChart.data.datasets[1].data = G[c];
  cmpChart.data.datasets[2].data = N[c];
  cmpChart.data.datasets[0].label = T("sensor_actual", { name: lab });
  cmpChart.update();
}
document.querySelectorAll("#cmpTabs button").forEach(b => b.addEventListener("click", () => setCmpMetric(b.dataset.c)));

let _perfBusy = false;
async function loadCmpBars() {
  if (_perfBusy) return; _perfBusy = true;
  try {
    const d = await (await fetch("/api/performance")).json();
    const grp = (test) => {
      const ms = Object.keys(d).filter(m => test(m) && m !== "MEDIAN");
      const avg = k => ms.length ? ms.reduce((a, m) => a + d[m][k], 0) / ms.length : 0;
      return { hr: avg("mae_hr"), sp: avg("mae_spo2"), n: ms.length };
    };
    const g = grp(m => !m.includes("NoGAN") && m.includes("GAN")), n = grp(m => m.includes("NoGAN"));
    cmpBar.data.datasets[0].data = [g.hr, g.sp];
    cmpBar.data.datasets[1].data = [n.hr, n.sp];
    cmpBar.update();
    const v = $("#cmpVerdict");
    if (g.n && n.n) {
      const gs = g.hr + g.sp, ns = n.hr + n.sp;
      v.style.display = ""; v.className = "chip " + (gs <= ns ? "ok" : "warn");
      v.textContent = gs <= ns ? T("gan_close") : T("nogan_close");
    }
  } catch {} finally { _perfBusy = false; }
}
loadCmpBars(); setInterval(loadCmpBars, 15000);

// ---- connection chips -----------------------------------------------------
function chip(id, txtId, cls, text) {
  $("#" + id).className = "chip " + cls;
  $("#" + id + " .dot").classList.toggle("live", cls === "ok");
  $("#" + txtId).textContent = text;
}
socket.on("connect", () => chip("wsChip", "wsTxt", "ok", T("socket_live")));
socket.on("disconnect", () => chip("wsChip", "wsTxt", "bad", T("socket_off")));
socket.on("connect_error", () => chip("wsChip", "wsTxt", "bad", T("socket_err")));

async function refreshSerial() {
  try {
    const r = await fetch("/api/serial/status");
    if (r.status === 401) return (location.href = "/login");
    const s = await r.json();
    chip("serialChip", "serialTxt", s.running ? "ok" : "bad",
         s.running ? `${s.port} @ ${s.baud}` : T("serial_stopped"));
  } catch { chip("serialChip", "serialTxt", "bad", T("server_offline")); }
}
$("#serialBtn").addEventListener("click", async () => {
  await fetch("/api/serial/stop", { method: "POST" });
  const r = await fetch("/api/serial/start", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
  });
  toast((await r.json()).msg);
  refreshSerial();
});
refreshSerial(); setInterval(refreshSerial, 5000);

// ---- helpers --------------------------------------------------------------
function toast(msg, bad = false) {
  const t = document.createElement("div");
  t.className = "toast" + (bad ? " bad" : ""); t.textContent = msg;
  $("#toasts").appendChild(t); setTimeout(() => t.remove(), 5000);
}
const fmt = (c, v) => (c === "SpO2" ? v.toFixed(1) : Math.round(v).toLocaleString());
const band = (c, v) => v < RANGE.lo[c] ? "LOW" : (v > RANGE.hi[c] ? "HIGH" : "ok");
const statusPill = s =>
  /LOW|HIGH/.test(s) ? `<span class="pill ${/LOW/.test(s) && !/HIGH/.test(s) ? "bad" : "warn"}">${localStatus(s)}</span>`
                     : `<span class="pill">${localStatus(s)}</span>`;

function setTile(c, v) {
  const tile = $("#tile-" + c), b = band(c, v);
  $(".v", tile).textContent = fmt(c, v);
  tile.classList.toggle("s-bad", b === "LOW");
  tile.classList.toggle("s-warn", b === "HIGH");
  const lo = RANGE.lo[c], hi = RANGE.hi[c];
  $("#rng-" + c).textContent =
    `${T("normal_word")} ${Math.round(lo).toLocaleString()} – ${Math.round(hi).toLocaleString()}` +
    (b !== "ok" ? `  ·  ${STATUS_WORD(b)}` : "");
  // bar: position of value across 0..1.25*hi (clamped)
  const pct = Math.max(0, Math.min(100, (v / (hi * 1.25)) * 100));
  $(".bar b", tile).style.width = pct + "%";
}

// ---- "All models · latest prediction" table ---------------------------------
let lastPred = null;   // newest payload that actually contained predictions
const pickChosen = rs => rs.find(r => r.Model === "LSTM_GAN") || rs.find(r => r.Model === "MEDIAN") || rs[rs.length - 1];
function renderResultsTable(results, chosen, note = "") {
  const tb = $("#resultsTable tbody"); tb.innerHTML = "";
  results.forEach(r => {
    const tr = document.createElement("tr");
    if (String(r.Status).includes("LOW")) tr.classList.add("low");
    if (String(r.Status).includes("HIGH")) tr.classList.add("high");
    if (r === chosen) tr.classList.add("best");
    tr.innerHTML = `<td>${r.Model.replace("_", " ")}</td><td>${Math.round(r.IR).toLocaleString()}</td><td>${Math.round(r.RED).toLocaleString()}</td><td>${Math.round(r.HR)}</td><td>${Number(r.SpO2).toFixed(1)}</td><td>${statusPill(r.Status)}</td>`;
    tb.appendChild(tr);
  });
  if (note) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="6" class="empty" style="padding:10px 0;font-size:12px">${note}</td>`;
    tb.appendChild(tr);
  }
}

// ---- render ---------------------------------------------------------------
function render(d) {
  // tiles
  for (const c of ["IR", "RED", "HR", "SpO2"]) setTile(c, d.input[c]);

  // banner
  const st = $("#status");
  const set = (cls, ico, txt) => { st.className = "banner " + cls; st.innerHTML = `<span>${ico}</span><span>${txt}</span>`; };
  const stTxt = channelStatus(d.input_status);
  if (d.alert_fired) set("bad", "🚨", T("banner_alert", { n: ALERT_WINDOW, reason: stTxt }));
  else if (!d.predicted) set("warn", "🫳", T("banner_nosignal", { status: stTxt, reason: reasonText(d) }));
  else if (d.abnormal) set("warn", "⚠️", T("banner_abn", { status: stTxt, streak: d.streak, n: ALERT_WINDOW, judged: judgedText(d) }));
  else set("ok", "✅", T("banner_ok", { status: stTxt, ts: d.ts }));

  // results table (keeps the last good prediction when the current reading is skipped)
  let chosen = null;
  if (d.predicted && d.results && d.results.length) {
    lastPred = d;
    chosen = pickChosen(d.results);
    renderResultsTable(d.results, chosen);
  } else if (lastPred) {
    chosen = pickChosen(lastPred.results);
    renderResultsTable(lastPred.results, chosen, T("last_pred_note", { ts: lastPred.ts, reason: reasonText(d) }));
  } else {
    $("#resultsTable tbody").innerHTML = `<tr><td colspan="6" class="empty">${T("pred_skipped", { reason: reasonText(d) })}</td></tr>`;
  }

  // verdict card
  const vb = $("#verdictBig"), vs = $("#verdictSub");
  if (chosen) {
    const bad = /LOW|HIGH/.test(chosen.Status);
    vb.textContent = bad ? T("verdict_abn") : T("verdict_norm");
    vb.style.color = bad ? "var(--coral)" : "var(--mint)";
    vs.textContent = T("verdict_says", { model: chosen.Model.replace("_", " "), status: localStatus(chosen.Status) });
    $("#verdictTbl tbody").innerHTML =
      ["HR", "SpO2"].map(c => `<tr><td>${NAMES()[c]} (${T("ai_suffix")})</td><td>${fmt(c, chosen[c])}</td><td>${statusPill(band(c, chosen[c]))}</td></tr>`).join("");
  } else {
    vb.textContent = d.abnormal ? T("verdict_abn") : "—";
    vb.style.color = d.abnormal ? "var(--coral)" : "var(--muted)";
    vs.textContent = T("verdict_skipped");
    $("#verdictTbl tbody").innerHTML = "";
  }
  for (let i = 0; i < ALERT_WINDOW; i++)
    $("#pip" + i).className = "chip" + (i < d.streak ? " bad" : "");
  $("#verdictCard").style.borderColor = d.alert_fired ? "rgba(255,92,122,.7)" : "";

  // charts
  B.ts.push(d.ts);
  for (const c of ["IR", "RED", "HR", "SpO2"]) B[c].push(d.input[c]);
  const liveChosen = (d.predicted && d.results && d.results.length) ? chosen : null;
  for (const c of ["IR", "RED", "HR", "SpO2"]) B["p" + c].push(liveChosen ? liveChosen[c] : null);
  pushGroups(d);
  trim(); vitalChart.update(); sigChart.update(); cmpChart.update();

  if (d.alert_fired) {
    if (d.telegram_sent === false) toast(T("toast_tg_fail"), true);
    else toast(d.manual ? T("toast_manual") : T("toast_alert"), true);
    loadAlerts();
  }
}

socket.on("new_reading", render);

function pushGroups(d) {
  const rs = (d.predicted && d.results) ? d.results.filter(r => r.Model !== "MEDIAN") : [];
  const gm = rs.filter(r => r.Model.endsWith("_GAN")), nm = rs.filter(r => r.Model.endsWith("_NoGAN"));
  const mean = (arr, c) => arr.length ? arr.reduce((a, r) => a + r[c], 0) / arr.length : null;
  for (const c of ["IR", "RED", "HR", "SpO2"]) { G[c].push(mean(gm, c)); N[c].push(mean(nm, c)); }
}

// ---- manual form ----------------------------------------------------------
const form = $("#inputForm");
document.querySelectorAll(".quick button").forEach(b => b.addEventListener("click", () => {
  const v = JSON.parse(b.dataset.fill); for (const k in v) form[k].value = v[k];
}));
form.addEventListener("submit", async e => {
  e.preventDefault();
  const body = {}; new FormData(form).forEach((v, k) => body[k] = parseFloat(v));
  const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (r.ok) toast(T("pred_done"));
});

// ---- alerts ---------------------------------------------------------------
async function loadAlerts() {
  try {
    const rows = await (await fetch("/api/alerts")).json();
    const ul = $("#alertsList");
    if (!rows.length) return;
    ul.innerHTML = rows.map(a => `<li><small>${a.ts}</small>${a.message.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/\*/g, "")}</li>`).join("");
  } catch {}
}
loadAlerts(); setInterval(loadAlerts, 10000);

// ---- bootstrap: config + history -----------------------------------------
(async () => {
  try { const c = await (await fetch("/api/config")).json(); RANGE = { lo: c.lo, hi: c.hi }; } catch {}
  try {
    const p = await (await fetch("/api/latest_prediction")).json();
    if (p && p.results && p.results.length) {
      lastPred = p;
      const chosen = pickChosen(p.results);
      renderResultsTable(p.results, chosen, T("last_pred_loaded", { ts: p.ts || "—" }));
      const bad = /LOW|HIGH/.test(chosen.Status);
      $("#verdictBig").textContent = bad ? T("verdict_abn") : T("verdict_norm");
      $("#verdictBig").style.color = bad ? "var(--coral)" : "var(--mint)";
      $("#verdictSub").textContent = T("verdict_says", { model: chosen.Model.replace("_", " "), status: localStatus(chosen.Status) });
      $("#verdictTbl tbody").innerHTML =
        ["HR", "SpO2"].map(c => `<tr><td>${NAMES()[c]} (${T("ai_suffix")})</td><td>${fmt(c, chosen[c])}</td><td>${statusPill(band(c, chosen[c]))}</td></tr>`).join("");
    }
  } catch {}
  try {
    const rows = await (await fetch("/api/readings?limit=60")).json();
    rows.forEach(r => {
      B.ts.push((r.ts || "").slice(11, 19));
      for (const c of ["IR", "RED", "HR", "SpO2"]) { B[c].push(r[c]); B["p" + c].push(null); G[c].push(null); N[c].push(null); }
    });
    if (rows.length) { const l = rows[rows.length - 1]; for (const c of ["IR", "RED", "HR", "SpO2"]) setTile(c, l[c]); }
    trim(); vitalChart.update(); sigChart.update(); cmpChart.update();
  } catch {}
})();
