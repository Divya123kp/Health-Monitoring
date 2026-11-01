Chart.defaults.color = "#8e91a8";
Chart.defaults.font.family = "Inter, system-ui, sans-serif";
Chart.defaults.borderColor = "rgba(255,255,255,.06)";
const $ = s => document.querySelector(s);

const MINT = "#34f5c5", CORAL = "#ff7a45";
let charts = {};

function bar(id, labels, data, colors, label) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($(id), {
    type: "bar",
    data: { labels, datasets: [{ label, data, backgroundColor: colors, borderRadius: 10, borderSkipped: false }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { backgroundColor: "#15151f", padding: 12 } },
      scales: { x: { grid: { display: false } }, y: { beginAtZero: true } },
    },
  });
}

async function loadPerf() {
  const r = await fetch("/api/performance");
  if (r.status === 401) return (location.href = "/login");
  const data = await r.json();
  const models = Object.keys(data).sort();
  if (!models.length) {
    $("#perfTable tbody").innerHTML = `<tr><td colspan="6" class="empty">${T("lab_empty")}</td></tr>`;
    return;
  }
  const labels = models.map(m => m.replace("_", " "));
  const mae = models.map(m => (data[m].mae_ir + data[m].mae_red + data[m].mae_hr + data[m].mae_spo2) / 4);
  const acc = mae.map(x => 1 / (1 + x));
  const colors = models.map(m => m.includes("NoGAN") ? CORAL : MINT);

  bar("#maeChart", labels, mae, colors, T("lab_avg_mae"));
  bar("#accChart", labels, acc, colors, T("lab_acc"));

  const best = mae.indexOf(Math.min(...mae));
  const w = $("#winner"); w.style.display = ""; w.textContent = T("lab_best", { name: labels[best] });

  $("#perfTable tbody").innerHTML = models.map((m, i) => {
    const d = data[m];
    return `<tr class="${i === best ? "best" : ""}"><td>${labels[i]}</td><td>${d.mae_ir.toFixed(0)}</td><td>${d.mae_red.toFixed(0)}</td><td>${d.mae_hr.toFixed(2)}</td><td>${d.mae_spo2.toFixed(2)}</td><td>${d.n}</td></tr>`;
  }).join("");
}

async function loadPreds() {
  const rows = await (await fetch("/api/predictions?limit=200")).json();
  const word = w => ({ ok: T("st_ok"), LOW: T("st_low"), HIGH: T("st_high") }[w] || w);
  const loc = s => String(s).split("/").map(word).join("/");
  const pill = s => /LOW|HIGH/.test(s) ? `<span class="pill ${/HIGH/.test(s) && !/LOW/.test(s) ? "warn" : "bad"}">${loc(s)}</span>` : `<span class="pill">${loc(s)}</span>`;
  $("#predTable tbody").innerHTML = rows.reverse().map(r => {
    const cls = r.status.includes("LOW") ? "low" : (r.status.includes("HIGH") ? "high" : "");
    return `<tr class="${cls}"><td>${r.id}</td><td>${r.model.replace("_", " ")}</td><td>${Math.round(r.IR || 0).toLocaleString()}</td><td>${Math.round(r.RED || 0).toLocaleString()}</td><td>${Math.round(r.HR || 0)}</td><td>${(r.SpO2 || 0).toFixed(1)}</td><td>${pill(r.status)}</td><td>${r.ts}</td></tr>`;
  }).join("");
}

loadPerf(); loadPreds();
setInterval(loadPreds, 5000); setInterval(loadPerf, 15000);
