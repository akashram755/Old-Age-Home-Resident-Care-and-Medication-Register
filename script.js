/* Resident Care & Medication Register — front-end logic
   Reads records from data.json. No build step required. */

const state = {
  records: [],
  filtered: [],
  activeId: null,
  latestDate: null,
  predictions: {},   // record_id -> { predicted_risk: "Yes"|"No"|null, confidence, in_test_set }
  hasModel: false,
};

const els = {
  recordList: document.getElementById("recordList"),
  emptyState: document.getElementById("emptyState"),
  resultCount: document.getElementById("resultCount"),
  searchInput: document.getElementById("searchInput"),
  dateFilter: document.getElementById("dateFilter"),
  statusFilter: document.getElementById("statusFilter"),
  todayLabel: document.getElementById("todayLabel"),
  countDue: document.getElementById("countDue"),
  countGiven: document.getElementById("countGiven"),
  countFlag: document.getElementById("countFlag"),
  detailPlaceholder: document.getElementById("detailPlaceholder"),
  detailContent: document.getElementById("detailContent"),
  detailClose: document.getElementById("detailClose"),
  adherenceValue: document.getElementById("adherenceValue"),
  residentMeta: document.getElementById("residentMeta"),
  recordCard: document.getElementById("recordCard"),
  historyList: document.getElementById("historyList"),
};

init();

async function init() {
  els.recordList.innerHTML = `<p class="empty-state">Loading records…</p>`;
  try {
    const res = await fetch("data.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (!Array.isArray(data) || data.length === 0) throw new Error("Empty dataset");

    state.records = data;
    state.latestDate = data
      .map((r) => r.date)
      .filter(Boolean)
      .sort()
      .at(-1);

    // Model predictions are optional: if train_model.py hasn't been run yet,
    // the rest of the register still works fine without them.
    try {
      const predRes = await fetch("predictions.json");
      if (predRes.ok) {
        state.predictions = await predRes.json();
        state.hasModel = true;
      }
    } catch {
      state.hasModel = false;
    }

    buildDateFilterOptions();
    wireControls();
    applyFilters();
    renderBoardCounts();
    els.todayLabel.textContent = formatDate(state.latestDate) || "No dated records found";
  } catch (err) {
    els.recordList.innerHTML = `<p class="empty-state">
      Could not load <code>data.json</code> (${escapeHtml(err.message)}).<br/>
      If you opened this file directly in a browser, serve the folder instead
      — e.g. run <code>python3 -m http.server</code> in this folder and open
      <code>http://localhost:8000</code>.
    </p>`;
    els.todayLabel.textContent = "Unable to load data";
  }
}

function buildDateFilterOptions() {
  const dates = [...new Set(state.records.map((r) => r.date).filter(Boolean))].sort();
  for (const d of dates) {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = formatDate(d);
    els.dateFilter.appendChild(opt);
  }
  if (state.latestDate) els.dateFilter.value = state.latestDate;
}

function wireControls() {
  els.searchInput.addEventListener("input", applyFilters);
  els.dateFilter.addEventListener("change", applyFilters);
  els.statusFilter.addEventListener("change", applyFilters);
  els.detailClose.addEventListener("click", closeDetail);
}

function applyFilters() {
  const q = els.searchInput.value.trim().toLowerCase();
  const dateSel = els.dateFilter.value;
  const statusSel = els.statusFilter.value;

  state.filtered = state.records.filter((r) => {
    if (dateSel !== "all" && r.date !== dateSel) return false;

    if (statusSel === "due" && !(r.given === "No" || r.given === "")) return false;
    if (statusSel === "given" && r.given !== "Yes") return false;
    if (statusSel === "flagged" && r.needs_attention !== "Yes") return false;

    if (q) {
      const haystack = [
        r.resident_name,
        r.medication,
        r.observation,
        r.dose_time,
        r.resident_id != null ? String(r.resident_id) : "",
      ]
        .join(" ")
        .toLowerCase();
      if (!haystack.includes(q)) return false;
    }
    return true;
  });

  renderList();
}

function renderList() {
  const list = state.filtered;
  els.resultCount.textContent = `${list.length} record${list.length === 1 ? "" : "s"}`;
  els.emptyState.hidden = list.length !== 0;
  els.recordList.innerHTML = "";

  const sorted = [...list].sort((a, b) => {
    const d = (b.date || "").localeCompare(a.date || "");
    if (d !== 0) return d;
    return (a.dose_time || "").localeCompare(b.dose_time || "");
  });

  for (const r of sorted) {
    els.recordList.appendChild(buildRow(r));
  }
}

function buildRow(r) {
  const row = document.createElement("button");
  row.type = "button";
  row.className = "record-row";
  row.dataset.id = r.record_id;
  if (r.record_id === state.activeId) row.classList.add("is-active");

  const givenPill =
    r.given === "Yes"
      ? `<span class="pill pill--given">Given</span>`
      : r.given === "No"
      ? `<span class="pill pill--due">Due</span>`
      : `<span class="pill pill--unknown">Not recorded</span>`;

  const isFlagged = r.needs_attention === "Yes";
  const riskPill = buildRiskPill(r.record_id);

  row.innerHTML = `
    <span class="col-time">${escapeHtml(r.dose_time || "—")}</span>
    <span class="col-resident">${escapeHtml(r.resident_name || "Unlabeled entry")}
      <small>ID ${r.resident_id !== "" && r.resident_id != null ? escapeHtml(String(r.resident_id)) : "—"}</small>
    </span>
    <span class="col-med">${escapeHtml(r.medication || "No medication on this entry")}</span>
    <span class="col-status">${givenPill}</span>
    <span class="col-flag"><span class="flag-dot ${isFlagged ? "is-flagged" : ""}" title="${isFlagged ? "Needs attention" : "No flag"}"></span></span>
    <span class="col-risk">${riskPill}</span>
  `;

  row.addEventListener("click", () => openDetail(r.record_id));
  return row;
}

function buildRiskPill(recordId) {
  if (!state.hasModel) return `<span class="risk-pill risk-pill--unsure">—</span>`;
  const p = state.predictions[recordId];
  if (!p) return `<span class="risk-pill risk-pill--unsure">—</span>`;
  if (p.predicted_risk === null) {
    return `<span class="risk-pill risk-pill--unsure" title="Confidence ${Math.round(p.confidence * 100)}% — below the threshold, so no prediction is shown">Not sure</span>`;
  }
  if (p.predicted_risk === "Yes") {
    return `<span class="risk-pill risk-pill--high" title="Confidence ${Math.round(p.confidence * 100)}%">High risk</span>`;
  }
  return `<span class="risk-pill risk-pill--low" title="Confidence ${Math.round(p.confidence * 100)}%">Low risk</span>`;
}

function renderBoardCounts() {
  const today = state.records.filter((r) => r.date === state.latestDate);
  const due = today.filter((r) => r.given === "No" || r.given === "").length;
  const given = today.filter((r) => r.given === "Yes").length;
  const flagged = today.filter((r) => r.needs_attention === "Yes").length;
  els.countDue.textContent = due;
  els.countGiven.textContent = given;
  els.countFlag.textContent = flagged;
}

function openDetail(recordId) {
  const record = state.records.find((r) => r.record_id === recordId);
  if (!record) return;
  state.activeId = recordId;

  document.querySelectorAll(".record-row").forEach((el) => {
    el.classList.toggle("is-active", el.dataset.id === recordId);
  });

  els.detailPlaceholder.hidden = true;
  els.detailContent.hidden = false;

  if (!record.resident_id && record.resident_id !== 0) {
    renderUnlinkedDetail(record);
    return;
  }

  const residentRecords = state.records.filter(
    (r) => r.resident_id === record.resident_id
  );
  const withKnownGiven = residentRecords.filter((r) => r.given === "Yes" || r.given === "No");
  const givenCount = residentRecords.filter((r) => r.given === "Yes").length;
  const adherence = withKnownGiven.length
    ? Math.round((givenCount / withKnownGiven.length) * 100)
    : null;
  const flaggedCount = residentRecords.filter((r) => r.needs_attention === "Yes").length;

  els.adherenceValue.textContent = adherence === null ? "No data" : `${adherence}%`;
  document.querySelector(".summary-figure__label").textContent =
    `medication given, ${residentRecords.length} logged dose${residentRecords.length === 1 ? "" : "s"} on record`;

  els.residentMeta.innerHTML = `
    <dt>Resident</dt><dd>${escapeHtml(record.resident_name || "—")}</dd>
    <dt>Resident ID</dt><dd>${escapeHtml(String(record.resident_id))}</dd>
    <dt>Times flagged</dt><dd>${flaggedCount} of ${residentRecords.length} entries</dd>
  `;

  els.recordCard.innerHTML = `
    <h3>Entry ${escapeHtml(record.record_id)} — ${escapeHtml(formatDate(record.date) || "No date recorded")}</h3>
    <div class="field-row"><span>Medication</span><span>${escapeHtml(record.medication || "Not recorded")}</span></div>
    <div class="field-row"><span>Dose time</span><span>${escapeHtml(record.dose_time || "Not recorded")}</span></div>
    <div class="field-row"><span>Given</span><span>${escapeHtml(record.given || "Not recorded")}</span></div>
    <div class="field-row"><span>Recorded flag (staff)</span><span>${escapeHtml(record.needs_attention || "No")}</span></div>
    <div class="field-row"><span>Model prediction</span><span>${describePrediction(record.record_id)}</span></div>
    <div class="observation-block">
      <strong>Observation</strong>
      ${record.observation ? escapeHtml(record.observation) : "No observation logged for this entry."}
    </div>
  `;

  const history = residentRecords
    .filter((r) => r.record_id !== recordId)
    .sort((a, b) => (b.date || "").localeCompare(a.date || ""))
    .slice(0, 6);

  els.historyList.innerHTML = history.length
    ? history
        .map(
          (r) => `
      <li class="${r.needs_attention === "Yes" ? "is-flagged" : ""}">
        <span>${escapeHtml(r.medication || "No medication logged")} — ${escapeHtml(r.given || "not recorded")}</span>
        <span class="hist-date">${escapeHtml(formatDate(r.date) || "—")}</span>
      </li>`
        )
        .join("")
    : `<li>No other entries found for this resident.</li>`;
}

function describePrediction(recordId) {
  if (!state.hasModel) return "Model not run yet — see train_model.py";
  const p = state.predictions[recordId];
  if (!p) return "No prediction available";
  if (p.predicted_risk === null) return `Not confident enough to call (${Math.round(p.confidence * 100)}%)`;
  const setLabel = p.in_test_set ? " · held-out test row" : "";
  return `${p.predicted_risk} (${Math.round(p.confidence * 100)}% confidence)${setLabel}`;
}

function renderUnlinkedDetail(record) {
  els.adherenceValue.textContent = "—";
  document.querySelector(".summary-figure__label").textContent =
    "no resident linked to this entry";
  els.residentMeta.innerHTML = `
    <dt>Resident</dt><dd>Not linked to a resident record</dd>
    <dt>Resident ID</dt><dd>—</dd>
    <dt>Times flagged</dt><dd>—</dd>
  `;
  els.recordCard.innerHTML = `
    <h3>Entry ${escapeHtml(record.record_id)}</h3>
    <div class="field-row"><span>Medication</span><span>Not recorded</span></div>
    <div class="field-row"><span>Dose time</span><span>Not recorded</span></div>
    <div class="field-row"><span>Given</span><span>Not recorded</span></div>
    <div class="observation-block">
      <strong>Observation</strong>
      ${escapeHtml(record.observation || "No observation logged.")}
    </div>
  `;
  els.historyList.innerHTML = `<li>This entry has no resident_id, so it cannot be matched to a resident file. Flagged for the supervisor to review during file audit.</li>`;
}

function closeDetail() {
  state.activeId = null;
  els.detailContent.hidden = true;
  els.detailPlaceholder.hidden = false;
  document.querySelectorAll(".record-row").forEach((el) => el.classList.remove("is-active"));
}

function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso + "T00:00:00");
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[c]));
}
