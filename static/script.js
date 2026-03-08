document.addEventListener("DOMContentLoaded", () => {

  const srcDbSel = document.getElementById("sourceDb");
  const tgtDbSel = document.getElementById("targetDb");
  const srcTblSel = document.getElementById("sourceTable");
  const tgtTblSel = document.getElementById("targetTable");
  const promptBox = document.getElementById("prompt");
  const goBtn = document.getElementById("submitBtn");

  const resPlaceholder = document.getElementById("resultPlaceholder");
  const resContent = document.getElementById("resultContent");

  async function getJSON(url, opts = {}) {
    const res = await fetch(url, opts);
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err);
    }
    return res.json();
  }

  function escapeHtml(text = "") {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function resetSel(sel, label) {
    sel.innerHTML = "";
    sel.appendChild(new Option(label, ""));
    sel.value = "";
    sel.disabled = true;
  }

  function fillSel(sel, items, label) {
    resetSel(sel, label);
    items.forEach(item => sel.appendChild(new Option(item, item)));
    sel.disabled = false;
  }

  async function loadDBs() {
    const dbs = await getJSON("/api/databases");
    fillSel(srcDbSel, dbs, "Select Source DB");
    fillSel(tgtDbSel, dbs, "Select Target DB");
  }

  async function load_tbls(dbName, sel, label) {
    resetSel(sel, "Loading tables...");
    try {
      const tables = await getJSON("/api/tables", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ database: dbName })
      });
      fillSel(sel, tables, label);
    } catch (err) {
      resetSel(sel, label);
      alert("Failed to load tables: " + err.message);
    }
  }

  srcDbSel.addEventListener("change", () => {
    resetSel(srcTblSel, "Select Source Table");
    if (srcDbSel.value) load_tbls(srcDbSel.value, srcTblSel, "Select Source Table");
  });

  tgtDbSel.addEventListener("change", () => {
    resetSel(tgtTblSel, "Select Target Table");
    if (tgtDbSel.value) load_tbls(tgtDbSel.value, tgtTblSel, "Select Target Table");
  });

  goBtn.addEventListener("click", async () => {
    if (!srcDbSel.value || !tgtDbSel.value) {
      alert("Please select both databases");
      return;
    }

    if (!srcTblSel.value || !tgtTblSel.value) {
      alert("Please select both tables");
      return;
    }

    if (!promptBox.value.trim()) {
      alert("Please enter a validation question");
      return;
    }

    resPlaceholder.style.display = "block";
    resContent.style.display = "none";
    resContent.innerHTML = "";

    const payload = {
      source_db: srcDbSel.value,
      target_db: tgtDbSel.value,
      source_table: srcTblSel.value,
      target_table: tgtTblSel.value,
      prompt: promptBox.value
    };

    try {
      const resp = await getJSON("/api/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      showResult(resp.result);
    } catch (err) {
      resContent.innerHTML = `<p style="color:red;">${escapeHtml(err.message)}</p>`;
      resContent.style.display = "block";
      resPlaceholder.style.display = "none";
    }
  });

  function showResult(result) {
    if (!result) {
      resContent.innerHTML = "<p>No results returned.</p>";
      resContent.style.display = "block";
      resPlaceholder.style.display = "none";
      return;
    }

    const pct = typeof result.pct_diff === "number" ? result.pct_diff.toFixed(2) : "0.00";
    const matchLabel = result.is_match ? "Match" : "Mismatch";

    resContent.innerHTML = "";

    const card = document.createElement("div");
    card.className = "result-card";

    card.innerHTML = `
      <h3>Validation Result</h3>
      <p><strong>Source Table:</strong> ${escapeHtml(result.source_db)}.${escapeHtml(result.source_table)}</p>
      <p><strong>Target Table:</strong> ${escapeHtml(result.target_db)}.${escapeHtml(result.target_table)}</p>

      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-label">Source Count</div>
          <div class="metric-value">${result.source_count}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Target Count</div>
          <div class="metric-value">${result.target_count}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Difference</div>
          <div class="metric-value">${result.diff}</div>
        </div>
      </div>

      <div class="summary-section">
        <div class="summary-title">Validation Report</div>
        <div>${escapeHtml(result.summary || "")}</div>
        <div class="muted">${matchLabel} (${pct}% difference)</div>
      </div>

      <details>
        <summary><strong>Show SQL</strong></summary>
        <pre class="sql-box">${escapeHtml(result.source_query || "")}</pre>
        <pre class="sql-box">${escapeHtml(result.target_query || "")}</pre>
      </details>
    `;

    resContent.appendChild(card);
    resPlaceholder.style.display = "none";
    resContent.style.display = "block";
  }

  loadDBs();

});
