document.addEventListener("DOMContentLoaded", () => {

  const sourceDb = document.getElementById("sourceDb");
  const targetDb = document.getElementById("targetDb");
  const promptInput = document.getElementById("prompt");
  const submitBtn = document.getElementById("submitBtn");

  const sourceTableList = document.getElementById("sourceTableList");
  const targetTableList = document.getElementById("targetTableList");

  const resultPlaceholder = document.getElementById("resultPlaceholder");
  const resultContent = document.getElementById("resultContent");

  /* ===================== HELPERS ===================== */

  async function fetchJSON(url, options = {}) {
    const res = await fetch(url, options);
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

  /* ===================== LOAD DATABASES ===================== */

  async function loadDatabases() {
    const dbs = await fetchJSON("/api/databases");
    dbs.forEach(db => {
      sourceDb.appendChild(new Option(db, db));
      targetDb.appendChild(new Option(db, db));
    });
  }

  /* ===================== LOAD TABLES ===================== */

  async function loadTables(database, target) {
    const tables = await fetchJSON("/api/tables", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ database })
    });
  
    renderTableList(tables, target);
  }
  
  function renderTableList(tables, container) {
    container.innerHTML = "";
  
    tables.forEach(t => {
      const chip = document.createElement("div");
      chip.className = "table-chip";
      chip.textContent = t;
      chip.title = "Click to copy";
  
      chip.onclick = () => {
        navigator.clipboard.writeText(t);
        chip.classList.add("copied");
        setTimeout(() => chip.classList.remove("copied"), 700);
      };
  
      container.appendChild(chip);
    });
  }  

  sourceDb.addEventListener("change", () => {
    if (sourceDb.value) loadTables(sourceDb.value, sourceTableList);
  });
  targetDb.addEventListener("change", () => {
    if (targetDb.value) loadTables(targetDb.value, targetTableList);
  });

  /* ===================== SUBMIT ===================== */

  submitBtn.addEventListener("click", async () => {
    if (!sourceDb.value || !targetDb.value) {
      alert("Please select both databases");
      return;
    }

    if (!promptInput.value.trim()) {
      alert("Please enter a validation condition");
      return;
    }

    resultPlaceholder.style.display = "block";
    resultContent.style.display = "none";
    resultContent.innerHTML = "";

    const payload = {
      source_db: sourceDb.value,
      target_db: targetDb.value,
      prompt: promptInput.value
    };

    try {
      const response = await fetchJSON("/api/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      displayResults(response.results || []);
    } catch (err) {
      resultContent.innerHTML = `<p style="color:red;">${err.message}</p>`;
      resultContent.style.display = "block";
      resultPlaceholder.style.display = "none";
    }
  });

  /* ===================== DISPLAY RESULTS ===================== */

  function displayResults(results) {
    if (!results.length) {
      resultContent.innerHTML = "<p>No results returned.</p>";
      resultContent.style.display = "block";
      resultPlaceholder.style.display = "none";
      return;
    }

    resultContent.innerHTML = "";

    results.forEach((r, i) => {
      const card = document.createElement("div");
      card.className = "result-card";

      card.innerHTML = `
        <h3>Result ${i + 1}</h3>
        <p><strong>Source Table:</strong> ${r.source_table}</p>
        <p><strong>Target Table:</strong> ${r.target_table}</p>

        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-label">Source Count</div>
            <div class="metric-value">${r.source_count}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Target Count</div>
            <div class="metric-value">${r.target_count}</div>
          </div>
        </div>

        <div class="summary-section">
          <div class="summary-title">🧠 AI Analysis</div>
          <div>${r.summary}</div>
        </div>

        <details>
          <summary><strong>Show SQL</strong></summary>
          <pre class="sql-box">${escapeHtml(r.source_query)}</pre>
          <pre class="sql-box">${escapeHtml(r.target_query)}</pre>
        </details>
      `;

      resultContent.appendChild(card);
    });

    resultPlaceholder.style.display = "none";
    resultContent.style.display = "block";
  }

  /* ===================== INIT ===================== */

  loadDatabases();

});