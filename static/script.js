document.addEventListener("DOMContentLoaded", () => {
  const sourceDbSelect = document.getElementById("sourceDb");
  const targetDbSelect = document.getElementById("targetDb");
  const sourceTableSelect = document.getElementById("sourceTableSelect");
  const targetTableSelect = document.getElementById("targetTableSelect");
  const promptInput = document.getElementById("prompt");
  const submitBtn = document.getElementById("submitBtn");
  const loadingSpinner = document.getElementById("loadingSpinner");
  const submitText = document.getElementById("submitText");
  const resultSection = document.getElementById("resultSection");
  const resultPlaceholder = document.getElementById("resultPlaceholder");
  const resultContent = document.getElementById("resultContent");

  // Utility function for API calls with timeout
  async function fetchWithTimeout(url, options = {}, timeout = 60000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      console.log("error at the begning result:", error);
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        throw new Error('Request timed out. Please try again.');
      }
      throw error;
    }
  }

  // Show loading state
  function showLoading(message = "Loading...") {
    submitBtn.disabled = true;
    loadingSpinner.style.display = "inline-block";
    submitText.textContent = message;
  }

  // Hide loading state
  function hideLoading() {
    submitBtn.disabled = false;
    loadingSpinner.style.display = "none";
    submitText.textContent = "🚀 Validate Row Counts";
  }

  // Show error message
  function showError(message) {
    console.error("Error:", message);
    alert(`Error: ${message}`);
  }

  // Load available databases
  async function loadDatabases() {
    try {
      showLoading("Loading databases...");

      const response = await fetchWithTimeout("/api/databases", {}, 30000);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP ${response.status}`);
      }

      const databases = await response.json();

      // Clear existing options
      sourceDbSelect.innerHTML = "<option value=''>Select Source DB</option>";
      targetDbSelect.innerHTML = "<option value=''>Select Target DB</option>";

      // Add database options
      databases.forEach(db => {
        sourceDbSelect.appendChild(new Option(db, db));
        targetDbSelect.appendChild(new Option(db, db));
      });

      console.log(`Loaded ${databases.length} databases:`, databases);

    } catch (error) {
      showError(`Failed to load databases: ${error.message}`);
    } finally {
      hideLoading();
    }
  }

  // Load tables for a specific database
  async function loadTables(database, targetSelect, placeholderText) {
    if (!database) {
      targetSelect.innerHTML = `<option value=''>${placeholderText}</option>`;
      return;
    }

    try {
      showLoading(`Loading tables for ${database}...`);

      const response = await fetchWithTimeout("/api/tables", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ database: database })
      }, 30000);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP ${response.status}`);
      }

      const tables = await response.json();

      // Clear and populate options
      targetSelect.innerHTML = `<option value=''>${placeholderText}</option>`;
      tables.forEach(table => {
        targetSelect.appendChild(new Option(table, table));
      });

      console.log(`Loaded ${tables.length} tables for ${database}:`, tables);

    } catch (error) {
      showError(`Failed to load tables for ${database}: ${error.message}`);
      targetSelect.innerHTML = `<option value=''>${placeholderText}</option>`;
    } finally {
      hideLoading();
    }
  }

  // Event listeners for database selection
  sourceDbSelect.addEventListener("change", () => {
    const db = sourceDbSelect.value;
    loadTables(db, sourceTableSelect, "Choose a source table");
  });

  targetDbSelect.addEventListener("change", () => {
    const db = targetDbSelect.value;
    loadTables(db, targetTableSelect, "Choose a target table");
  });

  // Validate form inputs
  function validateForm() {
    const sourceDb = sourceDbSelect.value.trim();
    const targetDb = targetDbSelect.value.trim();
    const sourceTable = sourceTableSelect.value.trim();
    const targetTable = targetTableSelect.value.trim();
    const prompt = promptInput.value.trim();

    if (!sourceDb) {
      showError("Please select a source database.");
      return false;
    }
    if (!targetDb) {
      showError("Please select a target database.");
      return false;
    }
    if (!sourceTable) {
      showError("Please select a source table.");
      return false;
    }
    if (!targetTable) {
      showError("Please select a target table.");
      return false;
    }
    if (!prompt) {
      showError("Please enter a validation condition.");
      return false;
    }

    return true;
  }

  // Submit validation request
  async function submitValidation() {
    if (!validateForm()) {
      return;
    }

    const formData = {
      source_db: sourceDbSelect.value,
      target_db: targetDbSelect.value,
      source_table: sourceTableSelect.value,
      target_table: targetTableSelect.value,
      prompt: promptInput.value
    };

    try {
      // Show loading state
      showLoading("Analyzing data...");
      resultSection.className = "result-section";
      resultPlaceholder.style.display = "flex";
      resultContent.style.display = "none";

      console.log("Submitting validation request:", formData);

      const response = await fetchWithTimeout("/api/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      }, 120000); // 2 minute timeout for validation

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("Validation result:", result);

      displayResults(result);

    } catch (error) {
      console.error("Validation error:", error);
      displayError(error.message);
    } finally {
      hideLoading();
    }
  }

  // Display validation results
  function displayResults(result) {
    resultSection.className = "result-section has-results";
    resultPlaceholder.style.display = "none";
    resultContent.style.display = "block";

    // Update counts
    document.getElementById("sourceCount").textContent = result.source_count?.toLocaleString() || 'N/A';
    document.getElementById("targetCount").textContent = result.target_count?.toLocaleString() || 'N/A';
    document.getElementById("lstmPrediction").textContent = result.lstm_prediction?.toLocaleString() || 'N/A';
    document.getElementById("arimaPrediction").textContent = result.arima_prediction?.toLocaleString() || 'N/A';
    document.getElementById("summaryText").textContent = result.summary || 'No summary available';

    // Update status
    const statusEl = document.getElementById("resultStatus");
    if (result.is_anomaly) {
      statusEl.textContent = "⚠️ Anomaly Detected";
      statusEl.className = "result-status status-warning";
    } else if (result.source_count === result.target_count) {
      statusEl.textContent = "✅ Perfect Match";
      statusEl.className = "result-status status-success";
    } else {
      statusEl.textContent = "⚠️ Count Mismatch";
      statusEl.className = "result-status status-warning";
    }
  }

  // Display error message
  function displayError(message) {
    resultSection.className = "result-section has-error";
    resultPlaceholder.style.display = "none";
    resultContent.innerHTML = `
      <div class="error-content">
        <h3>❌ Validation Error</h3>
        <p><strong>Error:</strong> ${message}</p>
        <p><strong>Troubleshooting:</strong></p>
        <ul>
          <li>Check if the selected databases and tables exist</li>
          <li>Verify your database connection settings</li>
          <li>Ensure the validation condition is valid SQL syntax</li>
          <li>Try refreshing the page and selecting options again</li>
        </ul>
      </div>
    `;
    resultContent.style.display = "block";
  }

  // Submit button click handler
  submitBtn.addEventListener("click", submitValidation);

  // Allow Enter key in prompt input
  promptInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      submitValidation();
    }
  });

  // Initialize the application
  console.log("Initializing application...");
  loadDatabases();
});