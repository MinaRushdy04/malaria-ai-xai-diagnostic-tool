const $ = (selector) => document.querySelector(selector);

const state = {
  activeRequestId: null,
};

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setMessage(text, isError = false) {
  const box = $("#messageBox");
  if (!text) {
    box.hidden = true;
    box.textContent = "";
    return;
  }
  box.hidden = false;
  box.textContent = text;
  box.style.background = isError ? "#fee2e2" : "#ecfeff";
  box.style.color = isError ? "#7f1d1d" : "#164e63";
}

function setBadge(element, text, className) {
  element.className = `status-pill ${className || "neutral"}`;
  element.textContent = text;
}

function newCorrelationId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return `web-${window.crypto.randomUUID()}`;
  }
  return `web-${Date.now()}`;
}

function apiHeaders(extra = {}) {
  const headers = { ...extra };
  const apiKeyInput = $("#apiKeyInput");
  const apiKey = apiKeyInput ? apiKeyInput.value.trim() : "";
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  return headers;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail || payload.message || response.statusText;
    const message = typeof detail === "string" ? detail : JSON.stringify(detail);
    throw new Error(message);
  }
  return payload;
}

async function loadHealth() {
  try {
    const health = await fetchJson("/health");
    setBadge(
      $("#serviceStatus"),
      health.status === "ok" ? "Service healthy" : "Service degraded",
      health.status === "ok" ? "success" : "warning",
    );
    $("#modelVersion").textContent = `${health.model_version} | threshold ${formatNumber(health.default_parasitized_threshold)}`;
    if (health.auth_required) {
      $("#modelVersion").textContent += " | API key required";
    }
    $("#thresholdInput").value = health.default_parasitized_threshold;
    $("#reviewMarginInput").value = health.default_review_margin;
    $("#thresholdValue").textContent = formatNumber(health.default_parasitized_threshold);
  } catch (error) {
    setBadge($("#serviceStatus"), "Service unavailable", "danger");
    $("#modelVersion").textContent = error.message;
  }
}

async function loadMonitoring() {
  try {
    const summary = await fetchJson("/monitoring/summary?limit=200", {
      headers: apiHeaders(),
    });
    $("#totalPredictions").textContent = summary.total_predictions;
    $("#reviewRate").textContent = formatPercent(summary.review_rate);
    $("#warningRate").textContent = formatPercent(summary.validation_warning_rate);
    $("#qualityPassRate").textContent = formatPercent(summary.quality_pass_rate);
  } catch (error) {
    $("#totalPredictions").textContent = "-";
    $("#reviewRate").textContent = "-";
    $("#warningRate").textContent = "-";
    $("#qualityPassRate").textContent = "-";
  }
}

function queueCard(row) {
  const score = formatNumber(row.parasitized_score || row.model_parasitized_score);
  const prediction = row.predicted_class || row.model_predicted_class || "Unknown";
  const reason = row.review_reason || "No review reason stored.";
  return `
    <article class="queue-card">
      <strong>${escapeHtml(prediction)} | score ${score}</strong>
      <span class="mono">${escapeHtml(row.request_id || "missing-request-id")}</span>
      <span>${escapeHtml(reason)}</span>
    </article>
  `;
}

async function loadQueue() {
  const list = $("#queueList");
  try {
    const payload = await fetchJson("/review/queue?limit=12", {
      headers: apiHeaders(),
    });
    const rows = payload.items || [];
    $("#queueCount").textContent = `${rows.length} cases`;
    list.innerHTML = rows.length
      ? rows.map(queueCard).join("")
      : '<p class="muted">No high-priority unreviewed cases.</p>';
  } catch (error) {
    $("#queueCount").textContent = "Unavailable";
    list.innerHTML = `<p class="muted">${error.message}</p>`;
  }
}

function renderWarnings(warnings) {
  const list = $("#warningList");
  if (!warnings || warnings.length === 0) {
    list.innerHTML = "<li>No validation or quality warnings.</li>";
    return;
  }
  list.innerHTML = warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("");
}

function renderPrediction(payload, correlationId) {
  const prediction = payload.prediction;
  const validation = payload.validation;
  const quality = validation.quality;

  state.activeRequestId = prediction.request_id;
  $("#feedbackRequestId").textContent = prediction.request_id;
  $("#correlationId").textContent = correlationId;

  $("#predictedClass").textContent = prediction.predicted_class;
  $("#parasitizedScore").textContent = formatNumber(prediction.parasitized_score);
  $("#decisionMargin").textContent = formatNumber(prediction.decision_margin);
  $("#thresholdValue").textContent = formatNumber(prediction.threshold);
  $("#reviewReason").textContent = prediction.review_reason;
  $("#recommendation").textContent = prediction.recommendation;

  setBadge(
    $("#reviewBadge"),
    prediction.review_required ? "Expert review" : "Outside review band",
    prediction.review_required ? "warning" : "success",
  );

  $("#brightnessValue").textContent = formatNumber(quality.brightness_mean, 1);
  $("#contrastValue").textContent = formatNumber(quality.contrast_std, 1);
  $("#focusValue").textContent = formatNumber(quality.focus_score, 1);
  $("#imageSizeValue").textContent = `${validation.width} x ${validation.height}`;

  setBadge(
    $("#qualityBadge"),
    quality.passed ? "Quality pass" : "Quality review",
    quality.passed ? "success" : "warning",
  );
  renderWarnings(validation.warnings);

  if (payload.xai && payload.xai.gradcam_overlay) {
    $("#gradcamPreview").src = payload.xai.gradcam_overlay;
  } else {
    $("#gradcamPreview").removeAttribute("src");
  }
}

async function handlePredictionSubmit(event) {
  event.preventDefault();
  setMessage("");

  const file = $("#fileInput").files[0];
  if (!file) {
    setMessage("Choose a JPG or PNG microscopy image first.", true);
    return;
  }

  const correlationId = newCorrelationId();
  const body = new FormData();
  body.append("file", file);
  body.append("threshold", $("#thresholdInput").value);
  body.append("review_margin", $("#reviewMarginInput").value);
  body.append("include_xai", $("#includeXaiInput").checked);
  body.append("route_warnings_to_review", $("#routeWarningsInput").checked);
  body.append("enable_logging", $("#enableLoggingInput").checked);

  $("#submitButton").disabled = true;
  $("#submitButton").textContent = "Running...";

  try {
    const payload = await fetchJson("/predict", {
      method: "POST",
      headers: apiHeaders({ "X-Correlation-ID": correlationId }),
      body,
    });
    renderPrediction(payload, correlationId);
    setMessage("Prediction record created.");
    await Promise.all([loadMonitoring(), loadQueue()]);
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    $("#submitButton").disabled = false;
    $("#submitButton").textContent = "Run Inference";
  }
}

async function handleFeedbackSubmit(event) {
  event.preventDefault();
  setMessage("");
  if (!state.activeRequestId) {
    setMessage("Run a prediction before saving reviewer feedback.", true);
    return;
  }

  try {
    await fetchJson("/review/feedback", {
      method: "POST",
      headers: apiHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        request_id: state.activeRequestId,
        reviewer_decision: $("#reviewerDecision").value,
        reviewer_notes: $("#reviewerNotes").value,
      }),
    });
    $("#reviewerNotes").value = "";
    setMessage("Reviewer feedback saved.");
    await Promise.all([loadMonitoring(), loadQueue()]);
  } catch (error) {
    setMessage(error.message, true);
  }
}

function bindEvents() {
  $("#fileInput").addEventListener("change", () => {
    const file = $("#fileInput").files[0];
    $("#fileName").textContent = file ? file.name : "Choose microscopy image";
    if (file) {
      $("#inputPreview").src = URL.createObjectURL(file);
    } else {
      $("#inputPreview").removeAttribute("src");
    }
  });

  $("#thresholdInput").addEventListener("input", () => {
    $("#thresholdValue").textContent = formatNumber($("#thresholdInput").value);
  });

  $("#predictionForm").addEventListener("submit", handlePredictionSubmit);
  $("#feedbackForm").addEventListener("submit", handleFeedbackSubmit);
  $("#refreshButton").addEventListener("click", () => {
    loadMonitoring();
    loadQueue();
  });
}

bindEvents();
loadHealth();
loadMonitoring();
loadQueue();
