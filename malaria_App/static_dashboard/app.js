const $ = (selector) => document.querySelector(selector);

const state = {
  activeRequestId: null,
  activeCorrelationId: null,
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

function formatMs(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Number(value).toFixed(1)} ms`;
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

function showView(viewId) {
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active-view", view.id === viewId);
  });
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === viewId);
  });

  if (viewId === "reviewView") {
    loadQueue();
  } else if (viewId === "monitoringView") {
    loadMonitoring();
  } else if (viewId === "traceView") {
    loadEvents();
  }
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
    const metrics = health.service_metrics || {};
    $("#serviceLatency").textContent = metrics.recent_request_count
      ? `p95 ${formatMs(metrics.p95_request_latency_ms)} | errors ${formatPercent(metrics.api_error_rate)}`
      : "No request metrics yet";
    $("#thresholdInput").value = health.default_parasitized_threshold;
    $("#reviewMarginInput").value = health.default_review_margin;
    $("#thresholdValue").textContent = formatNumber(health.default_parasitized_threshold);
  } catch (error) {
    setBadge($("#serviceStatus"), "Service unavailable", "danger");
    $("#serviceLatency").textContent = "-";
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
    $("#failureEvents").textContent = summary.failure_event_count ?? 0;
    $("#lastFailureStage").textContent = summary.last_failure_stage || "-";
    $("#recentRequestCount").textContent = summary.recent_request_count ?? 0;
    $("#avgLatency").textContent = formatMs(summary.avg_request_latency_ms);
    $("#p95Latency").textContent = formatMs(summary.p95_request_latency_ms);
    $("#maxLatency").textContent = formatMs(summary.max_request_latency_ms);
    $("#apiErrorRate").textContent = formatPercent(summary.api_error_rate);
  } catch (error) {
    $("#totalPredictions").textContent = "-";
    $("#reviewRate").textContent = "-";
    $("#warningRate").textContent = "-";
    $("#qualityPassRate").textContent = "-";
    $("#failureEvents").textContent = "-";
    $("#lastFailureStage").textContent = "-";
    $("#recentRequestCount").textContent = "-";
    $("#avgLatency").textContent = "-";
    $("#p95Latency").textContent = "-";
    $("#maxLatency").textContent = "-";
    $("#apiErrorRate").textContent = "-";
  }
}

function queueCard(row) {
  const score = formatNumber(row.parasitized_score || row.model_parasitized_score);
  const prediction = row.predicted_class || row.model_predicted_class || "Unknown";
  const reason = row.review_reason || "No review reason stored.";
  const requestId = row.request_id || "";
  return `
    <article class="queue-card">
      <strong>${escapeHtml(prediction)} | score ${score}</strong>
      <span class="mono">${escapeHtml(requestId || "missing-request-id")}</span>
      <span>${escapeHtml(reason)}</span>
      <button type="button" class="secondary-button queue-select" data-request-id="${escapeHtml(requestId)}">Select For Review</button>
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
  state.activeCorrelationId = correlationId;
  $("#feedbackRequestId").textContent = prediction.request_id;
  $("#correlationId").textContent = correlationId;
  $("#traceRequestInput").value = prediction.request_id;
  $("#traceCorrelationInput").value = correlationId;

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

function eventCard(row) {
  const severity = row.severity || "info";
  const stage = row.stage || "unknown";
  const status = row.status || "unknown";
  const message = row.message || "";
  const requestId = row.request_id || "";
  const correlationId = row.correlation_id || "";
  return `
    <article class="event-card">
      <strong>${escapeHtml(row.event_type || "event")} | ${escapeHtml(severity)} | ${escapeHtml(status)}</strong>
      <span>${escapeHtml(stage)} | ${escapeHtml(row.timestamp_utc || "")}</span>
      <span>${escapeHtml(message)}</span>
      <span class="mono">${escapeHtml(requestId || correlationId || "no trace id")}</span>
    </article>
  `;
}

async function loadEvents() {
  const list = $("#eventList");
  try {
    const payload = await fetchJson("/events/recent?limit=25", {
      headers: apiHeaders(),
    });
    const rows = payload.items || [];
    list.innerHTML = rows.length
      ? rows.map(eventCard).join("")
      : '<p class="muted">No events recorded yet.</p>';
  } catch (error) {
    list.innerHTML = `<p class="muted">${escapeHtml(error.message)}</p>`;
  }
}

async function loadTrace(url) {
  try {
    const payload = await fetchJson(url, {
      headers: apiHeaders(),
    });
    renderTrace(payload);
    $("#traceOutput").textContent = JSON.stringify(payload, null, 2);
  } catch (error) {
    $("#traceSummary").textContent = error.message;
    $("#traceTimeline").innerHTML = '<p class="muted">No trace loaded.</p>';
    $("#traceOutput").textContent = error.message;
  }
}

function renderTrace(payload) {
  const prediction = payload.prediction || {};
  const requestId = payload.request_id || prediction.request_id || "No request ID";
  const correlationId = payload.correlation_id || prediction.correlation_id || "No correlation ID";
  const timeline = payload.timeline || [];
  const feedbackCount = (payload.review_feedback || []).length;
  const eventCount = (payload.events || []).length;

  $("#traceSummary").innerHTML = `
    <div class="trace-summary-grid">
      <span><strong>Request</strong><br><span class="mono">${escapeHtml(requestId)}</span></span>
      <span><strong>Correlation</strong><br><span class="mono">${escapeHtml(correlationId)}</span></span>
      <span><strong>Prediction</strong><br>${escapeHtml(prediction.predicted_class || "Unavailable")}</span>
      <span><strong>Review</strong><br>${escapeHtml(prediction.review_required ? "Required" : "Not required")}</span>
      <span><strong>Feedback</strong><br>${feedbackCount}</span>
      <span><strong>Events</strong><br>${eventCount}</span>
    </div>
  `;

  $("#traceTimeline").innerHTML = timeline.length
    ? timeline.map(timelineCard).join("")
    : '<p class="muted">No timeline records found for this trace.</p>';
}

function timelineCard(item) {
  const severity = item.severity || "info";
  const status = item.status || "unknown";
  const details = item.details ? JSON.stringify(item.details, null, 2) : "{}";
  return `
    <article class="timeline-card ${escapeHtml(severity)}">
      <div class="timeline-marker"></div>
      <div class="timeline-content">
        <div class="timeline-head">
          <strong>${escapeHtml(item.title || item.stage || "Trace step")}</strong>
          <span>${escapeHtml(item.timestamp_utc || "")}</span>
        </div>
        <p>${escapeHtml(item.message || "")}</p>
        <div class="timeline-meta">
          <span>${escapeHtml(item.stage || "unknown stage")}</span>
          <span>${escapeHtml(status)}</span>
        </div>
        <details>
          <summary>Step details</summary>
          <pre>${escapeHtml(details)}</pre>
        </details>
      </div>
    </article>
  `;
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
    await Promise.all([loadMonitoring(), loadQueue(), loadEvents()]);
    await loadTrace(`/trace/${encodeURIComponent(payload.prediction.request_id)}`);
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
        reviewer_id: $("#reviewerId").value || "anonymous",
        reviewer_decision: $("#reviewerDecision").value,
        final_label: $("#finalLabel").value,
        follow_up_action: $("#followUpAction").value,
        reviewer_notes: $("#reviewerNotes").value,
      }),
    });
    $("#reviewerNotes").value = "";
    setMessage("Reviewer feedback saved.");
    await Promise.all([loadMonitoring(), loadQueue(), loadEvents()]);
  } catch (error) {
    setMessage(error.message, true);
  }
}

function bindEvents() {
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.addEventListener("click", () => showView(button.dataset.view));
  });

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
  });
  $("#refreshQueueButton").addEventListener("click", loadQueue);
  $("#refreshEventsButton").addEventListener("click", loadEvents);
  $("#traceRequestButton").addEventListener("click", () => {
    const requestId = $("#traceRequestInput").value.trim();
    if (requestId) {
      loadTrace(`/trace/${encodeURIComponent(requestId)}`);
    }
  });
  $("#traceCorrelationButton").addEventListener("click", () => {
    const correlationId = $("#traceCorrelationInput").value.trim();
    if (correlationId) {
      loadTrace(`/trace/correlation/${encodeURIComponent(correlationId)}`);
    }
  });
  $("#queueList").addEventListener("click", (event) => {
    const button = event.target.closest(".queue-select");
    if (!button) {
      return;
    }
    state.activeRequestId = button.dataset.requestId;
    $("#feedbackRequestId").textContent = state.activeRequestId;
    $("#traceRequestInput").value = state.activeRequestId;
    loadTrace(`/trace/${encodeURIComponent(state.activeRequestId)}`);
  });
}

bindEvents();
loadHealth();
loadMonitoring();
loadQueue();
loadEvents();
