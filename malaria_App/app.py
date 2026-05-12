# pyrefly: ignore [missing-import]
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

try:
    from .diagnostic_core import (
        DEFAULT_REVIEW_MARGIN,
        PARASITIZED_THRESHOLD,
        ImageValidationError,
        data_uri_to_image,
        diagnose_image,
        export_review_feedback_csv,
        load_keras_model,
        read_active_learning_queue,
        read_recent_logs,
        read_review_feedback,
        summarize_review_feedback,
        summarize_logs,
        validate_image_bytes,
        write_review_feedback,
        write_prediction_log,
    )
    from .middleware import CORRELATION_ID_HEADER
except ImportError:
    from diagnostic_core import (
        DEFAULT_REVIEW_MARGIN,
        PARASITIZED_THRESHOLD,
        ImageValidationError,
        data_uri_to_image,
        diagnose_image,
        export_review_feedback_csv,
        load_keras_model,
        read_active_learning_queue,
        read_recent_logs,
        read_review_feedback,
        summarize_review_feedback,
        summarize_logs,
        validate_image_bytes,
        write_review_feedback,
        write_prediction_log,
    )
    from middleware import CORRELATION_ID_HEADER


st.set_page_config(
    page_title="Malaria AI Diagnostic Dashboard",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: Inter, Segoe UI, system-ui, sans-serif;
}
.stApp {
    background: #0f1419;
    color: #e6edf3;
}
#MainMenu, footer, header {
    visibility: hidden;
}
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}
.dashboard-header {
    border-bottom: 1px solid #26323d;
    padding-bottom: 1rem;
    margin-bottom: 1rem;
}
.dashboard-title {
    font-size: 2rem;
    font-weight: 760;
    color: #f8fafc;
    letter-spacing: 0;
    margin: 0;
}
.dashboard-subtitle {
    color: #9aa8b6;
    font-size: 0.98rem;
    line-height: 1.55;
    margin-top: 0.35rem;
}
.section-label {
    color: #78d5c3;
    text-transform: uppercase;
    letter-spacing: 0.08rem;
    font-size: 0.76rem;
    font-weight: 760;
    margin-bottom: 0.45rem;
}
.panel {
    background: #151d24;
    border: 1px solid #273441;
    border-radius: 8px;
    padding: 1rem 1.05rem;
    margin-bottom: 0.85rem;
}
.compact-text {
    color: #a9b6c3;
    font-size: 0.9rem;
    line-height: 1.55;
}
.result-positive {
    background: rgba(185, 28, 28, 0.13);
    border: 1px solid rgba(248, 113, 113, 0.42);
    border-radius: 8px;
    padding: 1rem;
}
.result-negative {
    background: rgba(21, 128, 61, 0.13);
    border: 1px solid rgba(74, 222, 128, 0.36);
    border-radius: 8px;
    padding: 1rem;
}
.review-callout {
    background: rgba(217, 119, 6, 0.14);
    border: 1px solid rgba(251, 191, 36, 0.42);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin-top: 0.75rem;
}
.clear-callout {
    background: rgba(14, 165, 233, 0.10);
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin-top: 0.75rem;
}
.small-muted {
    color: #8b9aaa;
    font-size: 0.8rem;
}
[data-testid="stFileUploaderDropzone"] {
    background: #111820 !important;
    border: 1px dashed #415161 !important;
    border-radius: 8px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model_cached():
    return load_keras_model()


def post_to_api(
    api_url: str,
    uploaded_bytes: bytes,
    filename: str,
    content_type: str,
    threshold: float,
    review_margin: float,
    include_xai: bool,
    route_warnings_to_review: bool,
    enable_logging: bool,
    correlation_id: str | None,
) -> dict[str, Any]:
    endpoint = api_url.rstrip("/") + "/predict"
    headers = {CORRELATION_ID_HEADER: correlation_id} if correlation_id else {}
    response = requests.post(
        endpoint,
        headers=headers,
        files={"file": (filename, uploaded_bytes, content_type or "application/octet-stream")},
        data={
            "threshold": str(threshold),
            "review_margin": str(review_margin),
            "include_xai": str(include_xai).lower(),
            "route_warnings_to_review": str(route_warnings_to_review).lower(),
            "enable_logging": str(enable_logging).lower(),
        },
        timeout=90,
    )
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API returned HTTP {response.status_code}: {detail}")
    payload = response.json()
    payload["response_headers"] = {
        CORRELATION_ID_HEADER: response.headers.get(CORRELATION_ID_HEADER),
        "X-Process-Time-Ms": response.headers.get("X-Process-Time-Ms"),
    }
    return payload


def render_header() -> None:
    st.markdown(
        """
        <div class="dashboard-header">
            <div class="dashboard-title">Malaria Cell-Smear AI Dashboard</div>
            <div class="dashboard-subtitle">
                Threshold-aware inference, input quality checks, expert-review routing,
                Grad-CAM explainability, and audit-style monitoring in one workflow.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_strip(settings: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("Inference mode", settings["inference_mode"])
    cols[1].metric("Threshold", f"{settings['threshold']:.3f}")
    cols[2].metric("Review band", f"+/- {settings['review_margin']:.3f}")
    cols[3].metric("Logging", "On" if settings["enable_logging"] else "Off")


def render_prediction(prediction: dict[str, Any]) -> None:
    predicted_class = prediction["predicted_class"]
    positive = predicted_class == "Parasitized"
    css_class = "result-positive" if positive else "result-negative"
    title = "Parasitized screening flag" if positive else "Uninfected screening result"

    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="section-label">Decision support output</div>
            <h3 style="margin:0 0 0.45rem 0;color:#f8fafc;">{title}</h3>
            <div class="compact-text">{prediction["recommendation"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Parasitized score", f"{prediction['parasitized_score']:.3f}")
    metric_cols[1].metric("Class score", f"{prediction['prediction_score']:.3f}")
    metric_cols[2].metric("Distance to threshold", f"{prediction['decision_margin']:.3f}")
    metric_cols[3].metric("Review margin", f"{prediction['review_margin']:.3f}")

    st.progress(
        min(max(float(prediction["parasitized_score"]), 0.0), 1.0),
        text="Parasitized score",
    )

    if prediction["review_required"]:
        st.markdown(
            f"""
            <div class="review-callout">
                <strong>Expert review required</strong><br>
                <span class="compact-text">{prediction["review_reason"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="clear-callout">
                <strong>Outside configured review band</strong><br>
                <span class="compact-text">{prediction["review_reason"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_quality_gate(validation: dict[str, Any]) -> None:
    quality = validation.get("quality") or {}
    warnings = validation.get("warnings") or []

    st.markdown('<div class="section-label">Input quality gate</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Brightness", f"{quality.get('brightness_mean', 0.0):.1f}")
    cols[1].metric("Contrast", f"{quality.get('contrast_std', 0.0):.1f}")
    cols[2].metric("Focus score", f"{quality.get('focus_score', 0.0):.1f}")
    cols[3].metric("Gate", "Pass" if quality.get("passed") else "Review")

    if warnings:
        st.warning(" ".join(warnings))
    else:
        st.success("No validation or quality warnings.")

    with st.expander("Validation metadata", expanded=False):
        st.write(
            {
                "format": validation["format"],
                "original_mode": validation["mode"],
                "dimensions": f"{validation['width']} x {validation['height']} px",
                "file_size_bytes": validation["byte_size"],
                "file_extension": validation["file_extension"],
                "content_sha256": validation["content_sha256"],
                "filename_hash": validation["filename_hash"],
                "quality": quality,
            }
        )
        st.caption(
            "This validates technical suitability for the model pipeline. It is not a clinical "
            "specimen-quality assessment."
        )


def render_xai_local(package) -> None:
    if package.overlay_image and package.heatmap_image:
        cols = st.columns(2)
        cols[0].image(package.overlay_image, caption="Grad-CAM overlay", width="stretch")
        cols[1].image(package.heatmap_image, caption="Raw heatmap", width="stretch")
    elif package.xai_error:
        st.warning(f"Grad-CAM could not be generated: {package.xai_error}")
    else:
        st.info("Grad-CAM disabled for this run.")

    if package.activation_grid:
        with st.expander("Activation map debug view", expanded=False):
            st.image(
                package.activation_grid,
                caption=f"Top feature channels from {package.activation_layer}",
                width="stretch",
            )
            st.caption("Feature maps are for technical inspection, not clinical proof.")


def render_xai_api(payload: dict[str, Any]) -> None:
    xai = payload.get("xai") or {}
    overlay = xai.get("gradcam_overlay")
    heatmap = xai.get("heatmap")
    if overlay and heatmap:
        cols = st.columns(2)
        cols[0].image(data_uri_to_image(overlay), caption="Grad-CAM overlay", width="stretch")
        cols[1].image(data_uri_to_image(heatmap), caption="Raw heatmap", width="stretch")
    elif xai.get("xai_error"):
        st.warning(f"Grad-CAM could not be generated: {xai['xai_error']}")
    else:
        st.info("Grad-CAM disabled for this run.")


def log_rows(limit: int = 100) -> list[dict[str, Any]]:
    return read_recent_logs(limit=limit)


def logs_dataframe(limit: int = 100) -> pd.DataFrame:
    rows = log_rows(limit=limit)
    if not rows:
        return pd.DataFrame()

    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "time": row.get("timestamp_utc"),
                "source": row.get("source"),
                "correlation": (row.get("correlation_id") or "")[:12],
                "class": row.get("predicted_class"),
                "score": round(float(row.get("parasitized_score") or 0.0), 3),
                "threshold": round(float(row.get("threshold") or 0.0), 3),
                "review": str(row.get("review_required")).lower() in {"1", "true"},
                "quality_pass": str(row.get("quality_passed")).lower() in {"1", "true"},
                "focus": round(float(row.get("quality_focus_score") or 0.0), 1),
            }
        )
    return pd.DataFrame(display_rows)


def render_monitoring() -> None:
    summary = summarize_logs(limit=200)
    cols = st.columns(6)
    cols[0].metric("Predictions", summary["total_predictions"])
    cols[1].metric("Review rate", f"{summary['review_rate'] * 100:.1f}%")
    cols[2].metric("Warning rate", f"{summary['validation_warning_rate'] * 100:.1f}%")
    cols[3].metric("Quality pass", f"{summary['quality_pass_rate'] * 100:.1f}%")
    cols[4].metric("Avg focus", f"{summary['avg_focus_score']:.1f}")
    cols[5].metric("Avg brightness", f"{summary['avg_brightness']:.1f}")

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        st.markdown('<div class="section-label">Class mix</div>', unsafe_allow_html=True)
        if summary["class_counts"]:
            st.bar_chart(pd.DataFrame.from_dict(summary["class_counts"], orient="index", columns=["count"]))
        else:
            st.info("Class mix appears after predictions are logged.")
    with right:
        st.markdown('<div class="section-label">Recent audit rows</div>', unsafe_allow_html=True)
        frame = logs_dataframe(limit=12)
        if frame.empty:
            st.info("No local prediction logs yet.")
        else:
            st.dataframe(frame, width="stretch", hide_index=True)


def render_audit_log() -> None:
    frame = logs_dataframe(limit=100)
    if frame.empty:
        st.info("No local audit records yet. Enable logging and run an inference first.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)
    st.caption("Logs are local SQLite/CSV records. Raw uploaded images are not stored.")


def queue_dataframe(limit: int = 50) -> pd.DataFrame:
    rows = read_active_learning_queue(limit=limit)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "time": row.get("timestamp_utc"),
                "request_id": row.get("request_id"),
                "correlation": (row.get("correlation_id") or "")[:12],
                "class": row.get("predicted_class"),
                "score": round(float(row.get("parasitized_score") or 0.0), 3),
                "distance": round(float(row.get("decision_margin") or 0.0), 3),
                "review_reason": row.get("review_reason"),
                "quality_pass": str(row.get("quality_passed")).lower() in {"1", "true"},
            }
            for row in rows
        ]
    )


def feedback_dataframe(limit: int = 100) -> pd.DataFrame:
    rows = read_review_feedback(limit=limit)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def render_review_queue() -> None:
    summary = summarize_review_feedback(limit=500)
    cols = st.columns(3)
    cols[0].metric("Reviewed cases", summary["total_reviews"])
    cols[1].metric("Model disagreement", f"{summary['model_disagreement_rate'] * 100:.1f}%")
    cols[2].metric("Decision mix", str(summary["decision_counts"]))

    queue_rows = read_active_learning_queue(limit=50)
    queue_frame = queue_dataframe(limit=50)
    if queue_frame.empty:
        st.info("No unreviewed high-priority cases in the local queue yet.")
    else:
        st.markdown("#### Active Learning Queue")
        st.dataframe(queue_frame, width="stretch", hide_index=True)

        labels = [
            f"{row.get('timestamp_utc')} | {row.get('predicted_class')} | score={float(row.get('parasitized_score') or 0.0):.3f} | {str(row.get('request_id'))[:8]}"
            for row in queue_rows
        ]
        selected_label = st.selectbox("Select case for review", labels)
        selected_row = queue_rows[labels.index(selected_label)]

        detail_cols = st.columns(4)
        detail_cols[0].metric("Model class", selected_row.get("predicted_class"))
        detail_cols[1].metric("Parasitized score", f"{float(selected_row.get('parasitized_score') or 0.0):.3f}")
        detail_cols[2].metric("Threshold", f"{float(selected_row.get('threshold') or 0.0):.3f}")
        detail_cols[3].metric("Quality pass", "Yes" if str(selected_row.get("quality_passed")).lower() in {"1", "true"} else "No")
        st.caption(selected_row.get("review_reason") or "No review reason recorded.")

        with st.form("review_feedback_form"):
            reviewer_decision = st.radio(
                "Reviewer decision",
                ["correct", "incorrect", "uncertain", "needs_follow_up"],
                horizontal=True,
            )
            reviewer_notes = st.text_area(
                "Reviewer notes",
                placeholder="Optional notes about morphology, image quality, threshold behavior, or follow-up.",
            )
            submitted = st.form_submit_button("Save reviewer feedback")
            if submitted:
                status = write_review_feedback(selected_row, reviewer_decision, reviewer_notes)
                st.success(f"Saved feedback: {status['feedback_id']}")
                st.rerun()

    st.markdown("#### Feedback Records")
    feedback_frame = feedback_dataframe(limit=100)
    if feedback_frame.empty:
        st.caption("No feedback records yet.")
    else:
        st.dataframe(feedback_frame, width="stretch", hide_index=True)
        export_path = export_review_feedback_csv()
        st.caption(f"Feedback export written to `{export_path}`.")
        st.download_button(
            "Download feedback CSV",
            data=feedback_frame.to_csv(index=False),
            file_name="review_feedback_export.csv",
            mime="text/csv",
            width="stretch",
        )


def render_system_notes() -> None:
    st.markdown("#### What this dashboard demonstrates")
    st.markdown(
        """
        - A Streamlit decision-support interface connected to shared inference logic.
        - A FastAPI-compatible architecture for separating frontend and backend concerns.
        - Validation and quality checks before model inference.
        - Configurable threshold and expert-review routing.
        - Grad-CAM explainability and activation-map inspection.
        - Local audit logs with correlation IDs, model hash, and quality metrics.
        - Human-in-the-loop feedback capture for active-learning review queues.
        """
    )
    st.markdown("#### Boundaries")
    st.markdown(
        """
        - This is a non-clinical engineering project, not clinical software.
        - The dataset contains cropped cell images, not full-slide patient-level microscopy.
        - Grad-CAM shows model attention, not medical causality.
        - Quality scoring is heuristic and does not replace expert specimen review.
        """
    )


def sidebar_settings() -> dict[str, Any]:
    with st.sidebar:
        st.header("System Controls")
        inference_mode = st.radio("Inference mode", ["Local model", "FastAPI service"], index=0)
        threshold = st.slider(
            "Parasitized threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(PARASITIZED_THRESHOLD),
            step=0.005,
            help="Default comes from the validation threshold sweep.",
        )
        review_margin = st.slider(
            "Expert-review band",
            min_value=0.0,
            max_value=0.25,
            value=float(DEFAULT_REVIEW_MARGIN),
            step=0.005,
            help="Cases within this distance of the threshold are routed to review.",
        )
        st.divider()
        route_warnings_to_review = st.checkbox("Route validation warnings to review", value=True)
        include_xai = st.checkbox("Generate Grad-CAM", value=True)
        include_activation = st.checkbox("Show activation debug maps", value=False)
        enable_logging = st.checkbox("Write prediction log", value=True)
        correlation_id = st.text_input(
            "Correlation ID",
            value="",
            placeholder="Optional case/session id",
            help="Passed to FastAPI as X-Correlation-ID, or stored in local telemetry if provided.",
        ).strip()

        api_url = os.environ.get("MALARIA_API_URL", "http://127.0.0.1:8000")
        if inference_mode == "FastAPI service":
            st.divider()
            api_url = st.text_input("API URL", value=api_url)
            if st.button("Check API health", width="stretch"):
                try:
                    health = requests.get(api_url.rstrip("/") + "/health", timeout=10).json()
                    st.success(f"API status: {health.get('status')}")
                    st.caption(f"Model: {health.get('model_version')}")
                except Exception as exc:
                    st.error(f"Health check failed: {exc}")
            st.caption("Start API: uvicorn malaria_App.api:app --reload")

    return {
        "inference_mode": inference_mode,
        "threshold": threshold,
        "review_margin": review_margin,
        "route_warnings_to_review": route_warnings_to_review,
        "include_xai": include_xai,
        "include_activation": include_activation,
        "enable_logging": enable_logging,
        "correlation_id": correlation_id or None,
        "api_url": api_url,
    }


def run_local_inference(validated, settings: dict[str, Any]):
    model, model_error = load_model_cached()
    if model_error:
        st.error(model_error)
        st.stop()

    package = diagnose_image(
        model,
        validated,
        threshold=settings["threshold"],
        review_margin=settings["review_margin"],
        include_xai=settings["include_xai"],
        include_activation=settings["include_activation"],
        route_warnings_to_review=settings["route_warnings_to_review"],
        correlation_id=settings["correlation_id"],
    )
    log_status = (
        write_prediction_log(package, source="streamlit-local")
        if settings["enable_logging"]
        else {"enabled": False}
    )
    return package, log_status


def run_api_inference(uploaded_file, uploaded_bytes: bytes, settings: dict[str, Any]) -> dict[str, Any]:
    return post_to_api(
        api_url=settings["api_url"],
        uploaded_bytes=uploaded_bytes,
        filename=uploaded_file.name,
        content_type=uploaded_file.type,
        threshold=settings["threshold"],
        review_margin=settings["review_margin"],
        include_xai=settings["include_xai"],
        route_warnings_to_review=settings["route_warnings_to_review"],
        enable_logging=settings["enable_logging"],
        correlation_id=settings["correlation_id"],
    )


render_header()
settings = sidebar_settings()
render_status_strip(settings)

analysis_tab, monitoring_tab, review_tab, audit_tab, notes_tab = st.tabs(
    ["Analysis Workbench", "Monitoring", "Review Queue", "Audit Log", "System Notes"]
)

with analysis_tab:
    input_col, output_col = st.columns([0.85, 1.15], gap="large")

    with input_col:
        st.markdown('<div class="section-label">Case input</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a cropped thin-smear cell image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            try:
                validated = validate_image_bytes(uploaded_bytes, filename=uploaded_file.name)
            except ImageValidationError as exc:
                st.error(str(exc))
                with st.expander("Validation errors", expanded=True):
                    st.write(exc.details)
                st.stop()

            st.image(validated.image, caption="Uploaded sample", width="stretch")
            render_quality_gate(validated.metadata.to_dict())
        else:
            st.info("Upload an image to run validation, inference, explainability, and logging.")
            validated = None
            uploaded_bytes = None

    with output_col:
        st.markdown('<div class="section-label">Case analysis</div>', unsafe_allow_html=True)
        if uploaded_file is None or validated is None or uploaded_bytes is None:
            st.markdown(
                """
                <div class="panel">
                    <div class="compact-text">
                        The analysis workspace will show the model output, review routing,
                        Grad-CAM, and telemetry once a sample is uploaded.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif settings["inference_mode"] == "Local model":
            with st.spinner("Running local model inference..."):
                package, log_status = run_local_inference(validated, settings)

            render_prediction(package.result.to_dict())
            st.divider()
            st.markdown('<div class="section-label">Explainability</div>', unsafe_allow_html=True)
            render_xai_local(package)
            with st.expander("Technical telemetry", expanded=False):
                st.write(
                    {
                        "request_id": package.result.request_id,
                        "correlation_id": package.result.correlation_id,
                        "model_version": package.result.model_version,
                        "model_sha256": package.result.model_sha256,
                        "tensor_shape": package.tensor_shape,
                        "raw_uninfected_score": package.result.raw_uninfected_score,
                        "parasitized_score": package.result.parasitized_score,
                        "quality": package.validation.quality.to_dict(),
                        "gradcam_layer": package.result.gradcam_layer,
                        "logging": log_status,
                    }
                )
        else:
            with st.spinner("Calling FastAPI inference service..."):
                try:
                    payload = run_api_inference(uploaded_file, uploaded_bytes, settings)
                except Exception as exc:
                    st.error(str(exc))
                    st.stop()

            render_prediction(payload["prediction"])
            st.divider()
            st.markdown('<div class="section-label">Explainability</div>', unsafe_allow_html=True)
            render_xai_api(payload)
            with st.expander("Technical telemetry", expanded=False):
                st.write(
                    {
                        "request_id": payload["prediction"]["request_id"],
                        "correlation_id": payload["prediction"]["correlation_id"],
                        "headers": payload.get("response_headers"),
                        "model": payload.get("model"),
                        "tensor_shape": payload["tensor_shape"],
                        "validation": payload["validation"],
                        "logging": payload.get("logging"),
                        "api_url": settings["api_url"],
                    }
                )

with monitoring_tab:
    render_monitoring()

with review_tab:
    render_review_queue()

with audit_tab:
    render_audit_log()

with notes_tab:
    render_system_notes()

st.caption(
    "Research and educational use only. Not medical advice, not a clinical diagnostic device, and not intended for patient care."
)
