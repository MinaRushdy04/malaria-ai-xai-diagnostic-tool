# pyrefly: ignore [missing-import]
from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

try:
    from .diagnostic_core import (
        DEFAULT_REVIEW_MARGIN,
        PARASITIZED_THRESHOLD,
        ImageValidationError,
        data_uri_to_image,
        diagnose_image,
        load_keras_model,
        read_recent_logs,
        validate_image_bytes,
        write_prediction_log,
    )
except ImportError:
    from diagnostic_core import (
        DEFAULT_REVIEW_MARGIN,
        PARASITIZED_THRESHOLD,
        ImageValidationError,
        data_uri_to_image,
        diagnose_image,
        load_keras_model,
        read_recent_logs,
        validate_image_bytes,
        write_prediction_log,
    )


st.set_page_config(
    page_title="Malaria AI Diagnostic Prototype",
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
    background: #0b1117;
    color: #e5edf5;
}
#MainMenu, footer, header {
    visibility: hidden;
}
.main-title {
    font-size: 2.25rem;
    font-weight: 760;
    letter-spacing: 0;
    color: #f8fafc;
    margin-bottom: 0.25rem;
}
.subtle {
    color: #9aa8b6;
    font-size: 0.96rem;
    line-height: 1.55;
}
.section-label {
    color: #8fd3c7;
    text-transform: uppercase;
    letter-spacing: 0.08rem;
    font-size: 0.78rem;
    font-weight: 760;
    margin-bottom: 0.55rem;
}
.panel {
    background: #111a22;
    border: 1px solid #22303d;
    border-radius: 8px;
    padding: 1.1rem 1.15rem;
    margin-bottom: 1rem;
}
.result-card {
    border-radius: 8px;
    padding: 1.15rem 1.2rem;
    border: 1px solid;
    margin-bottom: 1rem;
}
.result-positive {
    background: rgba(185, 28, 28, 0.16);
    border-color: rgba(248, 113, 113, 0.45);
}
.result-negative {
    background: rgba(21, 128, 61, 0.14);
    border-color: rgba(74, 222, 128, 0.38);
}
.review-card {
    background: rgba(217, 119, 6, 0.15);
    border: 1px solid rgba(251, 191, 36, 0.45);
    border-radius: 8px;
    padding: 0.95rem 1rem;
    margin-bottom: 1rem;
}
.clear-card {
    background: rgba(14, 165, 233, 0.10);
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 8px;
    padding: 0.95rem 1rem;
    margin-bottom: 1rem;
}
.metric-caption {
    color: #9aa8b6;
    font-size: 0.82rem;
}
.small-code {
    font-family: Consolas, monospace;
    color: #c7d2fe;
    font-size: 0.82rem;
}
[data-testid="stFileUploaderDropzone"] {
    background: #0f1720 !important;
    border: 1px dashed #3a4a5a !important;
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
) -> dict[str, Any]:
    endpoint = api_url.rstrip("/") + "/predict"
    response = requests.post(
        endpoint,
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
    return response.json()


def render_prediction(prediction: dict[str, Any]) -> None:
    predicted_class = prediction["predicted_class"]
    result_class = "result-positive" if predicted_class == "Parasitized" else "result-negative"
    result_title = "Parasitized screening flag" if predicted_class == "Parasitized" else "Uninfected screening result"

    st.markdown(
        f"""
        <div class="result-card {result_class}">
            <div class="section-label">Model output</div>
            <h2 style="margin:0 0 0.45rem 0;color:#f8fafc;">{result_title}</h2>
            <div class="subtle">{prediction["recommendation"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Parasitized score", f"{prediction['parasitized_score']:.3f}")
    metric_cols[1].metric("Threshold", f"{prediction['threshold']:.3f}")
    metric_cols[2].metric("Distance", f"{prediction['decision_margin']:.3f}")
    metric_cols[3].metric("Class score", f"{prediction['prediction_score']:.3f}")
    st.progress(
        min(max(float(prediction["parasitized_score"]), 0.0), 1.0),
        text="Parasitized score",
    )

    if prediction["review_required"]:
        st.markdown(
            f"""
            <div class="review-card">
                <strong>Expert review required</strong><br>
                <span class="subtle">{prediction["review_reason"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="clear-card">
                <strong>Outside configured review band</strong><br>
                <span class="subtle">{prediction["review_reason"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_validation(validation: dict[str, Any]) -> None:
    with st.expander("Input validation details", expanded=False):
        st.write(
            {
                "format": validation["format"],
                "original_mode": validation["mode"],
                "dimensions": f"{validation['width']} x {validation['height']} px",
                "file_size_bytes": validation["byte_size"],
                "file_extension": validation["file_extension"],
                "content_sha256": validation["content_sha256"],
                "filename_hash": validation["filename_hash"],
            }
        )
        warnings = validation.get("warnings") or []
        if warnings:
            st.warning(" ".join(warnings))
        else:
            st.success("No validation warnings.")

        st.markdown(
            """
            The validation layer checks that the upload is a supported static image, can be
            decoded safely, is not empty or oversized, has usable dimensions, and can be
            normalized to RGB before inference. It does not prove the image is clinically
            appropriate; questionable inputs are routed to review instead of being silently trusted.
            """
        )


def render_xai_from_local(package) -> None:
    st.markdown('<div class="section-label">Explainability</div>', unsafe_allow_html=True)
    if package.overlay_image and package.heatmap_image:
        left, right = st.columns(2)
        left.image(package.overlay_image, caption="Grad-CAM overlay", width="stretch")
        right.image(package.heatmap_image, caption="Raw Grad-CAM heatmap", width="stretch")
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
            st.caption("Feature maps are useful for technical inspection, not clinical proof.")


def render_xai_from_api(payload: dict[str, Any]) -> None:
    st.markdown('<div class="section-label">Explainability</div>', unsafe_allow_html=True)
    xai = payload.get("xai") or {}
    overlay = xai.get("gradcam_overlay")
    heatmap = xai.get("heatmap")
    if overlay and heatmap:
        left, right = st.columns(2)
        left.image(data_uri_to_image(overlay), caption="Grad-CAM overlay", width="stretch")
        right.image(data_uri_to_image(heatmap), caption="Raw Grad-CAM heatmap", width="stretch")
    elif xai.get("xai_error"):
        st.warning(f"Grad-CAM could not be generated: {xai['xai_error']}")
    else:
        st.info("Grad-CAM disabled for this run.")


def render_recent_logs() -> None:
    rows = read_recent_logs(limit=12)
    if not rows:
        st.caption("No local prediction logs yet.")
        return

    display_rows = [
        {
            "time": row["timestamp_utc"],
            "source": row["source"],
            "class": row["predicted_class"],
            "score": round(float(row["parasitized_score"]), 3),
            "threshold": round(float(row["threshold"]), 3),
            "review": bool(int(row["review_required"])),
        }
        for row in rows
    ]
    st.dataframe(display_rows, width="stretch", hide_index=True)


st.markdown('<div class="main-title">Malaria Cell-Smear AI Diagnostic Prototype</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">Responsible AI workflow with threshold-aware inference, Grad-CAM, review routing, input validation, and prediction logging.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Run settings")
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
    route_warnings_to_review = st.checkbox("Route validation warnings to review", value=True)
    include_xai = st.checkbox("Generate Grad-CAM", value=True)
    include_activation = st.checkbox("Show activation debug maps", value=False)
    enable_logging = st.checkbox("Write prediction log", value=True)

    api_url = os.environ.get("MALARIA_API_URL", "http://127.0.0.1:8000")
    if inference_mode == "FastAPI service":
        api_url = st.text_input("API URL", value=api_url)
        if st.button("Check API health"):
            try:
                health = requests.get(api_url.rstrip("/") + "/health", timeout=10).json()
                st.success(health)
            except Exception as exc:
                st.error(f"Health check failed: {exc}")
        st.caption("Start the service with: uvicorn malaria_App.api:app --reload")

    with st.expander("Recent local logs", expanded=False):
        render_recent_logs()


left_col, right_col = st.columns([0.95, 1.15], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a cropped thin-smear cell image",
        type=["jpg", "jpeg", "png"],
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        try:
            validated = validate_image_bytes(uploaded_bytes, filename=uploaded_file.name)
        except ImageValidationError as exc:
            st.error(str(exc))
            with st.expander("Validation errors", expanded=True):
                st.write(exc.details)
            st.stop()

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)
        st.image(validated.image, width="stretch")
        st.markdown(
            f'<div class="metric-caption">{validated.metadata.width} x {validated.metadata.height} px, {validated.metadata.format}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        render_validation(validated.metadata.to_dict())
    else:
        st.info("Upload an image to run the diagnostic workflow.")

with right_col:
    if uploaded_file is None:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtle">Inference, review routing, Grad-CAM, and telemetry will appear here after an image is uploaded.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    elif inference_mode == "Local model":
        with st.spinner("Loading local model and running inference..."):
            model, model_error = load_model_cached()
            if model_error:
                st.error(model_error)
                st.stop()

            package = diagnose_image(
                model,
                validated,
                threshold=threshold,
                review_margin=review_margin,
                include_xai=include_xai,
                include_activation=include_activation,
                route_warnings_to_review=route_warnings_to_review,
            )
            log_status = write_prediction_log(package, source="streamlit-local") if enable_logging else {"enabled": False}

        render_prediction(package.result.to_dict())
        render_xai_from_local(package)

        with st.expander("Technical telemetry", expanded=False):
            st.write(
                {
                    "request_id": package.result.request_id,
                    "tensor_shape": package.tensor_shape,
                    "raw_uninfected_score": package.result.raw_uninfected_score,
                    "parasitized_score": package.result.parasitized_score,
                    "gradcam_layer": package.result.gradcam_layer,
                    "logging": log_status,
                }
            )

    else:
        with st.spinner("Calling FastAPI inference service..."):
            try:
                payload = post_to_api(
                    api_url=api_url,
                    uploaded_bytes=uploaded_bytes,
                    filename=uploaded_file.name,
                    content_type=uploaded_file.type,
                    threshold=threshold,
                    review_margin=review_margin,
                    include_xai=include_xai,
                    route_warnings_to_review=route_warnings_to_review,
                    enable_logging=enable_logging,
                )
            except Exception as exc:
                st.error(str(exc))
                st.stop()

        render_prediction(payload["prediction"])
        render_xai_from_api(payload)

        with st.expander("Technical telemetry", expanded=False):
            st.write(
                {
                    "request_id": payload["prediction"]["request_id"],
                    "tensor_shape": payload["tensor_shape"],
                    "validation": payload["validation"],
                    "logging": payload.get("logging"),
                    "api_url": api_url,
                }
            )


st.caption(
    "Research and educational use only. This is not medical advice, not a clinical diagnostic device, and not intended for patient care."
)
