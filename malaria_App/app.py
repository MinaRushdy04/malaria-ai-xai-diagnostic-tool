# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import numpy as np
from pathlib import Path
from PIL import Image
import time

# ─── Page Config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Malaria-AI | Cell Diagnostic System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Inject CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;800;900&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* ── Animated background ── */
.stApp {
    background: linear-gradient(135deg, #080b1a 0%, #0d1b2a 40%, #1a1240 100%);
    background-size: 300% 300%;
    animation: bgShift 18s ease infinite;
    min-height: 100vh;
}
@keyframes bgShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero title ── */
.hero-title {
    text-align: center;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00f2fe, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}
.hero-sub {
    text-align: center;
    font-size: 1.05rem;
    font-weight: 300;
    color: #94a3b8;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* ── Glassmorphism card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 28px 32px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
    margin-bottom: 24px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 50px rgba(0,0,0,0.6);
}

/* ── Section heading ── */
.section-heading {
    font-size: 1.05rem;
    font-weight: 600;
    color: #c4b5fd;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* ── File uploader override ── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(167,139,250,0.06) !important;
    border: 2px dashed rgba(167,139,250,0.4) !important;
    border-radius: 14px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #a78bfa !important;
    background: rgba(167,139,250,0.12) !important;
}

/* ── Diagnosis verdict cards ── */
.verdict-parasitized {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(185,28,28,0.22));
    border-left: 5px solid #ef4444;
    border-radius: 14px;
    padding: 24px 28px;
    box-shadow: 0 6px 30px rgba(239,68,68,0.18);
    animation: pulseRed 2.5s infinite ease-in-out;
}
@keyframes pulseRed {
    0%,100% { box-shadow: 0 6px 30px rgba(239,68,68,0.18); }
    50%      { box-shadow: 0 6px 44px rgba(239,68,68,0.38); }
}

.verdict-uninfected {
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(21,128,61,0.22));
    border-left: 5px solid #22c55e;
    border-radius: 14px;
    padding: 24px 28px;
    box-shadow: 0 6px 30px rgba(34,197,94,0.18);
    animation: pulseGreen 2.5s infinite ease-in-out;
}
@keyframes pulseGreen {
    0%,100% { box-shadow: 0 6px 30px rgba(34,197,94,0.18); }
    50%      { box-shadow: 0 6px 44px rgba(34,197,94,0.38); }
}

/* ── Confidence number ── */
.confidence-value {
    font-size: 4rem;
    font-weight: 900;
    line-height: 1;
    margin: 6px 0 4px;
    letter-spacing: -2px;
}

/* ── Info badge ── */
.info-badge {
    display: inline-block;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #c4b5fd;
    font-weight: 500;
    margin-right: 8px;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
}

/* ── Idle placeholder ── */
.idle-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 340px;
    color: #475569;
    text-align: center;
}
.idle-icon { font-size: 5rem; opacity: 0.35; margin-bottom: 16px; }
.idle-text  { font-size: 1rem; font-weight: 300; max-width: 70%; line-height: 1.7; }

/* ── Telemetry block ── */
.telemetry {
    background: rgba(0,0,0,0.25);
    border-radius: 10px;
    padding: 14px 18px;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #64748b;
    line-height: 1.9;
}
.telemetry span { color: #a78bfa; }

/* ── Streamlit progress bar label color ── */
div[data-testid="stProgress"] p { color: #94a3b8 !important; }

/* ── Expander text ── */
.streamlit-expanderHeader { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_keras_model():
    import tensorflow as tf
    import h5py
    import numpy as np
    MODEL_PATH = Path(__file__).with_name("malaria_cell_parasite_prediction_model.h5")

    # ── Strategy 1: plain Keras load (works if Keras version matches) ─────────
    try:
        m = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        return m, None
    except Exception:
        pass

    # ── Strategy 2: Keras 3 h5 — rebuild arch + inject weights manually ───────
    # H5 key path confirmed: model_weights/dense/sequential/dense/{kernel,bias}
    try:
        # Rebuild the exact Sequential architecture from the notebook
        base = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        base.trainable = False
        model = tf.keras.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        # Force a build so layers are initialised
        model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)

        # Inject dense weights from the h5 file
        with h5py.File(str(MODEL_PATH), "r") as f:
            mw = f["model_weights"]
            kernel = np.array(mw["dense"]["sequential"]["dense"]["kernel"])  # (1280,1)
            bias   = np.array(mw["dense"]["sequential"]["dense"]["bias"])    # (1,)
        model.layers[-1].set_weights([kernel, bias])   # Dense is last layer

        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


BILINEAR_RESAMPLE = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
PARASITIZED_THRESHOLD = 0.285


def normalise_map(values):
    values = values.astype(np.float32)
    values = values - np.min(values)
    max_value = np.max(values)
    if max_value > 0:
        values = values / max_value
    return values


def get_feature_layer(model):
    base_model = model.layers[0]
    for layer in reversed(base_model.layers):
        try:
            if len(layer.output.shape) == 4:
                return base_model, layer
        except Exception:
            continue
    raise ValueError("No spatial feature layer found for Grad-CAM.")


def apply_classifier_head(model, features):
    x = features
    for layer in model.layers[1:]:
        try:
            x = layer(x, training=False)
        except TypeError:
            x = layer(x)
    return x


def make_gradcam_heatmap(model, img_batch, predicted_class_index):
    import tensorflow as tf

    base_model, feature_layer = get_feature_layer(model)
    feature_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[feature_layer.output, base_model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = feature_model(img_batch, training=False)
        predictions = apply_classifier_head(model, base_outputs)
        uninfected_score = predictions[:, 0]
        target_score = uninfected_score if predicted_class_index == 1 else 1.0 - uninfected_score

    grads = tape.gradient(target_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy(), feature_layer.name


def colorize_heatmap(heatmap):
    heatmap = normalise_map(heatmap)
    red = np.clip(255 * np.minimum(1.0, heatmap * 2.2), 0, 255)
    green = np.clip(255 * np.maximum(0.0, (heatmap - 0.28) / 0.72), 0, 255)
    blue = np.clip(120 * (1.0 - heatmap), 0, 120)
    rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def build_gradcam_images(original_image, heatmap, alpha=0.48):
    heatmap = normalise_map(heatmap)
    heatmap_image = colorize_heatmap(heatmap).resize(original_image.size, BILINEAR_RESAMPLE)
    alpha_mask = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        original_image.size,
        BILINEAR_RESAMPLE,
    )

    base = np.asarray(original_image).astype(np.float32)
    heat = np.asarray(heatmap_image).astype(np.float32)
    mask = (np.asarray(alpha_mask).astype(np.float32) / 255.0) ** 1.35
    overlay = base * (1.0 - alpha * mask[..., None]) + heat * (alpha * mask[..., None])
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return heatmap_image, Image.fromarray(overlay)


def make_activation_grid(model, img_batch, max_maps=8, tile_size=96):
    import tensorflow as tf

    base_model, feature_layer = get_feature_layer(model)
    activation_model = tf.keras.Model(inputs=base_model.inputs, outputs=feature_layer.output)
    activations = activation_model(img_batch, training=False)[0].numpy()
    channel_strength = np.mean(np.abs(activations), axis=(0, 1))
    selected_channels = np.argsort(channel_strength)[-max_maps:][::-1]

    columns = 4
    rows = int(np.ceil(len(selected_channels) / columns))
    grid = Image.new("RGB", (columns * tile_size, rows * tile_size), color=(8, 11, 26))
    for index, channel in enumerate(selected_channels):
        feature_map = normalise_map(activations[:, :, channel])
        tile = colorize_heatmap(feature_map).resize((tile_size, tile_size), BILINEAR_RESAMPLE)
        x = (index % columns) * tile_size
        y = (index // columns) * tile_size
        grid.paste(tile, (x, y))

    return grid, feature_layer.name


# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>🔬 Malaria-AI Diagnostic</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>MobileNetV2 · Transfer Learning · Cell-Image Analysis</div>", unsafe_allow_html=True)

# ─── Load Model (with visible status) ─────────────────────────────────────────
with st.spinner("Initialising neural network weights…"):
    model, model_error = load_keras_model()

if model_error:
    st.error(f"⚠️ {model_error}")
    st.stop()

# ─── Two-column layout ────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.15, 1], gap="large")

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN  –  Upload + Preview
# ═══════════════════════════════════════════════════════════════════════════════
with col_left:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-heading'>📤 Upload Blood Cell Image</p>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b;font-size:0.88rem;margin-bottom:14px;'>"
        "Supported: JPG · PNG · JPEG &nbsp;|&nbsp; Thin-smear microscopy images recommended"
        "</p>",
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        label="Upload image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-heading'>🖼️ Uploaded Cell Sample</p>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown(
            f"<p style='color:#475569;font-size:0.8rem;margin-top:8px;'>"
            f"Original size: {image.size[0]} × {image.size[1]} px"
            f"</p>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN  –  Analysis + Result
# ═══════════════════════════════════════════════════════════════════════════════
with col_right:

    # ── Idle placeholder ──────────────────────────────────────────────────────
    if uploaded_file is None:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("""
        <div class='idle-box'>
            <div class='idle-icon'>🧫</div>
            <p class='idle-text'>
                Upload a thin-smear blood-cell image on the left to trigger
                the AI diagnostic pipeline.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Live analysis ─────────────────────────────────────────────────────────
    else:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-heading'>⚡ Running Diagnostic Pipeline</p>", unsafe_allow_html=True)

        # ── Dramatic staged loader ─────────────────────────────────────────
        stages = [
            "Loading MobileNetV2 inference graph…",
            "Resizing & normalising pixel tensor (224×224)…",
            "Extracting feature maps from frozen layers…",
            "Running sigmoid classification head…",
        ]
        prog = st.progress(0)
        status_text = st.empty()
        for i, stage in enumerate(stages):
            status_text.markdown(f"<p style='color:#94a3b8;font-size:0.85rem;'>{stage}</p>", unsafe_allow_html=True)
            prog.progress(int((i + 1) / len(stages) * 100))
            time.sleep(0.35)
        status_text.empty()
        prog.empty()

        # ── Preprocessing ─────────────────────────────────────────────────
        img_resized   = image.resize((224, 224))
        img_array     = np.array(img_resized, dtype=np.float32) / 255.0
        img_batch     = np.expand_dims(img_array, axis=0)

        # ── Model inference ───────────────────────────────────────────────
        #   Class mapping (from training): 0 = Parasitized, 1 = Uninfected
        raw_score = float(model.predict(img_batch, verbose=0)[0][0])
        parasitized_score = 1.0 - raw_score
        predicted_class_index = 0 if parasitized_score >= PARASITIZED_THRESHOLD else 1

        if predicted_class_index == 1:
            # Closer to 1.0 → Uninfected
            label      = "UNINFECTED"
            confidence = raw_score * 100
            card_class = "verdict-uninfected"
            icon       = "✅"
            title_color = "#86efac"
            desc = (
                "No malarial parasites detected in this cell sample. "
                "The morphology appears consistent with a healthy red blood cell."
            )
        else:
            # Closer to 0.0 → Parasitized
            label      = "PARASITIC INFECTION"
            confidence = parasitized_score * 100
            card_class = "verdict-parasitized"
            icon       = "⚠️"
            title_color = "#fca5a5"
            desc = (
                "The model has identified features consistent with <em>Plasmodium</em> "
                "infection. Please confirm with a certified medical professional."
            )

        st.markdown("</div>", unsafe_allow_html=True)   # close pipeline card

        # ── Verdict card ──────────────────────────────────────────────────
        st.markdown(f"""
        <div class='{card_class}'>
            <h2 style='color:{title_color};margin:0 0 4px;font-size:1.45rem;font-weight:800;'>
                {icon}&nbsp; {label}
            </h2>
            <div class='confidence-value' style='color:{title_color};'>
                {confidence:.1f}%
            </div>
            <p style='color:#94a3b8;font-size:0.82rem;margin:0 0 14px;font-weight:500;'>
                Model Score
            </p>
            <hr style='border:none;border-top:1px solid rgba(255,255,255,0.1);margin:14px 0;'>
            <p style='color:#cbd5e1;line-height:1.65;font-size:0.93rem;margin:0;'>
                {desc}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bar ────────────────────────────────────────────────
        st.progress(min(int(confidence), 100), text=f"Model score: {confidence:.1f} %")

        # Explainability: Grad-CAM + feature-map debug view
        try:
            heatmap, gradcam_layer_name = make_gradcam_heatmap(
                model,
                img_batch,
                predicted_class_index,
            )
            heatmap_image, overlay_image = build_gradcam_images(image, heatmap)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-heading'>Explainability: Grad-CAM</p>", unsafe_allow_html=True)
            st.markdown(
                "<p style='color:#94a3b8;font-size:0.88rem;line-height:1.6;margin-top:-6px;'>"
                "The highlighted regions contributed most strongly to the model's predicted class. "
                "This is an attention-style explanation, not medical proof of causality."
                "</p>",
                unsafe_allow_html=True,
            )
            gradcam_col, heatmap_col = st.columns(2)
            with gradcam_col:
                st.image(overlay_image, caption="Grad-CAM overlay", use_column_width=True)
            with heatmap_col:
                st.image(heatmap_image, caption="Raw attention heatmap", use_column_width=True)

            with st.expander("Activation map debug view", expanded=False):
                activation_grid, activation_layer_name = make_activation_grid(model, img_batch)
                st.image(
                    activation_grid,
                    caption=f"Top feature channels from {activation_layer_name}",
                    use_column_width=True,
                )
                st.caption(
                    "These feature maps are useful for technical inspection, but they are not a "
                    "clinician-facing explanation of diagnosis."
                )
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as xai_error:
            st.warning(f"Grad-CAM explanation could not be generated: {xai_error}")
            gradcam_layer_name = "unavailable"

        # ── Telemetry expander ────────────────────────────────────────────
        with st.expander("📡  Technical Telemetry", expanded=False):
            st.markdown(f"""
            <div class='telemetry'>
                <span>[MODEL]</span> Architecture  : MobileNetV2 (frozen) + GAP + Dropout(0.2) + Dense(sigmoid)<br>
                <span>[XAI]</span> Grad-CAM Layer  : {gradcam_layer_name}<br>
                <span>[MODEL]</span> Trainable Params : 1,281 / 2,259,265 total<br>
                <span>[INPUT]</span> Upload Size    : {image.size[0]}×{image.size[1]} px<br>
                <span>[INPUT]</span> Tensor Shape   : (1, 224, 224, 3)<br>
                <span>[INFER]</span> Raw Sigmoid    : {raw_score:.8f}<br>
                <span>[INFER]</span> Parasitized Score : {parasitized_score:.8f}<br>
                <span>[INFER]</span> Parasitized Threshold : {PARASITIZED_THRESHOLD:.3f}<br>
                <span>[INFER]</span> Class Mapping  : 0 = Parasitized · 1 = Uninfected<br>
                <span>[RESULT]</span> Predicted Class: {label}<br>
                <span>[RESULT]</span> Model Score     : {confidence:.4f}%
            </div>
            """, unsafe_allow_html=True)

        # ── Disclaimer ────────────────────────────────────────────────────
        st.markdown("""
        <p style='color:#334155;font-size:0.78rem;margin-top:14px;line-height:1.6;'>
            ⚕️ <em>This tool is intended for research and educational purposes only.
            It does not constitute medical advice. Always consult a qualified
            healthcare professional for clinical diagnosis.</em>
        </p>
        """, unsafe_allow_html=True)
