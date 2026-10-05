"""
Eye Anomaly Detection - Streamlit App

Run from the project root with:
    streamlit run app/streamlit_app.py
"""

import os
import sys
from datetime import datetime

import streamlit as st
import torch
from PIL import Image
from fpdf import FPDF

# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_resnet50_5class.pt")
MODEL_KEY = "resnet50"

# Second-opinion model (used for the cross-model agreement check)
# NOTE: after the data-augmentation bug fix and retrain, ResNet50 outperformed
# EfficientNet-B0 on the real test set (97% vs 96% accuracy, 0.94 vs 0.92
# macro F1) - so the primary/second-opinion roles were swapped here. Variable
# names below (RESNET_MODEL_PATH, load_resnet_model, resnet_* variables)
# were NOT renamed throughout the file to limit the size of this change;
# they now refer to the EfficientNet-B0 checkpoint, not literally ResNet50.
# All user-facing text/labels ARE correctly swapped - only internal
# variable/function names still carry the old naming.
RESNET_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_5class.pt")
RESNET_MODEL_KEY = "efficientnet_b0"

# Design tokens (kept in one place so the ring generator and CSS agree)
COLOR_AMBER = "#E1793F"   # Fundus Amber - primary model accent (brightened for dark bg)
COLOR_TEAL = "#2FBFB0"    # Iris Teal - second-opinion accent (brightened for dark bg)
COLOR_NAVY = "#F4F8FB"    # Near-white text on dark background

# -------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------

from src.models import build_model
from src.data_loader import get_transforms, CLASS_NAMES
from src.gradcam import generate_gradcam, preprocess_for_display
from src.uncertainty import mc_dropout_predict, confidence_flag


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Eye Anomaly Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# Custom styling - deep clinical dark theme
# -------------------------------------------------------------------

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;700&display=swap');

    /* Overall page */
    .stApp {
        background-color: #060B12;
        background-image:
            radial-gradient(circle at 15% 10%, rgba(225,121,63,0.10) 0%, transparent 45%),
            radial-gradient(circle at 85% 25%, rgba(47,191,176,0.10) 0%, transparent 45%);
        font-family: 'Inter', sans-serif;
        color: #EAF1F7;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    p, li, span, label, .stMarkdown, .stCaption {
        color: #C7D2DC;
    }

    h1, h2, h3, h4 {
        color: #EAF1F7 !important;
    }

    /* --- Hero: animated iris-ring signature element --- */
    .hero-wrap {
        display: flex;
        align-items: center;
        gap: 22px;
        margin-bottom: 4px;
    }

    .iris-ring {
        position: relative;
        width: 84px;
        height: 84px;
        flex-shrink: 0;
        border-radius: 50%;
        background: conic-gradient(from 0deg, #E1793F, #2FBFB0, #E1793F);
        animation: spin-slow 16s linear infinite;
        padding: 6px;
        box-sizing: border-box;
    }

    .iris-ring::before {
        content: "";
        position: absolute;
        inset: 6px;
        border-radius: 50%;
        background: #060B12;
    }

    .iris-ring::after {
        content: "";
        position: absolute;
        inset: 22px;
        border-radius: 50%;
        background: radial-gradient(circle, #0B2D48 0%, #050708 100%);
        box-shadow: 0 0 14px rgba(47,191,176,0.5);
        animation: pulse-glow 3.5s ease-in-out infinite;
    }

    @keyframes spin-slow {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 10px rgba(47,191,176,0.35); }
        50% { box-shadow: 0 0 22px rgba(225,121,63,0.55); }
    }

    /* Header */
    .main-title {
        font-family: 'Fraunces', serif;
        font-size: 44px;
        font-weight: 700;
        color: #F4F8FB;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }

    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 17px;
        color: #93A3B3;
        margin-bottom: 22px;
    }

    /* Section titles */
    .section-title {
        font-family: 'Fraunces', serif;
        font-size: 26px;
        font-weight: 600;
        color: #F4F8FB;
        margin-top: 26px;
        margin-bottom: 14px;
    }

    /* Result cards */
    .result-card {
        background: #101823;
        padding: 22px 18px;
        border-radius: 14px;
        border: 1px solid #1D2937;
        border-left: 4px solid #2A3B4D;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
        text-align: center;
        min-height: 105px;
    }

    .result-card.accent-amber {
        border-left: 4px solid #E1793F;
        box-shadow: 0 4px 18px rgba(225,121,63,0.12);
    }

    .result-card.accent-teal {
        border-left: 4px solid #2FBFB0;
        box-shadow: 0 4px 18px rgba(47,191,176,0.12);
    }

    .result-card.accent-warn {
        border-left: 4px solid #F5A623;
        box-shadow: 0 4px 18px rgba(245,166,35,0.12);
    }

    .result-card.accent-success {
        border-left: 4px solid #3ED598;
        box-shadow: 0 4px 18px rgba(62,213,152,0.12);
    }

    .result-label {
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        color: #7C8CA0;
        margin-bottom: 8px;
        font-weight: 600;
    }

    .result-value {
        font-family: 'Fraunces', serif;
        font-size: 24px;
        font-weight: 700;
        color: #F4F8FB;
    }

    /* Confidence ring - the app's signature element, reused per-prediction */
    .confidence-ring {
        position: relative;
        width: 84px;
        height: 84px;
        margin: 6px auto 0 auto;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .confidence-ring-inner {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        background: #0B1420;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        color: #F4F8FB;
        font-size: 15px;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background: #101823;
        padding: 18px;
        border-radius: 14px;
        border: 1px dashed #35485D;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 48px;
        font-size: 16px;
        font-weight: 650;
        font-family: 'Inter', sans-serif;
        background-color: #E1793F !important;
        color: #0B1420 !important;
        border: none !important;
        transition: box-shadow 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #F08A4F !important;
        color: #0B1420 !important;
        box-shadow: 0 0 18px rgba(225,121,63,0.5);
    }

    /* Information box */
    .info-card {
        background: #101823;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #1D2937;
        border-left: 4px solid #2FBFB0;
        margin-top: 18px;
    }

    .info-title {
        font-family: 'Fraunces', serif;
        font-size: 18px;
        font-weight: 700;
        color: #F4F8FB;
        margin-bottom: 8px;
    }

    .info-text {
        font-family: 'Inter', sans-serif;
        color: #93A3B3;
        font-size: 15px;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


def confidence_ring_html(confidence_pct: float, color: str) -> str:
    """
    Renders the confidence-ring signature element: a circular progress ring
    whose fill directly encodes the confidence percentage, with the number
    printed inside in monospace (an instrument-readout feel).
    """
    degrees = max(0, min(360, confidence_pct * 3.6))
    return f"""
    <div class="confidence-ring" style="background: conic-gradient({color} {degrees}deg, #1D2937 {degrees}deg);">
        <div class="confidence-ring-inner">{confidence_pct:.0f}%</div>
    </div>
    """


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <div class="iris-ring"></div>
        <div>
            <div class="main-title">Eye Anomaly Detection</div>
            <div class="subtitle">AI-Assisted Eye Anomaly Detection &amp; Explainability</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Intelligent screening • Confidence analysis • Explainable AI"
)

st.info(
    "🛡️ **Academic Research Notice** — This is a student research "
    "project for demonstration purposes. Its predictions are not medical "
    "diagnoses and should not replace evaluation by a qualified eye-care professional."
)

with st.expander("ℹ️ Model & Technology Information"):

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown(
            """
            **🧠 Model Architecture**
            EfficientNet-B0 was the original primary model; after fixing a data-augmentation bug and retraining, ResNet50 outperformed it on the real test set and is now primary

            **📦 Dataset**
            Kaggle — Cataract Dataset, APTOS 2019, ODIR-5K, RFMiD (~3,500 images)

            **🏷️ Classes Detected**
            Normal · Cataract · Glaucoma · Age-related Macular Degeneration · Diabetic Retinopathy
            """
        )

    with info_col2:
        st.markdown(
            """
            **📊 Uncertainty Method**
            Monte Carlo Dropout (30 stochastic passes)

            **🔬 Explainability**
            Grad-CAM (Gradient-weighted Class Activation Mapping)

            **⚖️ Architecture Comparison**
            Primary: ResNet50 — benchmarked against EfficientNet-B0, cross-checked live on every prediction
            """
        )

# -------------------------------------------------------------------
# How it works
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">⚙️ How It Works</div>',
    unsafe_allow_html=True,
)

how_col1, how_col2, how_col3, how_col4 = st.columns(4)

how_it_works_steps = [
    (
        "📤",
        "1. Upload",
        "Provide a clear eye or fundus image in JPG or PNG format.",
    ),
    (
        "🧠",
        "2. AI Prediction",
        "ResNet50 analyzes the image and estimates class probabilities.",
    ),
    (
        "📊",
        "3. Confidence Check",
        "Monte Carlo Dropout runs 30 passes to measure prediction uncertainty.",
    ),
    (
        "🔬",
        "4. Explainability",
        "Grad-CAM highlights the image regions that influenced the result.",
    ),
]

for how_col, (icon, title, desc) in zip(
    [how_col1, how_col2, how_col3, how_col4],
    how_it_works_steps,
):

    with how_col:
        st.markdown(
            f"""
            <div class="result-card" style="text-align:left; min-height:150px;">
                <div style="font-size:26px;">{icon}</div>
                <div style="font-family:'Fraunces', serif; font-weight:700; color:#F4F8FB; margin-top:6px;">
                    {title}
                </div>
                <div style="font-size:13px; color:#93A3B3; margin-top:6px;">
                    {desc}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:10px 0 20px 0;">
            <div style="display:flex; justify-content:center;">
                <div class="iris-ring" style="width:56px; height:56px;"></div>
            </div>
            <div style="font-family:'Fraunces', serif; font-size:22px; font-weight:700; color:#F4F8FB; margin-top:10px;">
                Eye Anomaly Detection
            </div>
            <div style="font-size:13px; color:#93A3B3;">
                AI-Assisted Screening System
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🧠 About the System")

    st.write(
        "This is a machine-learning based research system "
        "designed to analyze eye images and identify patterns "
        "associated with selected eye anomalies."
    )

    st.markdown("### 🔬 Analysis Pipeline")

    st.markdown(
        """
        **1. 📤 Image Upload**

        Upload an eye/fundus image.

        **2. 🧠 AI Prediction**

        The trained deep-learning model analyzes the image.

        **3. 📊 Confidence Analysis**

        Prediction probabilities and uncertainty are estimated.

        **4. 🔎 Second Opinion**

        A second, independently-trained architecture (EfficientNet-B0) cross-checks the result.

        **5. 🔬 Explainability**

        Grad-CAM highlights regions influencing the prediction.

        **6. 💡 Screening Guidance**

        The system provides responsible AI-based guidance.
        """
    )

    st.markdown("---")

    st.markdown("### 📌 System Features")

    st.markdown(
        """
        - 🧠 Deep Learning Classification
        - 📊 Probability Analysis
        - ⚠️ Uncertainty Estimation
        - 🔎 Second-Opinion Cross-Check (EfficientNet-B0)
        - 🔬 Grad-CAM Explainability
        - 💡 Screening Guidance
        - 📄 PDF Analysis Report
        """
    )

    st.markdown("---")

    st.caption(
        "⚠️ Academic research project only. "
        "Not intended for medical diagnosis."
    )

# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------

@st.cache_resource
def load_model():
    model = build_model(
        MODEL_KEY,
        num_classes=len(CLASS_NAMES),
        freeze_backbone=False,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


@st.cache_resource
def load_resnet_model():
    """Second-opinion model, used only for the cross-model agreement check."""
    model = build_model(
        RESNET_MODEL_KEY,
        num_classes=len(CLASS_NAMES),
        freeze_backbone=False,
    )

    checkpoint = torch.load(
        RESNET_MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


# -------------------------------------------------------------------
# PDF report generation
# -------------------------------------------------------------------

def generate_pdf_report(
    pred_class,
    confidence,
    uncertain_flag,
    image_path,
    second_opinion_class=None,
    second_opinion_confidence=None,
    models_agree=None,
):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(
        0,
        10,
        "Eye Anomaly Detection - Analysis Report",
        ln=True,
    )

    pdf.set_font("Helvetica", "", 11)

    pdf.cell(
        0,
        8,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ln=True,
    )

    pdf.ln(5)

    if os.path.exists(image_path):
        pdf.image(image_path, w=80)

    pdf.ln(5)

    pdf.cell(
        0,
        8,
        f"Predicted class (ResNet50): {pred_class}",
        ln=True,
    )

    pdf.cell(
        0,
        8,
        f"Model confidence: {confidence:.1f}%",
        ln=True,
    )

    if second_opinion_class is not None:
        pdf.cell(
            0,
            8,
            f"Second opinion (EfficientNet-B0): {second_opinion_class} "
            f"({second_opinion_confidence:.1f}%)",
            ln=True,
        )

        if models_agree is False:
            pdf.set_text_color(200, 120, 0)
            pdf.cell(
                0,
                8,
                "Note: the two models disagree on this image. "
                "Manual review is recommended.",
                ln=True,
            )
            pdf.set_text_color(0, 0, 0)

    if uncertain_flag:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(
            0,
            8,
            "Warning: Low-confidence prediction. Specialist review is recommended.",
            ln=True,
        )
        pdf.set_text_color(0, 0, 0)

    pdf.ln(5)

    pdf.set_font("Helvetica", "I", 9)

    pdf.multi_cell(
        0,
        6,
        "Disclaimer: This report is generated by a student research "
        "project for academic purposes only. It is not a substitute "
        "for professional medical diagnosis.",
    )

    output_path = os.path.join(PROJECT_ROOT, "report.pdf")

    pdf.output(output_path)

    return output_path


# -------------------------------------------------------------------
# Upload section
# -------------------------------------------------------------------

st.markdown("## 📤 Upload Eye Image")

st.caption(
    "Upload a clear eye or fundus image to begin AI-assisted screening."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG",
)

with st.expander("📸 Image Quality Guidelines — Tips for Best Results"):

    tip_col1, tip_col2 = st.columns(2)

    with tip_col1:
        st.markdown(
            """
            **✅ Do**
            - Use a clear, in-focus eye or fundus image
            - Ensure even, adequate lighting
            - Center the eye/pupil in the frame
            - Use high-resolution images when possible
            """
        )

    with tip_col2:
        st.markdown(
            """
            **🚫 Avoid**
            - Blurry or low-resolution photos
            - Heavy glare or reflections
            - Extreme angles or partial eye views
            - Images with obstructions (hair, glasses glare, etc.)
            """
        )

    st.caption(
        "Poor image quality can reduce model accuracy and increase "
        "prediction uncertainty."
    )

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    preview_col1, preview_col2 = st.columns(
        [1.15, 0.85],
        gap="large"
    )

    with preview_col1:

        st.markdown("### 🖼️ Uploaded Image")

        st.image(
            image,
            caption="Ready for AI analysis",
            use_container_width=True,
        )

    with preview_col2:

        st.markdown("### 🔬 What happens next?")

        st.info(
            "Your image will pass through the AI analysis pipeline."
        )

        st.markdown(
            """
            **🧠 AI Classification**  
            Identifies patterns associated with the trained classes.

            **📊 Confidence Analysis**  
            Estimates prediction confidence and uncertainty.

            **🔎 Second Opinion**  
            A second architecture (EfficientNet-B0) cross-checks the result.

            **🔬 Grad-CAM Explainability**  
            Shows regions that influenced the model's prediction.

            **💡 Screening Guidance**  
            Provides responsible, non-diagnostic guidance.
            """
        )

    st.markdown("### 🚀 Ready to Analyze?")

    analyze_button = st.button(
        "🔍 Analyze Eye Image",
        use_container_width=True,
        type="primary",
    )

    if analyze_button:

        progress_bar = st.progress(0, text="Loading model...")

        # -----------------------------------------------------------
        # Model + preprocessing
        # -----------------------------------------------------------

        model = load_model()

        transform = get_transforms(train=False)

        input_tensor = transform(image).unsqueeze(0)

        progress_bar.progress(25, text="Running inference...")

        # -----------------------------------------------------------
        # Prediction + uncertainty
        # -----------------------------------------------------------

        mean_probs, uncertainty, predicted_idx = mc_dropout_predict(
            model,
            input_tensor,
            num_samples=30,
        )

        progress_bar.progress(50, text="Getting second opinion...")

        # -----------------------------------------------------------
        # Second opinion (EfficientNet-B0 cross-check)
        # -----------------------------------------------------------

        resnet_model = load_resnet_model()

        with torch.no_grad():
            resnet_logits = resnet_model(input_tensor)
            resnet_probs = torch.softmax(resnet_logits, dim=1).squeeze(0).numpy()

        resnet_pred_idx = int(resnet_probs.argmax())
        resnet_confidence_pct = float(resnet_probs[resnet_pred_idx]) * 100
        models_agree = (resnet_pred_idx == predicted_idx)

        progress_bar.progress(70, text="Estimating uncertainty...")

        is_uncertain = confidence_flag(
            mean_probs,
            uncertainty,
        )

        predicted_class = CLASS_NAMES[predicted_idx]
        resnet_pred_class = CLASS_NAMES[resnet_pred_idx]

        DISPLAY_NAME_OVERRIDES = {
            "amd": "AMD",
        }

        def display_name(class_key):
            if class_key in DISPLAY_NAME_OVERRIDES:
                return DISPLAY_NAME_OVERRIDES[class_key]
            return class_key.replace("_", " ").title()

        predicted_class_display = display_name(predicted_class)
        resnet_pred_class_display = display_name(resnet_pred_class)

        confidence_pct = (
            float(mean_probs[predicted_idx]) * 100
        )

        progress_bar.progress(90, text="Computing Grad-CAM...")

        display_img = preprocess_for_display(image)

        heatmap = generate_gradcam(
            model,
            MODEL_KEY,
            input_tensor,
            display_img,
            target_class=predicted_idx,
        )

        progress_bar.progress(100, text="Analysis complete.")

        progress_bar.empty()

        # -----------------------------------------------------------
        # AI analysis result
        # -----------------------------------------------------------

        st.markdown(
            '<div class="section-title">🧠 AI Analysis Result</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="result-card accent-amber">
                    <div class="result-label">
                        Detected Condition
                    </div>
                    <div class="result-value">
                        🔍 {predicted_class_display}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="result-card accent-amber">
                    <div class="result-label">
                        Model Confidence
                    </div>
                    {confidence_ring_html(confidence_pct, COLOR_AMBER)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:

            if is_uncertain:
                status_label = "Needs Review"
                status_icon = "⚠️"
                badge_bg = "#2A2010"
                badge_border = "#F5A623"
                badge_text = "#F5C77E"
                card_accent_class = "accent-warn"
            else:
                status_label = "High Confidence"
                status_icon = "✅"
                badge_bg = "#0F2A20"
                badge_border = "#3ED598"
                badge_text = "#8FEFC7"
                card_accent_class = "accent-success"

            st.markdown(
                f"""
                <div class="result-card {card_accent_class}">
                    <div class="result-label">
                        Prediction Status
                    </div>
                    <div style="
                        display: inline-block;
                        margin-top: 4px;
                        padding: 6px 16px;
                        border-radius: 999px;
                        background-color: {badge_bg};
                        border: 1px solid {badge_border};
                        color: {badge_text};
                        font-size: 15px;
                        font-weight: 700;
                        font-family: 'Inter', sans-serif;
                    ">
                        {status_icon} {status_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -----------------------------------------------------------
        # Uncertainty warning
        # -----------------------------------------------------------

        if is_uncertain:
            st.warning(
                "⚠️ The model has relatively low confidence for this "
                "image. Consider professional eye evaluation."
            )

        # -----------------------------------------------------------
        # Second opinion (EfficientNet-B0 cross-check) card
        # -----------------------------------------------------------

        st.markdown(
            '<div class="section-title">🔎 Second Opinion — Cross-Model Check</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Every prediction is independently cross-checked by a second, "
            "separately-trained architecture (EfficientNet-B0). On our test set, "
            "the two models agree 96.4% of the time, with 98.0% accuracy on "
            "those agreed cases. When they disagree (3.6% of cases), ResNet50 "
            "(primary) is right 64.7% of the time vs. EfficientNet-B0 (second "
            "opinion) at 29.4% — disagreement is a meaningful signal, not noise."
        )

        so_col1, so_col2, so_col3 = st.columns(3)

        with so_col1:
            st.markdown(
                f"""
                <div class="result-card accent-amber">
                    <div class="result-label">
                        Primary Model (ResNet50)
                    </div>
                    <div class="result-value" style="font-size:20px; margin-bottom:6px;">
                        🧠 {predicted_class_display}
                    </div>
                    {confidence_ring_html(confidence_pct, COLOR_AMBER)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with so_col2:
            st.markdown(
                f"""
                <div class="result-card accent-teal">
                    <div class="result-label">
                        Second Opinion (EfficientNet-B0)
                    </div>
                    <div class="result-value" style="font-size:20px; margin-bottom:6px;">
                        🔁 {resnet_pred_class_display}
                    </div>
                    {confidence_ring_html(resnet_confidence_pct, COLOR_TEAL)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with so_col3:

            if models_agree:
                agree_label = "Models Agree"
                agree_icon = "✅"
                agree_bg = "#0F2A20"
                agree_border = "#3ED598"
                agree_text = "#8FEFC7"
                card_accent_class = "accent-success"
            else:
                agree_label = "Models Disagree"
                agree_icon = "⚠️"
                agree_bg = "#2A2010"
                agree_border = "#F5A623"
                agree_text = "#F5C77E"
                card_accent_class = "accent-warn"

            st.markdown(
                f"""
                <div class="result-card {card_accent_class}">
                    <div class="result-label">
                        Agreement Status
                    </div>
                    <div style="
                        display: inline-block;
                        margin-top: 4px;
                        padding: 6px 16px;
                        border-radius: 999px;
                        background-color: {agree_bg};
                        border: 1px solid {agree_border};
                        color: {agree_text};
                        font-size: 15px;
                        font-weight: 700;
                        font-family: 'Inter', sans-serif;
                    ">
                        {agree_icon} {agree_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if not models_agree:
            st.warning(
                "⚠️ The two independently-trained models disagree on this "
                "image. Historically, cases like this have a much higher "
                "error rate than cases where models agree. Manual / "
                "professional review is strongly recommended."
            )

        # -----------------------------------------------------------
        # Probability distribution
        # -----------------------------------------------------------

        st.markdown(
            '<div class="section-title">📊 Class Probability Distribution</div>',
            unsafe_allow_html=True,
        )

        prob_cols = st.columns(len(CLASS_NAMES))

        for col, name, prob in zip(
            prob_cols,
            CLASS_NAMES,
            mean_probs,
        ):

            with col:

                clean_name = display_name(name)

                probability = float(prob) * 100

                is_predicted = (name == predicted_class)

                value_color = "#F4F8FB" if is_predicted else "#7C8CA0"
                bar_color = "#E1793F" if is_predicted else "#2A3B4D"
                label_weight = "700" if is_predicted else "500"
                accent_class = "accent-amber" if is_predicted else ""

                st.markdown(
                    f"""
                    <div class="result-card {accent_class}">
                        <div class="result-label" style="font-weight:{label_weight};">
                            {clean_name}
                        </div>
                        <div class="result-value" style="color:{value_color}; font-family:'IBM Plex Mono', monospace;">
                            {probability:.1f}%
                        </div>
                        <div style="
                            width: 100%;
                            height: 8px;
                            border-radius: 999px;
                            background-color: #1D2937;
                            margin-top: 10px;
                            overflow: hidden;
                        ">
                            <div style="
                                width: {probability:.1f}%;
                                height: 100%;
                                border-radius: 999px;
                                background-color: {bar_color};
                            "></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # -----------------------------------------------------------
        # Grad-CAM
        # -----------------------------------------------------------

        st.markdown(
            '<div class="section-title">🔬 Explainable AI — Grad-CAM</div>',
            unsafe_allow_html=True,
        )

        gradcam_col1, gradcam_col2 = st.columns(2)

        with gradcam_col1:

            st.image(
                image,
                caption="Original Image",
                use_container_width=True,
            )

        with gradcam_col2:

            st.image(
                heatmap,
                caption="Grad-CAM — Model Attention",
                use_container_width=True,
            )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">🎨 How to Read This Heatmap</div>
                <div class="info-text">
                    <span style="color:#E1793F; font-weight:700;">Red / Orange</span>
                    regions had the strongest influence on the model's decision.
                    <span style="color:#2FBFB0; font-weight:700;">Blue / Dark</span>
                    regions had minimal influence. This visualization helps verify
                    the model is focusing on clinically relevant areas of the eye
                    rather than irrelevant background details.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "💡 Grad-CAM highlights image regions that contributed "
            "to the model's prediction. It helps make the model's "
            "decision more interpretable."
        )

        # -----------------------------------------------------------
        # Recommendations / tips
        # -----------------------------------------------------------

        st.markdown(
            '<div class="section-title">💡 Screening Guidance & Tips</div>',
            unsafe_allow_html=True,
        )

        if is_uncertain:

            st.warning(
                "The model is uncertain about this prediction. "
                "The image should not be interpreted as a reliable "
                "diagnosis. Professional evaluation is recommended "
                "if you have eye-related symptoms or concerns."
            )

        elif not models_agree:

            st.warning(
                "The primary and second-opinion models disagree on this "
                "image. This result should not be interpreted as a "
                "reliable diagnosis. Professional evaluation is "
                "recommended if you have eye-related symptoms or concerns."
            )

        elif predicted_class.lower() in [
            "normal",
            "healthy",
            "no disease",
        ]:

            st.success(
                "The model did not detect strong features associated "
                "with the target anomalies. This does not rule out "
                "eye disease. Routine professional eye examinations "
                "are still important."
            )

        else:

            st.info(
                f"The model detected features associated with "
                f"{predicted_class_display}. This result is an "
                "AI-assisted screening output, not a medical diagnosis. "
                "Consider professional eye evaluation if you have "
                "symptoms or concerns."
            )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">🛡️ Responsible AI Notice</div>
                <div class="info-text">
                    AI predictions can be affected by image quality,
                    dataset limitations, and model uncertainty.
                    Always interpret this demonstration together with
                    appropriate professional medical assessment.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -----------------------------------------------------------
        # PDF report
        # -----------------------------------------------------------

        temp_path = os.path.join(
            PROJECT_ROOT,
            "uploaded_temp.jpg",
        )

        image.save(temp_path)

        pdf_path = generate_pdf_report(
            predicted_class_display,
            confidence_pct,
            is_uncertain,
            temp_path,
            second_opinion_class=resnet_pred_class_display,
            second_opinion_confidence=resnet_confidence_pct,
            models_agree=models_agree,
        )

        st.markdown(
            '<div class="section-title">📄 Analysis Report</div>',
            unsafe_allow_html=True,
        )

        with open(pdf_path, "rb") as report_file:

            st.download_button(
                "📄 Download PDF Analysis Report",
                report_file,
                file_name="eye_analysis_report.pdf",
                mime="application/pdf",
                key="download_eye_analysis_report",
                use_container_width=True,
            )

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-family:'Fraunces', serif; font-size:14px; font-weight:600; color:#F4F8FB;">
            Eye Anomaly Detection — Academic Research Project
        </div>
        <div style="font-size:13px; color:#7C8CA0; margin-top:4px;">
            Built with PyTorch · EfficientNet-B0 · ResNet50 · Streamlit · Grad-CAM · Monte Carlo Dropout
        </div>
        <div style="font-size:12px; color:#5A6B7D; margin-top:8px;">
            For academic demonstration only — not a certified medical diagnostic tool.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)