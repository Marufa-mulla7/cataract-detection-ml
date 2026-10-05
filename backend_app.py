"""
OcuVision AI - backend server.

Serves the ocuvision-concept.html frontend AND exposes:
  - a real /predict endpoint backed by your actual trained models - same
    models, same MC Dropout uncertainty, same Grad-CAM, same second-opinion
    logic as the working Streamlit app.
  - a real /generate_report endpoint that builds an actual downloadable PDF
    clinical report (via reportlab) from a real analysis result, embedding
    the original image and the Grad-CAM heatmap side by side.

MODEL ROLES (updated after the data-augmentation bug fix and retrain):
ResNet50 now outperforms EfficientNet-B0 on the real test set (97% vs 96%
accuracy, 0.94 vs 0.92 macro F1), so ResNet50 is now the PRIMARY model and
EfficientNet-B0 is the SECOND-OPINION cross-check - the reverse of the
original (pre-fix) assignment.

Run from the project root with:
    python backend_app.py

Then open http://127.0.0.1:5000 in your browser.

Requires reportlab in addition to the existing project dependencies:
    pip install reportlab
"""

import os
import sys
import io
import json
import base64
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, send_file

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable,
)

# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Primary model: ResNet50 (best test-set performer after the augmentation fix)
PRIMARY_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_resnet50_5class.pt")
PRIMARY_MODEL_KEY = "resnet50"
PRIMARY_MODEL_LABEL = "ResNet50"

# Second opinion: EfficientNet-B0
SECOND_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_5class.pt")
SECOND_MODEL_KEY = "efficientnet_b0"
SECOND_MODEL_LABEL = "EfficientNet-B0"

HTML_FILE = "ocuvision-concept.html"  # must sit in the same folder as this script

# -------------------------------------------------------------------
# Project imports (same code your Streamlit app already uses)
# -------------------------------------------------------------------

from src.models import build_model
from src.data_loader import get_transforms, CLASS_NAMES
from src.gradcam import generate_gradcam, preprocess_for_display
from src.uncertainty import mc_dropout_predict, confidence_flag

DISPLAY_NAME_OVERRIDES = {"amd": "AMD"}


def display_name(class_key):
    if class_key in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[class_key]
    return class_key.replace("_", " ").title()


# -------------------------------------------------------------------
# Load both models once, at server startup (not per-request)
# -------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[startup] Using device: {DEVICE}")

print(f"[startup] Loading primary model: {PRIMARY_MODEL_LABEL}...")
primary_model = build_model(PRIMARY_MODEL_KEY, num_classes=len(CLASS_NAMES), freeze_backbone=False)
primary_model.load_state_dict(torch.load(PRIMARY_MODEL_PATH, map_location=DEVICE))
primary_model.eval()
primary_model = primary_model.to(DEVICE)

print(f"[startup] Loading second-opinion model: {SECOND_MODEL_LABEL}...")
second_model = build_model(SECOND_MODEL_KEY, num_classes=len(CLASS_NAMES), freeze_backbone=False)
second_model.load_state_dict(torch.load(SECOND_MODEL_PATH, map_location=DEVICE))
second_model.eval()
second_model = second_model.to(DEVICE)

print("[startup] Both models loaded. Ready.")

# -------------------------------------------------------------------
# Flask app
# -------------------------------------------------------------------

app = Flask(__name__)


def numpy_image_to_base64(np_img_uint8):
    """Grad-CAM returns a uint8 RGB numpy array - encode it as a PNG data URL."""
    pil_img = Image.fromarray(np_img_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.route("/")
def index():
    return send_from_directory(PROJECT_ROOT, HTML_FILE)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    transform = get_transforms(train=False)
    input_tensor = transform(image).unsqueeze(0)

    # --- Primary model: ResNet50, with MC Dropout uncertainty ---
    mean_probs, uncertainty, predicted_idx = mc_dropout_predict(
        primary_model, input_tensor, num_samples=30, device=DEVICE
    )
    is_uncertain = bool(confidence_flag(mean_probs, uncertainty))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence_pct = float(mean_probs[predicted_idx]) * 100

    # --- Second opinion: EfficientNet-B0, plain forward pass ---
    with torch.no_grad():
        second_input = input_tensor.to(DEVICE)
        second_logits = second_model(second_input)
        second_probs = torch.softmax(second_logits, dim=1).squeeze(0).cpu().numpy()

    second_pred_idx = int(second_probs.argmax())
    second_pred_class = CLASS_NAMES[second_pred_idx]
    second_confidence_pct = float(second_probs[second_pred_idx]) * 100
    models_agree = bool(second_pred_idx == predicted_idx)

    # --- Grad-CAM (for the primary model's predicted class) ---
    display_img = preprocess_for_display(image)
    heatmap = generate_gradcam(
        primary_model, PRIMARY_MODEL_KEY, input_tensor, display_img,
        target_class=predicted_idx, device=DEVICE,
    )
    heatmap_data_url = numpy_image_to_base64(heatmap)

    probabilities = {
        display_name(name): round(float(p) * 100, 1)
        for name, p in zip(CLASS_NAMES, mean_probs)
    }

    response = {
        "predicted_class": display_name(predicted_class),
        "confidence_pct": round(confidence_pct, 1),
        "is_uncertain": is_uncertain,
        "primary_model_label": PRIMARY_MODEL_LABEL,
        "second_model_label": SECOND_MODEL_LABEL,
        "second_opinion_predicted_class": display_name(second_pred_class),
        "second_opinion_confidence_pct": round(second_confidence_pct, 1),
        "models_agree": models_agree,
        "probabilities": probabilities,
        "heatmap_data_url": heatmap_data_url,
    }
    return jsonify(response)


# -------------------------------------------------------------------
# PDF report generation
# -------------------------------------------------------------------

def _data_url_to_pil(data_url):
    """Decode a data:image/...;base64,... URL (as returned by /predict) into a PIL Image."""
    _, encoded = data_url.split(",", 1)
    raw = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _pil_to_rlimage(pil_img, width):
    """Convert a PIL image into a reportlab Image flowable at a fixed width,
    preserving aspect ratio."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    ratio = pil_img.height / pil_img.width
    return RLImage(buf, width=width, height=width * ratio)


def build_clinical_report_pdf(report_data, original_pil_image):
    """Build a clinical-style PDF report from a real /predict result and the
    original uploaded image. Returns a BytesIO positioned at the start."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=18,
        textColor=colors.HexColor("#0B1B2B"), spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#5A6E85"), spaceAfter=14,
    )
    section_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=12,
        textColor=colors.HexColor("#0B1B2B"), spaceBefore=14, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=9.5, leading=14,
        textColor=colors.HexColor("#2A3A4A"),
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#5A6E85"), alignment=1,  # center
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=8,
        textColor=colors.HexColor("#7C8CA0"), leading=11,
    )

    story = []

    # --- Header ---
    story.append(Paragraph("Eye Anomaly Detection", title_style))
    story.append(Paragraph("AI-Assisted Ocular Screening Report", subtitle_style))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#D7DEE6"), thickness=1))
    story.append(Spacer(1, 10))

    # --- Scan info ---
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    info_rows = [
        ["Scan ID", report_data.get("scan_id", "N/A")],
        ["Report Generated", generated_at],
        ["Primary Model", report_data.get("primary_model_label", "ResNet50")],
        ["Second-Opinion Model", report_data.get("second_model_label", "EfficientNet-B0")],
    ]
    info_table = Table(info_rows, colWidths=[1.9 * inch, 4.1 * inch])
    info_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#5A6E85")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#0B1B2B")),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#EEF2F6")),
    ]))
    story.append(info_table)

    # --- AI finding ---
    story.append(Paragraph("AI Finding", section_style))
    finding_rows = [
        ["Primary Finding", report_data.get("predicted_class", "N/A")],
        ["AI Confidence", f'{report_data.get("confidence_pct", "N/A")}%'],
        ["Uncertainty Flag",
         "Low confidence — review recommended" if report_data.get("is_uncertain")
         else "Within normal confidence range"],
        ["Second Opinion", report_data.get("second_opinion_predicted_class", "N/A")],
        ["Second Opinion Confidence", f'{report_data.get("second_opinion_confidence_pct", "N/A")}%'],
        ["Model Agreement",
         "Models agree" if report_data.get("models_agree")
         else "Models disagree — treat with extra caution"],
    ]
    finding_table = Table(finding_rows, colWidths=[2.3 * inch, 3.7 * inch])
    finding_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#5A6E85")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#0B1B2B")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#EEF2F6")),
    ]))
    story.append(finding_table)

    # --- Class probability distribution ---
    story.append(Paragraph("Class Probability Distribution", section_style))
    probs = report_data.get("probabilities", {}) or {}
    prob_rows = [["Class", "Probability"]] + [
        [name, f"{pct}%"] for name, pct in sorted(probs.items(), key=lambda x: -x[1])
    ]
    prob_table = Table(prob_rows, colWidths=[3.6 * inch, 2.4 * inch])
    prob_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F6")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#EEF2F6")),
    ]))
    story.append(prob_table)

    # --- Images: original + Grad-CAM heatmap, side by side ---
    story.append(Paragraph("Image Analysis", section_style))
    img_w = 2.55 * inch

    heatmap_pil = None
    if report_data.get("heatmap_data_url"):
        try:
            heatmap_pil = _data_url_to_pil(report_data["heatmap_data_url"])
        except Exception:
            heatmap_pil = None

    orig_cell = _pil_to_rlimage(original_pil_image, img_w) if original_pil_image else Paragraph("—", body_style)
    heat_cell = _pil_to_rlimage(heatmap_pil, img_w) if heatmap_pil else Paragraph("—", body_style)

    img_table = Table(
        [
            [Paragraph("Original", caption_style), Paragraph("Grad-CAM Heatmap", caption_style)],
            [orig_cell, heat_cell],
        ],
        colWidths=[img_w + 0.3 * inch, img_w + 0.3 * inch],
    )
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 1), (-1, 1), 4),
    ]))
    story.append(img_table)

    # --- Recommendation ---
    story.append(Paragraph("Recommendation", section_style))
    story.append(Paragraph(
        report_data.get(
            "recommendation",
            "Clinical review by a qualified eye-care professional is advised.",
        ),
        body_style,
    ))

    # --- Signature line ---
    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#D7DEE6"), thickness=1))
    story.append(Spacer(1, 8))
    sig_table = Table(
        [["Reviewed by (name & signature): ______________________________", "Date: ____________"]],
        colWidths=[4.3 * inch, 1.7 * inch],
    )
    sig_table.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9.5)]))
    story.append(sig_table)

    # --- Disclaimer ---
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "This report is generated by an AI-assisted screening tool and is not a definitive medical "
        "diagnosis. All findings must be reviewed and confirmed by a qualified, licensed eye-care "
        "professional before any clinical decision is made. Model performance figures are based on "
        "internal test-set evaluation and may not generalize to all populations or imaging conditions.",
        disclaimer_style,
    ))

    doc.build(story)
    buf.seek(0)
    return buf


@app.route("/generate_report", methods=["POST"])
def generate_report():
    """Build and return a real PDF clinical report for a real analysis result.

    Expects multipart/form-data with:
      - original_image: the same image file that was sent to /predict
      - report_data: JSON string of the /predict response (plus a scan_id
        added client-side)
    """
    if "original_image" not in request.files:
        return jsonify({"error": "No original_image file provided"}), 400
    if "report_data" not in request.form:
        return jsonify({"error": "No report_data provided"}), 400

    try:
        report_data = json.loads(request.form["report_data"])
    except Exception as e:
        return jsonify({"error": f"Invalid report_data JSON: {e}"}), 400

    file = request.files["original_image"]
    try:
        original_image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read original image: {e}"}), 400

    try:
        pdf_buffer = build_clinical_report_pdf(report_data, original_image)
    except Exception as e:
        return jsonify({"error": f"Could not build PDF report: {e}"}), 500

    scan_id = report_data.get("scan_id", "scan")
    filename = f"eye-anomaly-report-{scan_id}.pdf"

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)