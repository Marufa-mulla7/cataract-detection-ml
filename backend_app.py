"""
Eye Anomaly Detection - backend server (updated).

Serves the ocuvision-concept.html frontend AND exposes:
  - /predict          : ResNet50 (primary, MC Dropout + Grad-CAM) plus
                        EfficientNet-B0 second opinion on the uploaded image.
  - /generate_report  : builds a downloadable PDF screening report.

MODEL ROLES:
ResNet50 is the PRIMARY model. EfficientNet-B0 is the SECOND-OPINION model.
Both were trained on the final leakage-aware 4,600-image dataset.

Run from the project root with:
    python backend_app.py
Then open http://127.0.0.1:5000 in your browser.

Requires: pip install reportlab
"""

import os
import re
import sys
import io
import json
import base64
import threading
import sqlite3
import time
import traceback
from datetime import datetime
from xml.sax.saxutils import escape

import torch
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_from_directory, send_file

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable,
)

# -------------------------------------------------------------------
# Project paths and settings
# -------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

PRIMARY_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_resnet50_5class_final.pt")
PRIMARY_MODEL_KEY = "resnet50"
PRIMARY_MODEL_LABEL = "ResNet50"

SECOND_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_efficientnet_b0_5class_final.pt")
SECOND_MODEL_KEY = "efficientnet_b0"
SECOND_MODEL_LABEL = "EfficientNet-B0"

HTML_FILE = "ocuvision-concept.html"  # must sit in the same folder as this script

MC_SAMPLES = 30
LOW_CONFIDENCE_PCT = 70.0        # (kept for reference)
MC_STD_REVIEW = 0.02             # review flag: MC Dropout std of the predicted class (chosen on the validation set)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# -------------------------------------------------------------------
# Project imports
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
# Load both models once, at server startup
# -------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[startup] Using device: {DEVICE}")
print(f"[startup] Class order used by the app: {list(CLASS_NAMES)}")


def load_model(key, path, label):
    print(f"[startup] Loading {label}...")
    model = build_model(key, num_classes=len(CLASS_NAMES), freeze_backbone=False)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


primary_model = load_model(PRIMARY_MODEL_KEY, PRIMARY_MODEL_PATH, PRIMARY_MODEL_LABEL)
second_model = load_model(SECOND_MODEL_KEY, SECOND_MODEL_PATH, SECOND_MODEL_LABEL)

# MC Dropout only works if the primary model actually contains Dropout layers.
_dropouts = [(n, m.p) for n, m in primary_model.named_modules()
             if isinstance(m, torch.nn.Dropout)]
if _dropouts:
    print(f"[startup] MC Dropout OK: {PRIMARY_MODEL_LABEL} has {len(_dropouts)} Dropout layer(s): {_dropouts}")
else:
    print(f"[startup] WARNING: {PRIMARY_MODEL_LABEL} has NO Dropout layers - "
          f"MC Dropout uncertainty will not vary between passes.")

print("[startup] Both models loaded. Ready.")

# -------------------------------------------------------------------
# Flask app
# -------------------------------------------------------------------

app = Flask(__name__)
MAX_BATCH = 30
app.config["MAX_CONTENT_LENGTH"] = MAX_BATCH * MAX_UPLOAD_BYTES  # per-file limit is checked below

# The models and Grad-CAM hooks are shared global objects, so handle one
# prediction at a time.
_inference_lock = threading.Lock()

# ---- Scan log (SQLite). Stores results only - uploaded images are never saved. ----
DB_PATH = os.path.join(PROJECT_ROOT, "scans.db")
with sqlite3.connect(DB_PATH) as _c:
    _c.execute("CREATE TABLE IF NOT EXISTS scans(id INTEGER PRIMARY KEY, ts TEXT, scan_id TEXT, "
               "predicted TEXT, confidence REAL, second TEXT, agree INTEGER, needs_review INTEGER, ms INTEGER)")


def check_fundus(image):
    """Heuristic only: fundus photos are red-dominant with a dark border. Used as a soft warning."""
    a = np.asarray(image.resize((96, 96)), dtype=np.float32)
    yy, xx = np.mgrid[:96, :96]
    d = np.hypot(yy - 48, xx - 48)
    inner, outer = a[d < 30], a[d > 50]
    ok = bool(inner[:, 0].mean() > 1.3 * inner[:, 2].mean() and outer.mean() < 0.7 * inner.mean())
    msg = "" if ok else ("This image does not look like a typical fundus (retinal) photograph, "
                         "so the result may be unreliable.")
    return {"looks_like_fundus": ok, "message": msg}


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Upload is too large (maximum 10 MB per image, 30 images per batch)."}), 413


def numpy_image_to_base64(np_img_uint8):
    """Grad-CAM returns a uint8 RGB numpy array - encode it as a PNG data URL."""
    pil_img = Image.fromarray(np_img_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def open_image(file_stream):
    """Open an uploaded image as RGB, fixing phone-photo rotation."""
    image = Image.open(file_stream)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def build_recommendation(predicted_class, needs_review, models_agree, second_class=""):
    """Server-side recommendation text (not taken from the browser)."""
    if not models_agree and second_class:
        text = (f"The two AI models disagree: the primary model suggested {predicted_class}, and the second "
                f"model suggested {second_class}. This image should be reviewed by a qualified eye-care "
                f"professional before any conclusion is drawn.")
    elif predicted_class.lower() == "normal":
        text = ("The AI did not flag an anomaly in this image. This does not rule out eye disease; "
                "please consult an eye-care professional if you have symptoms or are due for a "
                "routine eye examination.")
    else:
        text = (f"The AI found features consistent with {predicted_class}. Please have this "
                f"confirmed by a qualified eye-care professional; this result is not a diagnosis.")
    if needs_review:
        text += " This result was flagged for review because of model disagreement or high uncertainty."
    text += " Results are only valid for retinal or eye images of the type used in training."
    return text


@app.route("/")
def index():
    return send_from_directory(PROJECT_ROOT, HTML_FILE)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        upload = request.files["image"]
        upload.stream.seek(0, 2)
        if upload.stream.tell() > MAX_UPLOAD_BYTES:
            return jsonify({"error": "Image is too large (maximum 10 MB)."}), 413
        upload.stream.seek(0)
        image = open_image(upload.stream)
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    try:
        t0 = time.time()
        with _inference_lock:
            transform = get_transforms(train=False)
            input_tensor = transform(image).unsqueeze(0)

            # --- Primary model: ResNet50 with MC Dropout uncertainty ---
            mean_probs, uncertainty, predicted_idx = mc_dropout_predict(
                primary_model, input_tensor, num_samples=MC_SAMPLES, device=DEVICE
            )
            is_uncertain = bool(confidence_flag(mean_probs, uncertainty))
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence_pct = float(mean_probs[predicted_idx]) * 100

            # --- Second opinion: EfficientNet-B0, plain forward pass ---
            second_model.eval()
            with torch.no_grad():
                second_logits = second_model(input_tensor.to(DEVICE))
                second_probs = torch.softmax(second_logits, dim=1).squeeze(0).cpu().numpy()

            second_pred_idx = int(second_probs.argmax())
            second_pred_class = CLASS_NAMES[second_pred_idx]
            second_confidence_pct = float(second_probs[second_pred_idx]) * 100
            models_agree = bool(second_pred_idx == predicted_idx)

            # --- Grad-CAM (primary model, predicted class) ---
            display_img = preprocess_for_display(image)
            heatmap = generate_gradcam(
                primary_model, PRIMARY_MODEL_KEY, input_tensor, display_img,
                target_class=predicted_idx, device=DEVICE,
            )
            primary_model.eval()  # make sure nothing leaves the model in train mode
            w, h = image.size
            k = min(1.0, 800 / max(w, h))
            heatmap_img = Image.fromarray(heatmap).resize((max(1, int(w * k)), max(1, int(h * k))))
            heatmap_data_url = numpy_image_to_base64(np.asarray(heatmap_img))

        probabilities = {
            display_name(name): round(float(p) * 100, 1)
            for name, p in zip(CLASS_NAMES, mean_probs)
        }
        second_probabilities = {
            display_name(name): round(float(p) * 100, 1)
            for name, p in zip(CLASS_NAMES, second_probs)
        }

        # Review flag (threshold chosen on the validation set): models disagree OR MC Dropout std > 0.02.
        needs_review = bool(not models_agree or float(uncertainty[predicted_idx]) > MC_STD_REVIEW)

        elapsed_ms = int((time.time() - t0) * 1000)
        image_check = check_fundus(image)
        scan_id = "SCN-" + str(int(time.time() * 1000))[-8:]
        with sqlite3.connect(DB_PATH) as c:
            c.execute("INSERT INTO scans(ts, scan_id, predicted, confidence, second, agree, needs_review, ms) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), scan_id, display_name(predicted_class),
                       round(confidence_pct, 1), display_name(second_pred_class), int(models_agree),
                       int(needs_review), elapsed_ms))

        response = {
            "predicted_class": display_name(predicted_class),
            "confidence_pct": round(confidence_pct, 1),
            "is_uncertain": is_uncertain,
            "needs_review": needs_review,
            "primary_model_label": PRIMARY_MODEL_LABEL,
            "second_model_label": SECOND_MODEL_LABEL,
            "second_opinion_predicted_class": display_name(second_pred_class),
            "second_opinion_confidence_pct": round(second_confidence_pct, 1),
            "models_agree": models_agree,
            "probabilities": probabilities,
            "second_opinion_probabilities": second_probabilities,
            "heatmap_data_url": heatmap_data_url,
            "image_check": image_check,
            "processing_ms": elapsed_ms,
        }
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {e}"}), 500


# -------------------------------------------------------------------
# Batch analysis (no Grad-CAM, so it stays reasonably fast on CPU)
# -------------------------------------------------------------------

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided"}), 400
    if len(files) > MAX_BATCH:
        return jsonify({"error": f"Please upload at most {MAX_BATCH} images at a time."}), 400

    transform = get_transforms(train=False)
    rows = []
    for f in files:
        row = {"filename": f.filename}
        try:
            f.stream.seek(0, 2)
            size = f.stream.tell()
            f.stream.seek(0)
            if size > MAX_UPLOAD_BYTES:
                raise ValueError("file is larger than 10 MB")
            image = open_image(f.stream)
            t0 = time.time()
            with _inference_lock:
                x = transform(image).unsqueeze(0)
                mean_probs, unc, idx = mc_dropout_predict(
                    primary_model, x, num_samples=MC_SAMPLES, device=DEVICE)
                is_unc = bool(confidence_flag(mean_probs, unc))
                second_model.eval()
                with torch.no_grad():
                    sp = torch.softmax(second_model(x.to(DEVICE)), dim=1).squeeze(0).cpu().numpy()
                primary_model.eval()
            conf = float(mean_probs[idx]) * 100
            s_idx = int(sp.argmax())
            agree = bool(s_idx == idx)
            needs_review = bool(not agree or float(unc[idx]) > MC_STD_REVIEW)
            ms = int((time.time() - t0) * 1000)
            row.update({
                "predicted_class": display_name(CLASS_NAMES[idx]),
                "confidence_pct": round(conf, 1),
                "second_opinion_predicted_class": display_name(CLASS_NAMES[s_idx]),
                "second_opinion_confidence_pct": round(float(sp[s_idx]) * 100, 1),
                "models_agree": agree,
                "needs_review": needs_review,
                "looks_like_fundus": check_fundus(image)["looks_like_fundus"],
            })
            with sqlite3.connect(DB_PATH) as c:
                c.execute("INSERT INTO scans(ts, scan_id, predicted, confidence, second, agree, needs_review, ms) "
                          "VALUES (?,?,?,?,?,?,?,?)",
                          (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "SCN-" + str(int(time.time() * 1000))[-8:],
                           row["predicted_class"], row["confidence_pct"], row["second_opinion_predicted_class"],
                           int(agree), int(needs_review), ms))
        except Exception as e:
            row["error"] = str(e)
        rows.append(row)
    return jsonify({"results": rows})


# -------------------------------------------------------------------
# Live statistics (computed from the scan log, nothing is simulated)
# -------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok", "models": [PRIMARY_MODEL_LABEL, SECOND_MODEL_LABEL], "device": DEVICE})


@app.route("/api/stats")
def api_stats():
    from datetime import timedelta
    with sqlite3.connect(DB_PATH) as c:
        total, flagged, agree, conf, ms = c.execute(
            "SELECT COUNT(*), COALESCE(SUM(needs_review),0), COALESCE(AVG(agree),0), "
            "COALESCE(AVG(confidence),0), COALESCE(AVG(ms),0) FROM scans").fetchone()
        by_class = dict(c.execute("SELECT predicted, COUNT(*) FROM scans GROUP BY predicted").fetchall())
        per_day = dict(c.execute("SELECT substr(ts,1,10), COUNT(*) FROM scans GROUP BY 1").fetchall())
    days = [datetime.now() - timedelta(days=i) for i in range(6, -1, -1)]
    return jsonify({
        "total": total, "flagged": int(flagged), "agree_rate": round(agree * 100, 1),
        "avg_confidence": round(conf, 1), "avg_ms": round(ms), "by_class": by_class,
        "week": [{"day": d.strftime("%a"), "n": per_day.get(d.strftime("%Y-%m-%d"), 0)} for d in days],
    })


@app.route("/api/history")
def api_history():
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute("SELECT scan_id, ts, predicted, confidence, second, agree, needs_review "
                         "FROM scans ORDER BY id DESC LIMIT 8").fetchall()
    keys = ["scan_id", "ts", "predicted", "confidence", "second", "agree", "needs_review"]
    return jsonify([dict(zip(keys, r)) for r in rows])


# -------------------------------------------------------------------
# PDF report generation
# -------------------------------------------------------------------

def _data_url_to_pil(data_url):
    _, encoded = data_url.split(",", 1)
    raw = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _pil_to_rlimage(pil_img, width):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    ratio = pil_img.height / pil_img.width
    return RLImage(buf, width=width, height=width * ratio)


def build_clinical_report_pdf(report_data, original_pil_image):
    """Build the PDF report. Returns a BytesIO positioned at the start."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
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
        textColor=colors.HexColor("#5A6E85"), spaceAfter=14, alignment=1,
    )
    section_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=12,
        textColor=colors.HexColor("#0B1B2B"), spaceBefore=14, spaceAfter=6, keepWithNext=1,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=9.5, leading=14,
        textColor=colors.HexColor("#2A3A4A"),
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#5A6E85"), alignment=1,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=8,
        textColor=colors.HexColor("#7C8CA0"), leading=11,
    )

    story = []

    # --- Header ---
    story.append(Paragraph("Eye Anomaly Detection", title_style))
    story.append(Paragraph("AI-Assisted Eye Anomaly Screening Report", subtitle_style))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#D7DEE6"), thickness=1))
    story.append(Spacer(1, 10))

    # --- Scan info ---
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    info_rows = [
        ["Scan ID", str(report_data.get("scan_id", "N/A"))],
        ["Report Generated", generated_at],
        ["Primary Model", PRIMARY_MODEL_LABEL],
        ["Second-Opinion Model", SECOND_MODEL_LABEL],
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
    predicted_class = str(report_data.get("predicted_class", "N/A"))
    models_agree = bool(report_data.get("models_agree"))
    needs_review = bool(report_data.get("needs_review", report_data.get("is_uncertain")))

    story.append(Paragraph("AI Finding", section_style))
    finding_rows = [
        ["Primary Finding", predicted_class],
        ["AI Confidence", f'{report_data.get("confidence_pct", "N/A")}%'],
        ["Review Flag",
         "Flagged - review recommended" if needs_review else "Not flagged"],
        ["Second Opinion", str(report_data.get("second_opinion_predicted_class", "N/A"))],
        ["Second Opinion Confidence", f'{report_data.get("second_opinion_confidence_pct", "N/A")}%'],
        ["Model Agreement",
         "Models agree" if models_agree else "Models disagree - treat with extra caution"],
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
    story.append(Paragraph("Class Probability Distribution (primary model)", section_style))
    probs = report_data.get("probabilities", {}) or {}
    prob_rows = [["Class", "Probability"]] + [
        [str(name), f"{float(pct)}%"]
        for name, pct in sorted(probs.items(), key=lambda x: -float(x[1]))
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

    # --- Images: original + Grad-CAM heatmap ---
    story.append(Paragraph("Image Analysis", section_style))
    img_w = 2.55 * inch

    heatmap_pil = None
    if report_data.get("heatmap_data_url"):
        try:
            heatmap_pil = _data_url_to_pil(report_data["heatmap_data_url"])
        except Exception:
            heatmap_pil = None

    orig_cell = _pil_to_rlimage(original_pil_image, img_w) if original_pil_image else Paragraph("-", body_style)
    heat_cell = _pil_to_rlimage(heatmap_pil, img_w) if heatmap_pil else Paragraph("-", body_style)

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

    # --- Recommendation (built on the server, not taken from the browser) ---
    story.append(Paragraph("Recommendation", section_style))
    story.append(Paragraph(
        escape(build_recommendation(predicted_class, needs_review, models_agree,
                                   str(report_data.get("second_opinion_predicted_class", "")))),
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
        "This report is generated by an academic AI screening tool (Eye Anomaly Detection) and is not a medical "
        "diagnosis. It has not undergone clinical validation. All findings must be reviewed and "
        "confirmed by a qualified, licensed eye-care professional before any clinical decision is "
        "made. The models were trained on a limited dataset of five classes (Normal, Cataract, "
        "Glaucoma, AMD, Diabetic Retinopathy) and may not generalise to other populations, cameras "
        "or imaging conditions, or to images that are not eye images.",
        disclaimer_style,
    ))

    doc.build(story)
    buf.seek(0)
    return buf


@app.route("/generate_report", methods=["POST"])
def generate_report():
    """Build and return a PDF report for an analysis result.

    Expects multipart/form-data with:
      - original_image: the same image file that was sent to /predict
      - report_data: JSON string of the /predict response (plus a scan_id)
    """
    if "original_image" not in request.files:
        return jsonify({"error": "No original_image file provided"}), 400
    if "report_data" not in request.form:
        return jsonify({"error": "No report_data provided"}), 400

    try:
        report_data = json.loads(request.form["report_data"])
        if not isinstance(report_data, dict):
            raise ValueError("report_data must be a JSON object")
    except Exception as e:
        return jsonify({"error": f"Invalid report_data JSON: {e}"}), 400

    try:
        original_image = open_image(request.files["original_image"].stream)
    except Exception as e:
        return jsonify({"error": f"Could not read original image: {e}"}), 400

    try:
        pdf_buffer = build_clinical_report_pdf(report_data, original_image)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not build PDF report: {e}"}), 500

    scan_id = re.sub(r"[^A-Za-z0-9_-]", "", str(report_data.get("scan_id", "scan")))[:40] or "scan"
    filename = f"eye-anomaly-report-{scan_id}.pdf"

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    import webbrowser
    URL = "http://127.0.0.1:5000"
    # Models are already loaded at this point, so open the browser shortly after the server starts.
    threading.Timer(1.5, lambda: webbrowser.open(URL)).start()
    print(f"[startup] Opening {URL} in your browser...")
    app.run(host="127.0.0.1", port=5000, debug=False)