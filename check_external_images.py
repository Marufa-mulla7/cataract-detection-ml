"""
Run external images (e.g. DR photos from other sources) through the running app and summarise.
Usage (app must be running in another terminal):
    python check_external_images.py external_dr          <- folder with images
"""
import os, sys, requests

URL = "http://127.0.0.1:5000/predict"
folder = sys.argv[1] if len(sys.argv) > 1 else "external_dr"
exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]

rows, flagged = [], 0
for f in files:
    with open(os.path.join(folder, f), "rb") as fh:
        d = requests.post(URL, files={"image": fh}, timeout=300).json()
    review = next((v for k, v in d.items() if "review" in k.lower() and isinstance(v, (bool, int))), None)
    flagged += bool(review)
    rows.append((f, d.get("predicted_class"), d.get("confidence_pct"),
                 d.get("second_opinion_predicted_class"), review))
    print(f"{f[:28]:28s} primary={rows[-1][1]!s:22s} conf={rows[-1][2]}%  second={rows[-1][3]!s:22s} review_flag={review}")

n = len(rows)
if n:
    correct = sum(1 for r in rows if r[1] and "diabetic" in r[1].lower())
    print(f"\nPrimary said DR on {correct}/{n} images")
    print(f"Review flag raised on {flagged}/{n} images")
    wrong = [r for r in rows if not (r[1] and 'diabetic' in r[1].lower())]
    caught = sum(1 for r in wrong if r[4])
    print(f"Of {len(wrong)} wrong predictions, {caught} were flagged for review")