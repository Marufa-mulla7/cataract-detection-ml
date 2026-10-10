"""
Step 1 of the finish line: does the RUNNING app classify known internal test images correctly?

1) Start the app:  python backend_app.py
2) Find endpoint:  findstr /n "route" backend_app.py     (Windows)
3) Edit the 3 CONFIG lines below if needed, then:  python verify_app_on_test_images.py
"""
import os, random, json, requests
from collections import defaultdict

# ---------------- CONFIG ----------------
URL        = "http://127.0.0.1:5000/predict"
FILE_FIELD = "image"
TEST_DIR   = r"data/cataract_final/test"
N_PER_CLASS = 6
# ----------------------------------------

def find_label(obj):
    """Best-effort: find the predicted class string in the JSON response."""
    if isinstance(obj, dict):
        for k in ("predicted_class", "prediction", "predicted_class", "label", "class", "primary_prediction", "result"):
            if k in obj and isinstance(obj[k], str):
                return obj[k]
        for v in obj.values():
            r = find_label(v)
            if r: return r
    return None

def norm(s): return "".join(c for c in s.lower() if c.isalnum())

random.seed(42)
stats = defaultdict(lambda: [0, 0])
for cls in sorted(os.listdir(TEST_DIR)):
    folder = os.path.join(TEST_DIR, cls)
    if not os.path.isdir(folder): continue
    files = random.sample(os.listdir(folder), min(N_PER_CLASS, len(os.listdir(folder))))
    for f in files:
        with open(os.path.join(folder, f), "rb") as fh:
            r = requests.post(URL, files={FILE_FIELD: fh}, timeout=300)
        try:
            data = r.json()
        except Exception:
            print("Non-JSON response:", r.status_code, r.text[:200]); raise SystemExit
        pred = find_label(data) or "?"
        ok = norm(cls) in norm(pred) or norm(pred) in norm(cls)
        stats[cls][0] += ok; stats[cls][1] += 1
        if not ok: print(f"MISS  true={cls:22s} app said={pred}  file={f}")

print("\n=== App accuracy on known internal test images ===")
for c, (ok, n) in stats.items(): print(f"{c:25s} {ok}/{n}")
tot_ok = sum(v[0] for v in stats.values()); tot = sum(v[1] for v in stats.values())
print(f"TOTAL {tot_ok}/{tot}")
print("\nIf ~all correct -> external DR failure = domain shift (document as limitation).")
print("If many wrong   -> backend bug: check class order, transform, checkpoint path (see FINISH_PLAN.md).")
if stats and tot_ok == 0 and tot > 0: print("First response JSON for debugging:\n", json.dumps(data, indent=2)[:800])