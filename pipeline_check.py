"""
pipeline_check.py - sanity-check the OcuVision backend using the real test set.

It sends images through the actual /predict endpoint (Flask test client), so
it exercises the same preprocessing, MC Dropout, second opinion and Grad-CAM
code the web app uses. It does NOT start the web server.

Run from the project root (same folder as backend_app.py):

    python pipeline_check.py 20                 # 20 random test images per class
    python pipeline_check.py all                # whole test set (slow on CPU)
    python pipeline_check.py 20 EXT_DIR LABEL   # also test a folder of external
                                                # images whose true label is LABEL,
                                                # e.g. diabetic_retinopathy

What to look for in the output:
  1. CLASS_NAMES order matches the dataset folder order.
  2. 'modules left in train mode' is empty after requests (else BatchNorm bug).
  3. BatchNorm checksum is unchanged after requests (else the model is drifting).
  4. Per-class accuracy is close to the final test results (about 95% overall).
"""
import os
import sys
import random
import collections

import torch

ARG_N = sys.argv[1] if len(sys.argv) > 1 else "20"
N = None if ARG_N.lower() == "all" else int(ARG_N)
TEST_DIR = os.path.join("data", "cataract_final", "test")
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# If your folder names differ from the labels the backend returns, map them here.
FOLDER_TO_LABEL = {}  # e.g. {"dr": "Diabetic Retinopathy"}

import backend_app as be  # loads both models; the server is not started

client = be.app.test_client()


def norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def train_mode_modules(model):
    return dict(collections.Counter(type(m).__name__ for m in model.modules() if m.training))


def bn_checksum(model):
    return round(sum(float(m.running_mean.sum()) for m in model.modules()
                     if isinstance(m, torch.nn.BatchNorm2d)), 6)


def post(path, name):
    with open(path, "rb") as fh:
        return client.post("/predict", data={"image": (fh, name)},
                           content_type="multipart/form-data")


def run_folder(folder, true_label, n):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))
    if n is not None and len(files) > n:
        files = random.sample(files, n)
    ok = agree = uncertain = done = 0
    wrong = []
    for f in files:
        r = post(os.path.join(folder, f), f)
        if r.status_code != 200:
            print(f"   HTTP {r.status_code} for {f}: {r.get_data(as_text=True)[:120]}")
            continue
        j = r.get_json()
        done += 1
        agree += int(j["models_agree"])
        uncertain += int(j["is_uncertain"])
        if norm(j["predicted_class"]) == norm(true_label):
            ok += 1
        else:
            wrong.append((f, j["predicted_class"], j["confidence_pct"],
                          j["second_opinion_predicted_class"]))
    acc = 100 * ok / done if done else 0
    print(f"{true_label:<22} {ok:>4}/{done:<4} correct ({acc:5.1f}%) | "
          f"models agree {agree}/{done} | flagged uncertain {uncertain}/{done}")
    for w in wrong[:5]:
        print(f"      wrong: {w[0]} -> primary {w[1]} ({w[2]}%), second opinion {w[3]}")
    return ok, done


random.seed(0)

# ---- 1. class order -------------------------------------------------------
folders = sorted(d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d)))
print("Dataset folders (alphabetical = ImageFolder order):", folders)
print("Backend CLASS_NAMES                               :", list(be.CLASS_NAMES))
if [norm(c) for c in folders] != [norm(c) for c in be.CLASS_NAMES]:
    print("WARNING: class order/names differ - check src/data_loader.py against training.")

# ---- 2. model state before ------------------------------------------------
before = (bn_checksum(be.primary_model), bn_checksum(be.second_model))
print("\nBatchNorm checksum before:", before)
print("Modules in train mode before:", train_mode_modules(be.primary_model),
      train_mode_modules(be.second_model))

# ---- 3. accuracy through the real endpoint --------------------------------
print("\nPer-class results through /predict:")
tot_ok = tot = 0
for c in folders:
    label = FOLDER_TO_LABEL.get(c, be.display_name(c.lower()))
    o, d = run_folder(os.path.join(TEST_DIR, c), label, N)
    tot_ok += o
    tot += d
print(f"{'OVERALL':<22} {tot_ok:>4}/{tot:<4} correct ({100 * tot_ok / max(tot, 1):5.1f}%)")

# ---- 4. optional external folder ------------------------------------------
if len(sys.argv) >= 4:
    print(f"\nExternal images in {sys.argv[2]} (true label: {sys.argv[3]}):")
    run_folder(sys.argv[2], FOLDER_TO_LABEL.get(sys.argv[3], be.display_name(sys.argv[3].lower())), None)

# ---- 5. model state after -------------------------------------------------
after = (bn_checksum(be.primary_model), bn_checksum(be.second_model))
print("\nBatchNorm checksum after :", after,
      "OK (unchanged)" if before == after else "<-- CHANGED: BatchNorm stats are being updated at inference!")
print("Modules left in train mode after requests (should be empty):",
      train_mode_modules(be.primary_model), train_mode_modules(be.second_model))

# ---- 6. repeatability on one image ----------------------------------------
first = os.path.join(TEST_DIR, folders[0])
img = sorted(f for f in os.listdir(first) if f.lower().endswith(EXTS))[0]
confs = []
preds = set()
for _ in range(5):
    j = post(os.path.join(first, img), img).get_json()
    preds.add(j["predicted_class"])
    confs.append(j["confidence_pct"])
print(f"\nSame image x5: predictions {preds}, confidence {confs} "
      "(small MC Dropout variation is normal; a changing class is not)")