"""
uncertainty_analysis.py - READ-ONLY. Changes no code and no model files.

Measures how well each "review" signal catches ResNet50's mistakes, so you can
see why the MC Dropout flag (threshold 0.15) never fires and what would work.

Signals compared:
  - MC Dropout std of the predicted class at several thresholds (0.15 = current)
  - Low confidence (mean MC probability of the predicted class)
  - Disagreement with EfficientNet-B0

Run from the project root:

    python uncertainty_analysis.py            # validation set, every 2nd image
    python uncertainty_analysis.py 1 val      # whole validation set (slow on CPU)
    python uncertainty_analysis.py 2 test     # test set - use ONLY to report a
                                              # threshold already chosen on 'val'

Choose any threshold on the validation set, then report it on the test set.
"""
import os
import sys

import numpy as np
import torch

STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 2
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "val"
FOLDER = {"val": "val", "test": "test"}.get(SPLIT, SPLIT)
DATA_DIR = os.path.join("data", "cataract_final", FOLDER)
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

import backend_app as be  # loads the models; the server is not started

if not os.path.isdir(DATA_DIR):
    raise SystemExit(f"Folder not found: {DATA_DIR} (try 'validation' instead of 'val')")

files = []
for cls in sorted(os.listdir(DATA_DIR)):
    folder = os.path.join(DATA_DIR, cls)
    if os.path.isdir(folder):
        label = be.CLASS_NAMES.index(cls.lower())
        files += [(os.path.join(folder, f), label)
                  for f in sorted(os.listdir(folder)) if f.lower().endswith(EXTS)]
files = files[::STRIDE]
print(f"Analysing {len(files)} images from {DATA_DIR} (MC Dropout, {be.MC_SAMPLES} passes each)...")

transform = be.get_transforms(train=False)
rows = []  # (correct, std_pred, conf, disagree)
for i, (path, y) in enumerate(files):
    x = transform(be.open_image(path)).unsqueeze(0)
    mean_probs, unc, idx = be.mc_dropout_predict(
        be.primary_model, x, num_samples=be.MC_SAMPLES, device=be.DEVICE)
    be.primary_model.eval()
    with torch.no_grad():
        b = int(torch.softmax(be.second_model(x.to(be.DEVICE)), 1).argmax())
    rows.append((int(idx) == y, float(unc[idx]), float(mean_probs[idx]), b != int(idx)))
    if (i + 1) % 50 == 0:
        print(f"  ...{i + 1}/{len(files)}")

ok = np.array([r[0] for r in rows])
std = np.array([r[1] for r in rows])
conf = np.array([r[2] for r in rows])
dis = np.array([r[3] for r in rows])
n, errors = len(rows), int((~ok).sum())

print(f"\nImages: {n}   ResNet50 (MC) errors: {errors} ({100 * errors / n:.1f}%)")
print(f"MC std of predicted class: median {np.median(std):.4f}, max {std.max():.4f}")
if errors:
    print(f"  median std when correct {np.median(std[ok]):.4f} | when wrong {np.median(std[~ok]):.4f}")
    print(f"  median confidence when correct {np.median(conf[ok]):.3f} | when wrong {np.median(conf[~ok]):.3f}")


def line(name, flag):
    caught = int((flag & ~ok).sum())
    print(f"{name:<38} flagged {int(flag.sum()):>4} ({100 * flag.mean():4.1f}%)   "
          f"errors caught {caught}/{errors}")


print("\nSignal                                  flagged images             errors caught")
for t in (0.15, 0.10, 0.05, 0.02, 0.01, 0.005):
    line(f"MC std > {t}" + ("  (current)" if t == 0.15 else ""), std > t)
for t in (0.70, 0.90, 0.95, 0.99):
    line(f"confidence < {t:.2f}", conf < t)
line("models disagree", dis)
line("disagree OR confidence < 0.90", dis | (conf < 0.90))
line("disagree OR std > 0.02", dis | (std > 0.02))