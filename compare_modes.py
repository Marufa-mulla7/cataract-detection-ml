"""
compare_modes.py - READ-ONLY check. It changes no code and no model files.

It answers three questions on the 583-image test set, using the app's own
preprocessing and loaded models:

  1. ResNet50 accuracy with plain deterministic inference (model.eval()).
     This should match the reported 555/583 = 95.20%.
  2. EfficientNet-B0 accuracy (the report says 561/583 = 96.23%, derived).
  3. How often ResNet50's MC Dropout answer (what the app shows) differs
     from its deterministic answer.

Run from the project root:

    python compare_modes.py        # MC Dropout checked on every 5th image
    python compare_modes.py 1      # MC Dropout on every image (slow on CPU)
"""
import os
import sys

import torch

STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TEST_DIR = os.path.join("data", "cataract_final", "test")
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

import backend_app as be  # loads the models; the server is not started

files = []
for cls in sorted(os.listdir(TEST_DIR)):
    folder = os.path.join(TEST_DIR, cls)
    if not os.path.isdir(folder):
        continue
    label = be.CLASS_NAMES.index(cls.lower())
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(EXTS):
            files.append((os.path.join(folder, f), label))

print(f"Test images found: {len(files)} (expected 583)")
transform = be.get_transforms(train=False)

n = res_ok = eff_ok = agree = agree_ok = both_wrong_disagree = 0
disagree = res_ok_disagree = eff_ok_disagree = 0
mc_n = mc_ok = mc_det_ok = mc_changed = 0

for i, (path, y) in enumerate(files):
    x = transform(be.open_image(path)).unsqueeze(0).to(be.DEVICE)
    with torch.no_grad():
        a = int(torch.softmax(be.primary_model(x), 1).argmax())
        b = int(torch.softmax(be.second_model(x), 1).argmax())
    n += 1
    res_ok += a == y
    eff_ok += b == y
    if a == b:
        agree += 1
        agree_ok += a == y
    else:
        disagree += 1
        res_ok_disagree += a == y
        eff_ok_disagree += b == y
        both_wrong_disagree += (a != y and b != y)

    if i % STRIDE == 0:
        mean_probs, _, mc_idx = be.mc_dropout_predict(
            be.primary_model, x.cpu(), num_samples=be.MC_SAMPLES, device=be.DEVICE)
        be.primary_model.eval()
        mc_n += 1
        mc_ok += int(mc_idx) == y
        mc_det_ok += a == y
        mc_changed += int(mc_idx) != a
    if (i + 1) % 100 == 0:
        print(f"  ...{i + 1}/{len(files)} done")

print("\n=== Deterministic inference (model.eval, no MC Dropout) ===")
print(f"ResNet50        : {res_ok}/{n} = {100 * res_ok / n:.2f}%   (report: 555/583 = 95.20%)")
print(f"EfficientNet-B0 : {eff_ok}/{n} = {100 * eff_ok / n:.2f}%   (report: 561/583 = 96.23%)")
print("\n=== Agreement between the two models ===")
print(f"Agree    : {agree}/{n}  (report: 552)   accuracy when agree: {agree_ok}/{agree}  (report: 543/552)")
print(f"Disagree : {disagree}/{n}  (report: 31)    ResNet50 right: {res_ok_disagree}  (report: 12)"
      f"   EfficientNet right: {eff_ok_disagree}  (report: 18)   both wrong: {both_wrong_disagree}  (report: 1)")
print(f"\n=== App path (MC Dropout, {be.MC_SAMPLES} passes) on {mc_n} images (every {STRIDE}th) ===")
print(f"ResNet50 with MC Dropout     : {mc_ok}/{mc_n} correct")
print(f"ResNet50 deterministic, same : {mc_det_ok}/{mc_n} correct")
print(f"Predicted class changed by MC Dropout: {mc_changed}/{mc_n}")