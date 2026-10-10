"""
dataset_stats.py - READ-ONLY. Summarises data/cataract_final/manifest.csv.

Shows, for every class and split, how many images are real vs augmented, and the
source mix of the REAL images (filename prefix such as rfmid_, adam_amd_, odir_).
Use it to fill the dataset-sources part of the report.

    python dataset_stats.py
"""
import os
import re

import pandas as pd

path = os.path.join("data", "cataract_final", "manifest.csv")
df = pd.read_csv(path)
def source_of(name):
    stem = os.path.splitext(str(name))[0]
    if "_" in stem:
        return stem.split("_")[0]
    if re.fullmatch(r"[0-9a-f]{12}", stem):
        return "hex-id (APTOS style)"
    return stem


df["source"] = df["parent_source"].map(source_of)

print("Images per class / split / type:")
print(df.groupby(["class", "split", "sample_type"]).size().unstack(fill_value=0).to_string())

print("\nREAL images by source prefix (all splits):")
real = df[df["sample_type"] == "real"]
print(real.groupby(["class", "source"]).size().unstack(fill_value=0).to_string())

print("\nREAL images by source prefix and split:")
print(real.groupby(["class", "split", "source"]).size().to_string())