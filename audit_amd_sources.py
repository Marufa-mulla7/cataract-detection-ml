"""
Read-only audit: how many REAL AMD images could still be added?

1. Counts the AMD images already in data/cataract/amd by source
   (filename prefix: rfmid_, adam_amd_, odir_, ...).
2. Reads the label CSVs INSIDE _rfmid_download/rfmid.zip (no extraction of the
   7.6 GB archive) and counts ARMD-positive rows in each split, so you can see
   how many were never used.
3. Checks whether an ODIR download is still on disk.

It does not copy, move or delete anything.

Run from the project root:
    python audit_amd_sources.py
"""

import io
import os
import zipfile
from collections import Counter

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
AMD_DIR = os.path.join(PROJECT_ROOT, "data", "cataract", "amd")
RFMID_ZIP = os.path.join(PROJECT_ROOT, "_rfmid_download", "rfmid.zip")
ODIR_DIR = os.path.join(PROJECT_ROOT, "_odir_download")


def main():
    # 1. What is already in the AMD folder
    print("=" * 60)
    print("AMD IMAGES CURRENTLY IN data\\cataract\\amd, BY SOURCE")
    print("=" * 60)
    if os.path.isdir(AMD_DIR):
        counts = Counter()
        for name in os.listdir(AMD_DIR):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            prefix = name.split("_")[0] if "_" in name else "other"
            if name.startswith("adam_amd"):
                prefix = "adam_amd"
            counts[prefix] += 1
        for prefix, n in counts.most_common():
            print(f"  {prefix:12s} {n}")
        print(f"  {'TOTAL':12s} {sum(counts.values())}")
    else:
        print(f"  Folder not found: {AMD_DIR}")

    # 2. RFMiD label CSVs inside the zip
    print("\n" + "=" * 60)
    print("RFMiD: ARMD-POSITIVE ROWS PER LABEL FILE (read from the zip)")
    print("=" * 60)
    if not os.path.isfile(RFMID_ZIP):
        print(f"  Zip not found: {RFMID_ZIP}")
    else:
        with zipfile.ZipFile(RFMID_ZIP) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                print("  No CSV files found in the zip.")
            for name in csv_names:
                try:
                    df = pd.read_csv(io.BytesIO(zf.read(name)))
                except Exception as e:
                    print(f"  {name}: could not read ({e})")
                    continue
                if "ARMD" not in df.columns:
                    print(f"  {name}: no ARMD column, skipped")
                    continue
                armd = int((df["ARMD"] == 1).sum())
                armd_no_dr = int(
                    ((df["ARMD"] == 1) & (df["DR"] == 0)).sum()
                ) if "DR" in df.columns else armd
                print(f"  {name}")
                print(f"     rows: {len(df)}   ARMD=1: {armd}   "
                      f"ARMD=1 and DR=0: {armd_no_dr}")

    # 3. ODIR on disk?
    print("\n" + "=" * 60)
    print("ODIR DOWNLOAD ON DISK?")
    print("=" * 60)
    print(f"  {ODIR_DIR}: "
          f"{'FOUND' if os.path.isdir(ODIR_DIR) else 'not found (would need re-download)'}")


if __name__ == "__main__":
    main()