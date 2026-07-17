# Run this from the repo root, with the same WORK_DIR you used:
#     export WORK_DIR=./work_meeti
#     python filter_imaged.py
import os, pandas as pd, config as C

path = os.path.join(C.WORK_DIR, "labels.csv")
df = pd.read_csv(path, index_col="ecg_id")
before = len(df)
df = df[df["has_image"] == 1]
df.to_csv(path)

tr = df.strat_fold.isin(C.TRAIN_FOLDS).sum()
va = (df.strat_fold == C.VAL_FOLD).sum()
te = (df.strat_fold == C.TEST_FOLD).sum()
print(f"{path}: {before} -> {len(df)} rows (image-backed only)")
print(f"train {tr} | val {va} | test {te}")
print("per-class positives:")
for c in C.CLASSES:
    print(f"  {c:5s}: {int(df[c].sum())}")
