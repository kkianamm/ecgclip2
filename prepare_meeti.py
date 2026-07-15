"""
Adapt the MEETI dataset (MIMIC-IV-Ext ECG-Text-Image, https://zenodo.org/records/15893351)
to the exact on-disk contract this repo already uses for PTB-XL, so that
`zero_shot_eval.py`, `extract_features.py`, `linear_probe.py` and
`finetune_clip.py` all run on MEETI *unchanged*.

What the repo expects (the whole contract):
  1. WORK_DIR/labels.csv   indexed by `ecg_id`, with columns:
       strat_fold, report, superclasses (pipe-joined), and one 0/1 column per
       class in config.CLASSES (NORM, MI, STTC, CD, HYP).
  2. WORK_DIR/images/<ecg_id>.png   one rendered ECG image per record,
       reachable via prepare_data.image_path_for(ecg_id).
Downstream scripts only ever read those two things.

Key differences from PTB-XL and how we bridge them:
  * MEETI already ships plotted PNGs (12x1 leads) -> we reuse them directly
    (copy or symlink) instead of rendering from a waveform. The Zenodo package
    has no WFDB signals; if you separately have the MIMIC-IV-ECG waveforms you
    can render in the identical PTB-XL style with --render-from-wfdb.
  * MEETI has no structured labels -> we weak-label the free-text `report`
    into the 5 superclasses (see meeti_labeling.py). Use --label-mode none to
    skip this entirely (fine for contrastive fine-tuning on image<->report).
  * MEETI has no fold column -> we build a deterministic, patient-disjoint
    split by subject_id into folds 1..10 (train 1-8, val 9, test 10), matching
    how the rest of the repo slices strat_fold.

Typical use (keep MEETI separate from your PTB-XL run via WORK_DIR):

    export WORK_DIR=./work_meeti
    python prepare_meeti.py --meeti-dir /path/to/MEETI --limit 500   # smoke test
    python prepare_meeti.py --meeti-dir /path/to/MEETI               # full

    python zero_shot_eval.py                       # 5-class zero-shot on MEETI
    python extract_features.py && python linear_probe.py
    python finetune_clip.py --caption report       # MIMIC reports are English
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C
from meeti_labeling import map_report_to_superclasses

try:
    # Reuse the exact image-path convention from the rest of the repo.
    from prepare_data import image_path_for
except Exception:  # noqa: BLE001 - wfdb (pulled in transitively) may be absent
    def image_path_for(ecg_id):  # identical to prepare_data.image_path_for
        return os.path.join(C.IMG_DIR, f"{int(ecg_id):05d}.png")


# ---------------------------------------------------------------------------
# .mat helpers -- MEETI stores id/report/LLM_Interpretation (and per-beat
# features) inside a MATLAB .mat file; scipy returns nested numpy arrays.
# ---------------------------------------------------------------------------
def _flatten_to_str(value) -> str:
    """Recursively join whatever scipy.io.loadmat returned into a plain string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").strip()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("U", "S"):
            return " ".join(str(x).strip() for x in value.ravel() if str(x).strip())
        parts = [_flatten_to_str(v) for v in value.ravel()]
        return " ".join(p for p in parts if p).strip()
    if isinstance(value, (list, tuple)):
        parts = [_flatten_to_str(v) for v in value]
        return " ".join(p for p in parts if p).strip()
    return str(value).strip()


def read_mat_fields(mat_path: str, keys=("report", "LLM_Interpretation", "id")) -> Dict[str, str]:
    """Return the requested string fields from a MEETI .mat file (best effort)."""
    from scipy.io import loadmat  # imported lazily; scipy is already a dependency

    data = loadmat(mat_path)
    out: Dict[str, str] = {}
    lower = {k.lower(): k for k in data.keys()}
    for want in keys:
        real = lower.get(want.lower())
        out[want] = _flatten_to_str(data[real]) if real is not None else ""
    return out


# ---------------------------------------------------------------------------
# Record discovery
# ---------------------------------------------------------------------------
def _study_id_from_stem(stem: str) -> Optional[int]:
    """MEETI study files are named <study_id>.png/.mat; keep only int-castable stems."""
    try:
        return int(stem)
    except ValueError:
        return None


def _subject_from_path(mat_path: Path) -> str:
    """Extract subject id from .../pXXXX/pXXXXXXXX/sZZZ/ZZZ.mat -> 'pXXXXXXXX'."""
    for part in mat_path.parts:
        if part.startswith("p") and part[1:].isdigit() and len(part) > 5:
            return part
    # Fall back to the immediate grandparent directory name.
    return mat_path.parent.parent.name


def discover_records(meeti_dir: Path) -> List[Dict]:
    """Return [{ecg_id, subject_id, mat, png}] by reading record_list.csv if
    usable, otherwise by walking the directory tree for *.mat files."""
    records: List[Dict] = []

    record_list = meeti_dir / "record_list.csv"
    used_csv = False
    if record_list.exists():
        try:
            df = pd.read_csv(record_list)
            cols = {c.lower(): c for c in df.columns}
            path_col = cols.get("path") or cols.get("file_path")
            name_col = cols.get("file_name") or cols.get("filename") or cols.get("record")
            subj_col = cols.get("subject_id") or cols.get("subject")
            if path_col or name_col:
                for _, row in df.iterrows():
                    rel = str(row[path_col]) if path_col else ""
                    stem = str(row[name_col]) if name_col else Path(rel).stem
                    stem = Path(stem).stem  # drop any extension
                    ecg_id = _study_id_from_stem(stem)
                    if ecg_id is None:
                        continue
                    base = (meeti_dir / rel) if rel else None
                    # `rel` may point at the folder or at the file with/without ext.
                    if base and base.suffix:
                        base = base.with_suffix("")
                    mat = _first_existing(
                        base.with_suffix(".mat") if base else None,
                        list(meeti_dir.rglob(f"{stem}.mat")),
                    )
                    if mat is None:
                        continue
                    png = mat.with_suffix(".png")
                    subj = str(row[subj_col]) if subj_col else _subject_from_path(mat)
                    records.append(
                        {"ecg_id": ecg_id, "subject_id": subj,
                         "mat": mat, "png": png if png.exists() else None}
                    )
                used_csv = len(records) > 0
        except Exception as exc:  # noqa: BLE001 - fall back to walking the tree
            print(f"  record_list.csv present but unreadable ({exc}); walking tree instead")

    if not used_csv:
        for mat in meeti_dir.rglob("*.mat"):
            ecg_id = _study_id_from_stem(mat.stem)
            if ecg_id is None:
                continue
            png = mat.with_suffix(".png")
            records.append(
                {"ecg_id": ecg_id, "subject_id": _subject_from_path(mat),
                 "mat": mat, "png": png if png.exists() else None}
            )

    # De-duplicate on ecg_id (first occurrence wins).
    seen, unique = set(), []
    for r in records:
        if r["ecg_id"] in seen:
            continue
        seen.add(r["ecg_id"])
        unique.append(r)
    return unique


def _first_existing(single: Optional[Path], candidates: List[Path]) -> Optional[Path]:
    if single is not None and Path(single).exists():
        return Path(single)
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return None


# ---------------------------------------------------------------------------
# Patient-disjoint fold assignment (deterministic, machine-independent)
# ---------------------------------------------------------------------------
def fold_for_subject(subject_id: str, n_folds: int = 10) -> int:
    """Stable hash of subject_id -> fold in 1..n_folds (train 1-8, val 9, test 10)."""
    h = hashlib.md5(str(subject_id).encode("utf-8")).hexdigest()
    return (int(h, 16) % n_folds) + 1


# ---------------------------------------------------------------------------
# Image placement
# ---------------------------------------------------------------------------
def place_image(src_png: Optional[Path], ecg_id: int, mode: str,
                render_from_wfdb: Optional[Path]) -> bool:
    """Make WORK_DIR/images/<ecg_id>.png exist. Returns True on success."""
    dst = Path(image_path_for(ecg_id))
    if dst.exists():
        return True

    if src_png is not None and Path(src_png).exists():
        if mode == "symlink":
            try:
                os.symlink(os.path.abspath(src_png), dst)
                return True
            except OSError:
                pass  # fall through to copy (e.g. Windows without privilege)
        shutil.copyfile(src_png, dst)
        return True

    if render_from_wfdb is not None:
        # Optional: render in the identical PTB-XL style if you have the
        # matching MIMIC-IV-ECG waveform for this study id.
        from ecg_to_image import load_signal, render_to_file
        matches = list(Path(render_from_wfdb).rglob(f"{ecg_id}.hea"))
        if matches:
            rec = str(matches[0].with_suffix(""))
            signal, fields = load_signal("", rec)
            fs = int(fields.get("fs", C.SAMPLING_RATE))
            render_to_file(signal, fs, str(dst))
            return True

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meeti-dir", required=True,
                    help="root of the extracted MEETI dataset (contains the pNNNN/ tree "
                         "and, usually, record_list.csv)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N discovered records (quick test)")
    ap.add_argument("--label-mode", choices=["rule", "none"], default="rule",
                    help="'rule' = weak-label reports into the 5 superclasses; "
                         "'none' = leave labels empty (contrastive fine-tuning only)")
    ap.add_argument("--caption-source", choices=["report", "llm", "report+llm"],
                    default="report",
                    help="which text goes into the `report` column that "
                         "finetune_clip.py --caption report consumes")
    ap.add_argument("--image-mode", choices=["copy", "symlink"], default="copy",
                    help="how to place MEETI PNGs into WORK_DIR/images/")
    ap.add_argument("--render-from-wfdb", default=None,
                    help="optional dir of MIMIC-IV-ECG WFDB records; render missing "
                         "images in the identical PTB-XL style instead of using PNGs")
    ap.add_argument("--n-folds", type=int, default=10)
    args = ap.parse_args()

    meeti_dir = Path(args.meeti_dir).expanduser()
    if not meeti_dir.exists():
        raise SystemExit(f"MEETI dir not found: {meeti_dir}")

    print(f"Scanning {meeti_dir} ...")
    records = discover_records(meeti_dir)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No MEETI records discovered. Check --meeti-dir points at "
                         "the folder containing the pNNNN/ subject tree.")
    print(f"Discovered {len(records)} records")

    render_dir = Path(args.render_from_wfdb).expanduser() if args.render_from_wfdb else None

    rows = []
    n_images, n_missing_img, label_counts = 0, 0, {c: 0 for c in C.CLASSES}
    n_unlabeled = 0

    for rec in tqdm(records, desc="Preparing MEETI"):
        fields = read_mat_fields(rec["mat"])
        report = fields.get("report", "")
        llm = fields.get("LLM_Interpretation", "")

        if args.caption_source == "llm":
            caption_text = llm or report
        elif args.caption_source == "report+llm":
            caption_text = (report + " " + llm).strip()
        else:
            caption_text = report

        if args.label_mode == "rule":
            supers = map_report_to_superclasses(report or llm)
        else:
            supers = []
        if not supers:
            n_unlabeled += 1
        for c in supers:
            label_counts[c] += 1

        ok = place_image(rec["png"], rec["ecg_id"], args.image_mode, render_dir)
        n_images += int(ok)
        n_missing_img += int(not ok)

        row = {
            "ecg_id": rec["ecg_id"],
            "subject_id": rec["subject_id"],
            "strat_fold": fold_for_subject(rec["subject_id"], args.n_folds),
            "report": caption_text,
            "report_raw": report,
            "llm_interpretation": llm,
            "superclasses": "|".join(supers),
            "has_image": int(ok),
        }
        for c in C.CLASSES:
            row[c] = 1.0 if c in supers else 0.0
        rows.append(row)

    df = pd.DataFrame(rows).set_index("ecg_id")

    # Order columns to mirror PTB-XL's labels.csv (index first, then meta, classes).
    ordered = ["subject_id", "strat_fold", "report", "report_raw",
               "llm_interpretation", "superclasses", "has_image"] + list(C.CLASSES)
    df = df[ordered]

    out = os.path.join(C.WORK_DIR, "labels.csv")
    df.to_csv(out)

    # ---- report ----
    train = df.strat_fold.isin(C.TRAIN_FOLDS).sum()
    val = (df.strat_fold == C.VAL_FOLD).sum()
    test = (df.strat_fold == C.TEST_FOLD).sum()
    print(f"\nSaved labels -> {out}")
    print(f"Records: {len(df)} | train {train} val {val} test {test} "
          f"(patient-disjoint by subject_id)")
    print(f"Images placed: {n_images} | missing image: {n_missing_img}")
    if args.label_mode == "rule":
        for c in C.CLASSES:
            print(f"  {c:5s}: {label_counts[c]} positive")
        print(f"  (no superclass matched: {n_unlabeled} records)")
        print("\nNOTE: these labels are WEAK (regex over free-text reports), not "
              "curated ground truth. See meeti_labeling.py to inspect/adjust rules.")
    else:
        print("Label mode 'none': class columns are all zero. Use MEETI for "
              "contrastive fine-tuning (finetune_clip.py --caption report).")
    if n_missing_img:
        print(f"\nWARNING: {n_missing_img} records have no image and will be skipped "
              "by evaluation. Provide --render-from-wfdb or check the PNGs exist.")


if __name__ == "__main__":
    main()
