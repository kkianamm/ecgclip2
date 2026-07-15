# Adding the MEETI dataset

This repo was built around **PTB-XL**. `prepare_meeti.py` lets you run the same
pipeline — zero-shot, linear probe, contrastive fine-tuning — on
**MEETI** (MIMIC-IV-Ext ECG-Text-Image, <https://zenodo.org/records/15893351>),
without touching any of the evaluation/training scripts.

The trick: the whole repo only ever reads two things — `WORK_DIR/labels.csv`
(indexed by `ecg_id`, with `strat_fold`, `report`, `superclasses`, and one 0/1
column per class in `config.CLASSES`) and `WORK_DIR/images/<ecg_id>.png`.
`prepare_meeti.py` produces exactly those from MEETI, so `zero_shot_eval.py`,
`extract_features.py`, `linear_probe.py`, and `finetune_clip.py` all work as-is.

## How MEETI differs from PTB-XL (and how we bridge it)

| | PTB-XL | MEETI | Bridge |
|---|---|---|---|
| Waveforms | WFDB signals | none in the Zenodo zip (`.mat` = features+text) | reuse MEETI's shipped PNGs |
| Images | rendered by `ecg_to_image.py` | shipped as `.png` (12×1 leads) | copied/symlinked into `images/` |
| Labels | SCP codes → 5 superclasses | free-text `report` only | **weak-labeled** from text (`meeti_labeling.py`) |
| Split | official `strat_fold` | none | deterministic, **patient-disjoint** by `subject_id` |
| Report language | often German | English (MIMIC) | use `--caption report` directly |

Two consequences to keep in mind:

- **Labels are weak supervision.** They come from regex keyword rules over the
  report text, not curated annotations. Use MEETI numbers as indicative, and do
  **not** compare them head-to-head with PTB-XL's curated-label numbers. The
  rules live in `meeti_labeling.py` — read and edit them. Run
  `python meeti_labeling.py` to see them fire on sample statements.
- **MEETI PNGs are a different visual style** than this repo's renders. Fine for
  MEETI-only experiments. If you want cross-dataset transfer with pixel-identical
  inputs, render MEETI from the raw MIMIC-IV-ECG waveforms instead (see
  `--render-from-wfdb`).

## Download

The 3.3 GB Zenodo package (~10k imaged records):

```bash
# from https://zenodo.org/records/15893351
wget https://zenodo.org/records/15893351/files/MEETI.zip?download=1 -O MEETI.zip
unzip MEETI.zip -d MEETI      # yields MEETI/pNNNN/pXXXXXXXX/sZZZ/<id>.{mat,png}
```

The full 784k-record dataset lives on Hugging Face
(`PKUDigitalHealth/MEETI`) if you need scale.

## Prepare

Keep MEETI outputs separate from your PTB-XL run with `WORK_DIR`:

```bash
export WORK_DIR=./work_meeti

python prepare_meeti.py --meeti-dir ./MEETI --limit 500   # smoke test first
python prepare_meeti.py --meeti-dir ./MEETI               # full
```

This writes `./work_meeti/labels.csv` and fills `./work_meeti/images/`. It also
prints the per-class positive counts and how many reports matched no superclass.

Useful flags:

- `--label-mode none` — skip weak-labeling entirely (class columns all zero).
  Use this when you only want **contrastive fine-tuning** on image↔report pairs.
- `--caption-source {report,llm,report+llm}` — which text lands in the `report`
  column that `finetune_clip.py --caption report` consumes. `report` = the
  clinician text; `llm` = the richer GPT-4o interpretation. The clinician
  `report_raw` and `llm_interpretation` are always kept as extra columns too.
- `--image-mode symlink` — symlink PNGs instead of copying (saves disk).
- `--render-from-wfdb /path/to/mimic-iv-ecg` — if you have the matching MIMIC
  WFDB records, render missing images in the identical PTB-XL style via this
  repo's own `ecg_to_image.py`.

## Run the existing pipeline (unchanged)

```bash
export WORK_DIR=./work_meeti

# Zero-shot BiomedCLIP, 5-class multi-label
python zero_shot_eval.py

# Linear probe on frozen features
python extract_features.py && python linear_probe.py

# Contrastive fine-tuning — MEETI's English reports make this the natural caption
python finetune_clip.py --caption report
python zero_shot_eval.py --ckpt work_meeti/checkpoints/biomedclip_ft.pt
```

> For MEETI, prefer `--caption report` over the default `--caption label`: the
> reports are informative English, and records with no matched superclass would
> otherwise produce an empty label caption.

## Files added

- `prepare_meeti.py` — the adapter (discovery, `.mat` parsing, weak-labeling,
  patient-disjoint split, image placement, `labels.csv` writer).
- `meeti_labeling.py` — the transparent report → superclass keyword rules.
- `README_MEETI.md` — this file.

No existing files were modified.
