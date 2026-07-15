"""
Weak-label MEETI / MIMIC-IV-ECG free-text reports into the 5 PTB-XL diagnostic
superclasses (NORM, MI, STTC, CD, HYP) used everywhere else in this repo.

WHY THIS EXISTS
---------------
PTB-XL ships structured SCP codes that map cleanly to 5 superclasses. MEETI does
*not*: it provides a free-text clinical `report` (from MIMIC-IV-ECG) plus a GPT-4o
`LLM_Interpretation`, but no structured multi-label targets. To evaluate the same
5-class task (zero-shot, linear probe) on MEETI, we derive those labels from the
report text with the transparent keyword rules below.

THIS IS WEAK SUPERVISION, NOT GROUND TRUTH.
Regex keyword matching on cardiology free text is approximate: it will miss
paraphrases and occasionally mislabel. Treat MEETI numbers derived this way as
indicative, not directly comparable to PTB-XL's curated labels. The rules are
deliberately kept here, in plain sight, so you can inspect and edit them.

For pure contrastive fine-tuning (image<->report), you do NOT need these labels
at all -- use `prepare_meeti.py --label-mode none`.

Public API
----------
    map_report_to_superclasses(text) -> sorted list of superclass codes
    label_matrix(text)               -> dict {code: 0/1} over config.CLASSES
"""
from __future__ import annotations

import re
from typing import Dict, List

import config as C

# ---------------------------------------------------------------------------
# Keyword rules.  Each pattern is a case-insensitive regex searched in the
# report text.  Edit freely -- add synonyms, tighten boundaries, etc.
# ---------------------------------------------------------------------------
ABNORMAL_PATTERNS: Dict[str, List[str]] = {
    # Myocardial infarction / acute injury
    "MI": [
        r"myocardial infarction",
        r"infarct",
        r"\bstemi\b",
        r"st[-\s]?elevation",
        r"pathologic(?:al)? q[-\s]?wave",
        r"\bq[-\s]?waves?\b",
    ],
    # ST/T changes, ischemia, repolarization abnormalities
    "STTC": [
        r"st[-\s]?depression",
        r"st[-\s/]?t\b",
        r"st[-\s]segment",
        r"t[-\s]?wave (?:abnormalit|change|inversion|flatten)",
        r"nonspecific|non[-\s]specific",
        r"repolar",
        r"ischemi|ischaemi",
    ],
    # Conduction disturbances / blocks / pre-excitation
    "CD": [
        r"\bblock\b",
        r"bundle[-\s]?branch",
        r"\blbbb\b|\brbbb\b|\bivcd\b|\blafb\b|\blpfb\b",
        r"intraventricular conduction",
        r"fascicular|hemiblock",
        r"conduction (?:delay|disturbance|abnormalit)",
        r"wolff[-\s]?parkinson|\bwpw\b|pre[-\s]?excitation",
        r"(?:first|second|third)[-\s]degree",
        r"\bav\b.{0,6}block|atrioventricular block",
    ],
    # Hypertrophy / chamber enlargement / atrial abnormality
    "HYP": [
        r"hypertroph",
        r"\blvh\b|\brvh\b|\blae\b|\brae\b",
        r"(?:ventricular|atrial) (?:hypertroph|enlargement|abnormalit)",
        r"enlargement",
        r"biatrial",
    ],
}

# A record is called NORM only when a "normal" cue is present AND no abnormal
# class fired (so "normal sinus rhythm with LBBB" -> CD, not NORM).
NORMAL_PATTERNS: List[str] = [
    r"normal ecg",
    r"normal electrocardiogram",
    r"normal tracing",
    r"within normal limits",
    r"otherwise normal",
    r"normal sinus rhythm",
]

_ABNORMAL_COMPILED = {
    cls: [re.compile(p, re.IGNORECASE) for p in pats]
    for cls, pats in ABNORMAL_PATTERNS.items()
}
_NORMAL_COMPILED = [re.compile(p, re.IGNORECASE) for p in NORMAL_PATTERNS]


def map_report_to_superclasses(text: str) -> List[str]:
    """Return the sorted subset of config.CLASSES implied by ``text``.

    Multi-label: several abnormal classes may fire. NORM is mutually exclusive
    with the abnormal classes and only assigned as a fallback normal cue.
    """
    if not text:
        return []
    hits = set()
    for cls, patterns in _ABNORMAL_COMPILED.items():
        if cls not in C.CLASSES:
            continue
        if any(rx.search(text) for rx in patterns):
            hits.add(cls)

    if not hits and "NORM" in C.CLASSES:
        if any(rx.search(text) for rx in _NORMAL_COMPILED):
            hits.add("NORM")

    return sorted(hits, key=C.CLASSES.index)


def label_matrix(text: str) -> Dict[str, int]:
    """Return a {class: 0/1} dict over config.CLASSES for one report string."""
    active = set(map_report_to_superclasses(text))
    return {c: int(c in active) for c in C.CLASSES}


if __name__ == "__main__":
    # Quick self-check on representative MIMIC-style statements.
    samples = [
        "Normal ECG. Normal sinus rhythm.",
        "Normal sinus rhythm with left bundle branch block",
        "Anteroseptal myocardial infarction, age undetermined",
        "Nonspecific ST-T wave changes; consider ischemia",
        "Left ventricular hypertrophy with repolarization abnormality",
        "First degree AV block",
        "Left atrial enlargement",
        "Sinus bradycardia",  # -> no superclass (expected: empty)
    ]
    for s in samples:
        print(f"{map_report_to_superclasses(s)!s:35s} <- {s}")
