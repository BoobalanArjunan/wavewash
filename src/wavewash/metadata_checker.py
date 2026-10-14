"""
metadata_checker.py
---------------------
Closes the "dataset-level: speaker diversity, metadata consistency, session
leakage" gap. All of this is optional and only activates if you provide a
metadata file mapping filenames to speaker_id / session_id.

Metadata file format (CSV, header required):
    filename,speaker_id,session_id
    spk1_001.wav,speaker_01,session_A
    spk1_002.wav,speaker_01,session_A
    spk2_001.wav,speaker_02,session_B

What it does:
1. Speaker balance report -- utterance count + total duration per speaker,
   flags speakers that are wildly over/under-represented.
2. Cross-contamination check -- for each file, compares its audio "fingerprint"
   (cheap MFCC-based proxy -- see quality_checker.speaker_embedding) against
   the centroid of its claimed speaker vs. every other speaker's centroid.
   If a file is much closer to a *different* speaker's centroid, it's flagged
   as a likely mislabel worth a human check.
3. Session-aware split -- builds train/val file lists so that no session ID
   appears in both splits (prevents the classic leakage bug that inflates
   eval scores).

This is a lightweight heuristic layer, not a replacement for a real speaker
verification model -- treat contamination flags as "worth a listen", not
ground truth.
"""

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np


def load_metadata(path: str) -> Dict[str, Dict[str, str]]:
    """Returns {filename: {"speaker_id": ..., "session_id": ...}}"""
    mapping = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = Path(row.get("filename", "").strip()).name
            if not fname:
                continue
            mapping[fname] = {
                "speaker_id": row.get("speaker_id", "").strip() or None,
                "session_id": row.get("session_id", "").strip() or None,
            }
    return mapping


def speaker_balance_report(records: List[dict]) -> Tuple[str, Dict[str, dict]]:
    """
    records: list of dicts each with at least 'speaker_id', 'duration_sec'.
    Returns (markdown_text, per_speaker_stats).
    """
    per_speaker = defaultdict(lambda: {"count": 0, "total_duration": 0.0})
    for r in records:
        spk = r.get("speaker_id")
        if not spk:
            continue
        per_speaker[spk]["count"] += 1
        per_speaker[spk]["total_duration"] += r.get("duration_sec") or 0.0

    if not per_speaker:
        return "", {}

    durations = [v["total_duration"] for v in per_speaker.values()]
    mean_dur = np.mean(durations)
    lines = ["## Speaker balance", "",
             f"- {len(per_speaker)} distinct speaker(s)",
             f"- Mean audio per speaker: {mean_dur:.1f}s", "",
             "| Speaker | Utterances | Total duration (s) | vs. mean |",
             "|---|---|---|---|"]
    flagged = []
    for spk, stats in sorted(per_speaker.items(), key=lambda x: -x[1]["total_duration"]):
        ratio = stats["total_duration"] / mean_dur if mean_dur > 0 else 1.0
        flag = ""
        if ratio > 3:
            flag = "  overrepresented"
            flagged.append(spk)
        elif ratio < 0.2:
            flag = "  underrepresented"
            flagged.append(spk)
        lines.append(f"| {spk} | {stats['count']} | {stats['total_duration']:.1f} | "
                     f"{ratio:.2f}x{flag} |")

    if flagged:
        lines.append("")
        lines.append(f"Flagged: {len(flagged)} speaker(s) notably imbalanced relative "
                      f"to the dataset mean -- consider rebalancing before training.")

    return "\n".join(lines), dict(per_speaker)


def cross_contamination_check(
    embeddings: Dict[str, np.ndarray],
    speaker_ids: Dict[str, str],
    distance_margin: float = 1.15,
) -> Dict[str, str]:
    """
    embeddings: {filename: embedding_vector}
    speaker_ids: {filename: claimed_speaker_id}

    Returns {filename: warning_message} for files that look closer to a
    different speaker's centroid than their own (possible mislabel).
    """
    by_speaker = defaultdict(list)
    for fname, spk in speaker_ids.items():
        if fname in embeddings and spk:
            by_speaker[spk].append(embeddings[fname])

    if len(by_speaker) < 2:
        return {}

    centroids = {spk: np.mean(np.stack(vecs), axis=0) for spk, vecs in by_speaker.items()
                 if len(vecs) >= 1}

    warnings = {}
    for fname, spk in speaker_ids.items():
        if fname not in embeddings or spk not in centroids:
            continue
        vec = embeddings[fname]
        own_dist = np.linalg.norm(vec - centroids[spk])
        other_dists = {s: np.linalg.norm(vec - c) for s, c in centroids.items() if s != spk}
        if not other_dists:
            continue
        nearest_other, nearest_dist = min(other_dists.items(), key=lambda x: x[1])
        if nearest_dist * distance_margin < own_dist:
            warnings[fname] = (f"closer to speaker '{nearest_other}' (dist {nearest_dist:.1f}) "
                                f"than claimed speaker '{spk}' (dist {own_dist:.1f}) -- possible mislabel")
    return warnings


def session_aware_split(
    filenames: List[str],
    session_ids: Dict[str, str],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Splits files into (train, val) such that no session_id appears in both.
    Files with no known session_id are treated as their own singleton session
    (so they can land on either side freely).
    """
    sessions = defaultdict(list)
    for fname in filenames:
        sid = session_ids.get(fname) or f"__no_session__{fname}"
        sessions[sid].append(fname)

    session_keys = list(sessions.keys())
    rng = random.Random(seed)
    rng.shuffle(session_keys)

    total_files = len(filenames)
    target_train = int(total_files * train_ratio)

    train, val = [], []
    train_count = 0
    for sid in session_keys:
        files = sessions[sid]
        if train_count < target_train:
            train.extend(files)
            train_count += len(files)
        else:
            val.extend(files)

    return train, val
