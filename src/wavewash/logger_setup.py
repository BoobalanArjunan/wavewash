"""
logger_setup.py
----------------
Sets up two things:
1. A normal rolling text log (human-readable, for watching progress / debugging).
2. A structured per-file CSV + JSON log (machine-readable, for auditing every
   decision the pipeline made: what was checked, what was fixed, what failed,
   and why).
"""

import csv
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def build_text_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("wavewash")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_dir / "pipeline_run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


@dataclass
class FileRecord:
    """One row of the structured audit log -- one file, full decision trail."""
    source_path: str
    filename: str
    purpose: str
    status: str = "pending"          # pending | passed | fixed | failed | duplicate
    issues_found: List[str] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    unfixable_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duplicate_of: Optional[str] = None
    final_destination: Optional[str] = None
    original_sample_rate: Optional[int] = None
    original_channels: Optional[int] = None
    original_duration_sec: Optional[float] = None
    final_duration_sec: Optional[float] = None
    snr_db: Optional[float] = None
    clipped_ratio: Optional[float] = None
    silence_ratio: Optional[float] = None
    max_internal_silence_sec: Optional[float] = None
    dc_offset: Optional[float] = None
    lufs: Optional[float] = None
    spectral_rolloff_hz: Optional[float] = None
    spectral_rolloff_fraction: Optional[float] = None
    rt60_ms: Optional[float] = None
    crest_factor_db: Optional[float] = None
    vad_backend: Optional[str] = None
    speaker_id: Optional[str] = None
    session_id: Optional[str] = None
    transcript: Optional[str] = None
    wer: Optional[float] = None
    detected_language: Optional[str] = None
    language_match: Optional[bool] = None
    notes: str = ""

    def to_dict(self):
        return asdict(self)


class AuditLog:
    """Collects FileRecord entries and writes them to CSV + JSON at the end
    (and can also flush incrementally so partial runs still leave a usable log)."""

    CSV_FIELDS = [
        "source_path", "filename", "purpose", "status",
        "issues_found", "fixes_applied", "unfixable_reasons", "warnings", "duplicate_of",
        "final_destination", "original_sample_rate", "original_channels",
        "original_duration_sec", "final_duration_sec", "snr_db",
        "clipped_ratio", "silence_ratio", "max_internal_silence_sec", "dc_offset", "lufs",
        "spectral_rolloff_hz", "spectral_rolloff_fraction", "rt60_ms", "crest_factor_db",
        "vad_backend", "speaker_id", "session_id", "transcript", "wer",
        "detected_language", "language_match", "notes",
    ]

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[FileRecord] = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"audit_log_{ts}.csv"
        self.json_path = self.log_dir / f"audit_log_{ts}.json"
        self.summary_path = self.log_dir / f"summary_{ts}.md"

    def add(self, record: FileRecord):
        self.records.append(record)
        self._flush()

    def _flush(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for r in self.records:
                row = r.to_dict()
                row["issues_found"] = "; ".join(row["issues_found"])
                row["fixes_applied"] = "; ".join(row["fixes_applied"])
                row["unfixable_reasons"] = "; ".join(row["unfixable_reasons"])
                row["warnings"] = "; ".join(row["warnings"])
                writer.writerow(row)

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.records], f, indent=2)

    def write_summary(self, purpose: str, extra_stats: dict):
        total = len(self.records)
        passed = sum(1 for r in self.records if r.status == "passed")
        fixed = sum(1 for r in self.records if r.status == "fixed")
        failed = sum(1 for r in self.records if r.status == "failed")
        dupes = sum(1 for r in self.records if r.status == "duplicate")

        lines = [
            f"# WaveWash Pipeline Summary",
            f"",
            f"- Purpose profile: **{purpose}**",
            f"- Total files scanned: **{total}**",
            f"- Passed as-is: **{passed}**",
            f"- Passed after auto-fix: **{fixed}**",
            f"- Duplicates removed: **{dupes}**",
            f"- Failed (moved to error folder): **{failed}**",
            f"",
        ]
        if extra_stats:
            lines.append("## Additional stats")
            for k, v in extra_stats.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        if failed:
            lines.append("## Failure reasons breakdown")
            reason_counts = {}
            for r in self.records:
                if r.status == "failed":
                    for reason in r.unfixable_reasons:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                lines.append(f"- {reason}: {count} file(s)")

        self.summary_path.write_text("\n".join(lines), encoding="utf-8")
        return self.summary_path
