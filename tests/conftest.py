"""Shared helpers: tiny synthetic audio with KNOWN defects, so every test knows the right answer."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 16000


def speechlike(seconds=4.0, f0=180.0, seed=0, noise=0.002):
    """A voiced-sounding signal with pauses (so SNR estimation has quiet parts)."""
    rng = np.random.RandomState(seed)
    t = np.arange(int(SR * seconds)) / SR
    sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3, 4))
    gate = (np.sin(2 * np.pi * 1.2 * t) > -0.2).astype(float)       # on/off like words and pauses
    env = np.convolve(gate, np.ones(400) / 400, mode="same")
    return (0.25 * sig * env + noise * rng.randn(len(t))).astype(np.float32)


def write(path: Path, audio, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    return path


def tree_hashes(root: Path):
    """Hash every file under root: used to prove inputs are never modified."""
    return {str(p.relative_to(root)): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def read_audit(out_base: Path):
    files = sorted((out_base / "_logs").glob("audit_log_*.json"))
    assert files, f"no audit log in {out_base / '_logs'}"
    rows = json.loads(files[-1].read_text())
    return {r["filename"]: r for r in rows}


@pytest.fixture
def sandbox(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    return tmp_path, inp


@pytest.fixture
def run_pipeline(tmp_path):
    from wavewash.pipeline import main

    def _run(inp: Path, purpose="asr", extra=None, name="out"):
        base = tmp_path / name
        main(["--input", str(inp), "--output", str(base / "clean"), "--purpose", purpose,
              "--review-sample-size", "0", "--no-diversity-report"] + (extra or []))
        return base, read_audit(base)
    return _run
