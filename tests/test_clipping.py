"""Clipping: the repair must not make a file worse than leaving it, and heavy clipping must be refused.

Known answers: we take an unclipped waveform, clip it ourselves by a known amount, repair it, and compare with the
original. (Before 0.2 the repair lowered the peaks and was 5 to 18 dB further from the original than the untouched
clipped file at clipping up to 2%, while reporting success.)"""
import dataclasses

import numpy as np
import pytest
import soundfile as sf

from conftest import SR, speechlike, write
from wavewash.audio_fixer import fix_declip
from wavewash.config import PROFILES
from wavewash.quality_checker import analyze_file


def clip_to_ratio(base, target):
    """Scale `base` so that about `target` of the samples are at full scale, then clip. Returns (truth, clipped)."""
    lo, hi = 0.5, 400.0
    for _ in range(50):
        g = (lo * hi) ** 0.5
        lo, hi = (g, hi) if np.mean(np.abs(base * g) >= 1.0) < target else (lo, g)
    x = base * g
    return x, np.clip(x, -1.0, 1.0)


def best_gain_error(a, truth):
    """Error energy of `a` against `truth` after the best overall gain (the repair may lower the level)."""
    g = float(np.dot(a, truth) / np.dot(a, a))
    return float(np.sum((g * a - truth) ** 2))


@pytest.fixture(scope="module")
def base():
    return speechlike(seconds=8.0, f0=140, seed=3, noise=0.002).astype(np.float64)


@pytest.mark.parametrize("ratio,min_gain_db", [(0.001, -0.5), (0.005, -0.5), (0.02, 0.5), (0.05, 5.0)])
def test_repair_is_neutral_at_light_clipping_and_clearly_better_above(base, ratio, min_gain_db):
    truth, clipped = clip_to_ratio(base, ratio)
    fixed, desc = fix_declip(clipped.astype(np.float32), SR)
    assert desc and desc.startswith("restored")
    gain_db = 10 * np.log10(best_gain_error(clipped, truth) / best_gain_error(fixed.astype(np.float64), truth))
    assert gain_db >= min_gain_db, gain_db


def test_restored_peaks_are_scaled_to_fit_and_the_description_says_so(base):
    truth, clipped = clip_to_ratio(base, 0.05)
    fixed, desc = fix_declip(clipped.astype(np.float32), SR)
    assert np.abs(fixed).max() <= 0.999 + 1e-6
    assert "level lowered" in desc and "dB" in desc
    # and a file whose repaired peaks stay inside full scale is not rescaled
    quiet = np.sin(2 * np.pi * 300 * np.arange(SR) / SR) * 0.6
    flat = np.clip(quiet, -0.5, 0.5)
    out, d2 = fix_declip(flat.astype(np.float32), SR, threshold=0.5)
    assert d2 and "level lowered" not in d2


def test_a_repaired_sample_is_never_below_the_sample_it_replaces():
    t = np.arange(SR) / SR
    x = np.clip(np.sin(2 * np.pi * 300 * t) * 0.9, -0.5, 0.5)           # flat tops at +-0.5
    out, _ = fix_declip(x, SR, threshold=0.5)
    m = np.abs(x) >= 0.5
    assert np.all(np.abs(out[m]) >= 0.5 - 1e-12)
    assert np.sign(out[m]).tolist() == np.sign(x[m]).tolist()            # the sign never flips


def test_long_runs_and_edge_runs_are_left_untouched():
    x = 0.3 * np.sin(2 * np.pi * 200 * np.arange(SR) / SR)
    x[8000:8400] = 1.0                       # 25 ms flat run in the middle: longer than the 8 ms limit
    x[:3] = 1.0                              # a clipped run at the very start: no samples before it to fit a curve
    y = x.copy().astype(np.float32)
    out, desc = fix_declip(y, SR)
    assert out is y and desc is None         # nothing repairable: returned as is, no claim of a fix
    x2 = x.copy()
    x2[4000:4002] = 1.0                      # add one short, repairable run
    out2, d2 = fix_declip(x2.astype(np.float32), SR)
    assert d2 and "1 clipped run" in d2 and "2 run(s) left untouched" in d2
    assert np.array_equal(out2[8000:8400] / out2[8000], np.ones(400))


def test_no_clipping_means_no_change_and_no_claim():
    x = (0.3 * np.sin(2 * np.pi * 200 * np.arange(SR) / SR)).astype(np.float32)
    out, desc = fix_declip(x, SR)
    assert out is x and desc is None


def test_stereo_channels_keep_their_balance(base):
    truth, clipped = clip_to_ratio(base, 0.05)
    left = clipped.astype(np.float32)
    right = (0.3 * clipped).astype(np.float32)             # a quiet, unclipped copy
    st = np.stack([left, right], axis=1)
    out, desc = fix_declip(st, SR)
    assert desc and out.shape == st.shape
    ratio = np.linalg.norm(out[:, 1]) / np.linalg.norm(out[:, 0])
    assert 0.2 < ratio < 0.31                              # was exactly 0.3 before repair; lowered together, not separately
    assert np.abs(out).max() <= 0.999 + 1e-6


def test_integer_input_is_handled():
    x = np.clip(np.sin(2 * np.pi * 300 * np.arange(SR) / SR) * 1.4, -1, 1)
    out, desc = fix_declip((x * 32767).astype(np.int16).astype(np.float64) / 32767, SR)
    assert desc and out.dtype == np.float64


# ------------------------------------------------------------------ detection levels and the pipeline

def test_three_clipping_levels_in_the_quality_check(tmp_path, base):
    prof = PROFILES["asr"]
    assert prof.max_clipped_ratio < prof.max_repairable_clipped_ratio
    for ratio, expect in ((0.0, None), (0.01, "CLIPPING:"), (0.15, "CLIPPING_HEAVY:")):
        if ratio:
            _, y = clip_to_ratio(base, ratio)
        else:
            y = base * 0.3
        p = write(tmp_path / f"c{ratio}.wav", y.astype(np.float32))
        rep = analyze_file(p, prof)
        flags = [i.split(" ")[0] for i in rep.issues if i.startswith("CLIPPING")]
        assert flags == ([expect] if expect else []), (ratio, rep.issues)


def test_the_heavy_limit_is_a_setting(tmp_path, base):
    _, y = clip_to_ratio(base, 0.08)
    p = write(tmp_path / "c.wav", y.astype(np.float32))
    strict = PROFILES["asr"]
    loose = dataclasses.replace(strict, max_repairable_clipped_ratio=0.2)
    assert any(i.startswith("CLIPPING_HEAVY") for i in analyze_file(p, strict).issues)
    assert [i for i in analyze_file(p, loose).issues if i.startswith("CLIPPING")][0].startswith("CLIPPING:")


def test_heavily_clipped_file_is_rejected_without_a_repair_attempt(sandbox, run_pipeline, base):
    _, inp = sandbox
    _, y = clip_to_ratio(base, 0.15)
    write(inp / "heavy.wav", y.astype(np.float32))
    out, audit = run_pipeline(inp)
    row = audit["heavy.wav"]
    assert row["status"] == "failed" and not row["fixes_applied"]
    assert any(r.startswith("CLIPPING_HEAVY") for r in row["unfixable_reasons"])
    assert (out / "clean" / "heavy.wav").exists() is False
    assert any(p.name == "heavy.wav" for p in (out / "clean" / "_rejected").rglob("*.wav")) or \
        any(p.name == "heavy.wav" for p in out.rglob("clipping_heavy/heavy.wav"))


def test_heavy_clipping_is_still_rejected_when_another_problem_is_fixed_alongside(sandbox, run_pipeline, base):
    _, inp = sandbox
    _, y = clip_to_ratio(base, 0.15)
    write(inp / "heavy_dc.wav", (y + 0.05).clip(-1, 1).astype(np.float32))      # clipped AND off-centre (DC 0.05 is fixable)
    _, audit = run_pipeline(inp)
    row = audit["heavy_dc.wav"]
    assert row["status"] == "failed"
    assert any(r.startswith("CLIPPING_HEAVY") for r in row["unfixable_reasons"])


def test_moderate_clipping_is_repaired_and_the_log_says_how(sandbox, run_pipeline, base):
    _, inp = sandbox
    _, y = clip_to_ratio(base, 0.02)
    write(inp / "mod.wav", y.astype(np.float32))
    out, audit = run_pipeline(inp, purpose="custom")
    row = audit["mod.wav"]
    assert row["status"] == "fixed"
    assert any(f.startswith("restored") and "never below the clip level" in f for f in row["fixes_applied"])
    o, _ = sf.read(str(out / "clean" / "mod.wav"))
    assert np.abs(o).max() <= 0.999 + 1e-4


def _clipped_sine(n=4000, f=300.0, amp=1.3, ceiling=1.0):
    t = np.arange(n) / SR
    truth = amp * np.sin(2 * np.pi * f * t)
    return truth, np.clip(truth, -ceiling, ceiling)


def test_a_run_at_the_very_end_of_the_file_is_left_untouched():
    x = np.zeros(2000)
    x[:1000] = 0.3 * np.sin(np.arange(1000) / 7.0)
    x[-3:] = 1.0                      # clipped samples with no context after them
    x[-5:-3] = 0.9
    fixed, desc = fix_declip(x.copy(), SR)
    assert np.array_equal(np.asarray(fixed)[-3:], x[-3:])


def test_the_repair_follows_the_curve_using_enough_neighbouring_samples():
    truth, clipped = _clipped_sine(f=1500.0, amp=1.2)      # a fast wave: few samples per clipped run, curve matters
    fixed, _ = fix_declip(clipped.copy(), SR)
    fixed = np.asarray(fixed, dtype=np.float64)
    g = np.dot(fixed, truth) / np.dot(fixed, fixed)
    # measured: four neighbours each side leave 2.5% of the clipped file's error, two neighbours leave 7%
    assert np.sum((g * fixed - truth) ** 2) < 0.04 * np.sum((clipped - truth) ** 2)


def test_only_samples_at_the_ceiling_count_as_clipped_not_slightly_lower_peaks():
    x = 0.2 * np.sin(np.arange(3000) / 5.0)
    x[1000:1004] = 1.0                # real plateau at the ceiling
    x[2000] = 0.9995                  # a natural peak a hair below it: must not be touched
    x[1999], x[2001] = 0.5, 0.5
    fixed, desc = fix_declip(x.copy(), SR)
    assert desc and "1 clipped run" in desc          # only the plateau was repaired
    f = np.asarray(fixed)
    assert f[2000] / x[2000] == pytest.approx(f[2500] / x[2500], rel=1e-6)   # scaled like any untouched sample


def test_heavy_clipping_issue_does_not_trigger_the_declip_step():
    from wavewash.audio_fixer import apply_fix_pipeline
    _, clipped = _clipped_sine()
    out, _, applied = apply_fix_pipeline(clipped.astype(np.float32), SR, PROFILES["asr"],
                                         ["CLIPPING_HEAVY: 9% samples saturated"])
    assert not any("clipped run" in a for a in applied)
