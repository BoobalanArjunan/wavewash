"""End-to-end behaviour of the pipeline on synthetic files whose defects we planted ourselves."""
import shutil

import numpy as np
import soundfile as sf

from conftest import SR, speechlike, write, tree_hashes


def test_good_file_passes_and_lands_in_clean(sandbox, run_pipeline):
    _, inp = sandbox
    write(inp / "good.wav", speechlike())
    base, audit = run_pipeline(inp)
    assert audit["good.wav"]["status"] in ("passed", "fixed")
    assert (base / "clean" / "good.wav").exists()


def test_dc_offset_is_repaired_and_output_is_centered(sandbox, run_pipeline):
    _, inp = sandbox
    write(inp / "dc.wav", speechlike(seed=1) + 0.1)
    base, audit = run_pipeline(inp)
    assert audit["dc.wav"]["status"] == "fixed"
    assert any("DC" in f.upper() for f in audit["dc.wav"]["fixes_applied"])
    out, _sr = sf.read(str(base / "clean" / "dc.wav"))
    assert abs(float(np.mean(out))) < 0.01


def test_clipping_is_never_silently_passed(sandbox, run_pipeline):
    _, inp = sandbox
    clipped = np.clip(speechlike(seed=2) * 6.0, -1.0, 1.0).astype(np.float32)
    write(inp / "clip.wav", clipped)
    base, audit = run_pipeline(inp)
    row = audit["clip.wav"]
    # Allowed outcomes: repaired (fewer clipped samples) or rejected with a reason. Never "passed" untouched.
    assert row["status"] in ("fixed", "failed")
    if row["status"] == "fixed":
        out, _ = sf.read(str(base / "clean" / "clip.wav"))
        assert float(np.mean(np.abs(out) >= 0.999)) < float(np.mean(np.abs(clipped) >= 0.999))
    else:
        assert row["unfixable_reasons"]


def test_exact_duplicate_is_moved_not_deleted(sandbox, run_pipeline):
    _, inp = sandbox
    write(inp / "a.wav", speechlike(seed=3))
    shutil.copy(inp / "a.wav", inp / "b.wav")
    base, audit = run_pipeline(inp)
    statuses = sorted(r["status"] for r in audit.values())
    assert statuses.count("duplicate") == 1
    assert len(list((base / "_duplicates").glob("*.wav"))) == 1
    assert (inp / "a.wav").exists() and (inp / "b.wav").exists()      # originals untouched


def test_different_recordings_are_not_called_duplicates(sandbox, run_pipeline):
    _, inp = sandbox
    write(inp / "x.wav", speechlike(seed=4, f0=140))
    write(inp / "y.wav", speechlike(seed=5, f0=230))
    write(inp / "z.wav", speechlike(seed=6, f0=300))
    _, audit = run_pipeline(inp)
    assert all(r["status"] != "duplicate" for r in audit.values())


def test_reencoded_copy_is_flagged_as_near_duplicate(sandbox, run_pipeline):
    _, inp = sandbox
    a = speechlike(seed=7)
    write(inp / "orig.wav", a)
    write(inp / "quieter_copy.wav", (a * 0.5).astype(np.float32))       # same take, different level
    _, audit = run_pipeline(inp)
    assert sorted(r["status"] for r in audit.values()).count("duplicate") == 1


def test_broken_file_goes_to_error_folder_with_reason(sandbox, run_pipeline):
    _, inp = sandbox
    write(inp / "ok.wav", speechlike(seed=8))
    (inp / "broken.wav").write_bytes(b"RIFF....not really a wav")
    base, audit = run_pipeline(inp)
    assert audit["broken.wav"]["status"] == "failed"
    assert audit["broken.wav"]["unfixable_reasons"]
    assert (base / "_error" / "corrupt_or_unreadable" / "broken.wav").exists()
    assert not (base / "clean" / "broken.wav").exists()


def test_empty_and_tiny_files_do_not_crash_the_run(sandbox, run_pipeline):
    _, inp = sandbox
    write(inp / "ok.wav", speechlike(seed=9))
    (inp / "zero_bytes.wav").write_bytes(b"")
    write(inp / "tiny.wav", np.zeros(50, dtype=np.float32))
    base, audit = run_pipeline(inp)
    assert audit["ok.wav"]["status"] in ("passed", "fixed")
    assert audit["zero_bytes.wav"]["status"] == "failed"
    assert audit["tiny.wav"]["status"] == "failed"


def test_inputs_are_never_modified(sandbox, run_pipeline):
    _, inp = sandbox
    write(inp / "dc.wav", speechlike(seed=10) + 0.1)
    write(inp / "clip.wav", np.clip(speechlike(seed=11) * 6, -1, 1).astype(np.float32))
    write(inp / "sub" / "good.wav", speechlike(seed=12))
    (inp / "broken.wav").write_bytes(b"junk")
    before = tree_hashes(inp)
    run_pipeline(inp)
    assert tree_hashes(inp) == before


def test_subfolders_are_scanned_and_single_file_input_works(sandbox, run_pipeline):
    _, inp = sandbox
    write(inp / "a" / "b" / "deep.wav", speechlike(seed=13))
    _, audit = run_pipeline(inp, name="folder_run")
    assert "deep.wav" in audit
    _, audit2 = run_pipeline(inp / "a" / "b" / "deep.wav", name="single_run")
    assert list(audit2) == ["deep.wav"]


def test_stereo_and_other_sample_rates_are_handled(sandbox, run_pipeline):
    _, inp = sandbox
    mono = speechlike(seed=14)
    write(inp / "stereo.wav", np.stack([mono, mono * 0.8], axis=1))
    write(inp / "sr44.wav", speechlike(seed=15, seconds=3), sr=44100)
    _, audit = run_pipeline(inp)
    assert audit["stereo.wav"]["status"] in ("passed", "fixed", "failed")
    assert audit["sr44.wav"]["status"] in ("passed", "fixed", "failed")
    assert audit["stereo.wav"]["original_channels"] == 2


def test_dry_run_writes_no_audio(sandbox, run_pipeline):
    _, inp = sandbox
    write(inp / "dc.wav", speechlike(seed=16) + 0.1)
    base, _ = run_pipeline(inp, extra=["--dry-run"])
    assert not list((base / "clean").glob("*.wav")) if (base / "clean").exists() else True


def test_every_purpose_runs(sandbox, run_pipeline):
    from wavewash.config import PROFILES
    _, inp = sandbox
    write(inp / "good.wav", speechlike(seed=17))
    for purpose in PROFILES:
        _, audit = run_pipeline(inp, purpose=purpose, name=f"run_{purpose}")
        assert "good.wav" in audit, purpose
