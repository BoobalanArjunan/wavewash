"""Music Pack: dead channel, fake stereo, mono cancellation, digital dropouts, abrupt edges.

Known answers: every file is built here with exactly one planted defect (or none)."""
import dataclasses

import numpy as np
import pytest

from conftest import write
from wavewash import music_checks as mc
from wavewash.audio_fixer import fix_dead_channel
from wavewash.config import PROFILES
from wavewash.quality_checker import analyze_file

SR = 44100


def music(seconds=3.0, seed=0, stereo=True):
    """A chord-and-melody signal with a gentle fade in and out (so its edges are clean) and different content per channel."""
    rng = np.random.RandomState(seed)
    t = np.arange(int(seconds * SR)) / SR

    def voice(freqs, phase):
        return sum(np.sin(2 * np.pi * f * t + phase + 0.3 * np.sin(2 * np.pi * 5 * t)) / (i + 1) for i, f in enumerate(freqs))
    env = np.minimum(1.0, np.minimum(t / 0.2, (seconds - t) / 0.2)) * (0.05 + 0.95 * np.sin(2 * np.pi * 1 * t) ** 2)     # deep dips every second: loud bars and quiet gaps, like music
    hats = rng.randn(len(t)) * (np.sin(2 * np.pi * 4 * t) > 0.9) * 0.05
    left = 0.2 * voice([220, 277.2, 329.6, 440], 0.0) * env + hats
    right = 0.2 * voice([220, 277.2, 392.0, 523.3], 1.3) * env + 0.6 * np.roll(hats, 300)
    out = np.stack([left, right], axis=1) if stereo else left
    return out.astype(np.float32)


# ------------------------------------------------------------------ dead channel
def test_an_empty_channel_next_to_a_live_one_is_dead():
    x = music()
    x[:, 1] = 0.0
    assert mc.dead_channels(x) == [1]
    x2 = music()
    x2[:, 0] = 0.0
    assert mc.dead_channels(x2) == [0]


def test_healthy_mono_and_silent_files_have_no_dead_channel():
    assert mc.dead_channels(music()) == []
    assert mc.dead_channels(music(stereo=False)) == []
    assert mc.dead_channels(np.zeros((SR, 2), dtype=np.float32)) == []           # all silent: nothing to call dead


def test_dead_channel_threshold_sits_between_minus_80_and_minus_60_dbfs():
    x = music()
    quiet = np.random.RandomState(1).randn(len(x)).astype(np.float32)
    x[:, 1] = quiet * 3e-5 / quiet.std()                                          # rms 3e-5: empty
    assert mc.dead_channels(x) == [1]
    x[:, 1] = quiet * 3e-4 / quiet.std()                                          # rms 3e-4: quiet but alive
    assert mc.dead_channels(x) == []


def test_fix_copies_the_live_channel_and_says_it_invented_no_stereo():
    x = music()
    x[:, 0] = 0.0
    fixed, desc = fix_dead_channel(x)
    assert np.array_equal(fixed[:, 0], x[:, 1]) and np.array_equal(fixed[:, 1], x[:, 1])
    assert "channel 1" in desc and "channel 2" in desc and "no stereo information" in desc
    ok, none = fix_dead_channel(music())
    assert none is None and ok is not None


# ------------------------------------------------------------------ fake stereo and mono cancellation
def test_identical_channels_are_dual_mono_but_real_stereo_and_mono_are_not():
    m = music(stereo=False)
    assert mc.is_dual_mono(np.stack([m, m], axis=1))
    assert mc.is_dual_mono(np.stack([m, m + 1e-6], axis=1))                       # equal to within 1e-4 of the peak
    assert not mc.is_dual_mono(np.stack([m, m * 0.5], axis=1))
    assert not mc.is_dual_mono(music())
    assert not mc.is_dual_mono(m)
    assert not mc.is_dual_mono(np.zeros((100, 2), dtype=np.float32))


def test_mono_sum_loss_is_zero_for_identical_minus_3_for_unrelated_and_large_for_opposite_channels():
    m = music(stereo=False)
    assert mc.mono_sum_loss_db(np.stack([m, m], axis=1)) == pytest.approx(0.0, abs=0.01)
    noise = np.random.RandomState(0).randn(SR * 3, 2).astype(np.float32)
    assert mc.mono_sum_loss_db(noise) == pytest.approx(-3.0, abs=0.3)
    anti = mc.mono_sum_loss_db(np.stack([m, -m], axis=1))
    assert anti <= -50
    assert mc.mono_sum_loss_db(m) is None and mc.mono_sum_loss_db(np.zeros((100, 2))) is None
    partly = mc.mono_sum_loss_db(np.stack([m, -0.5 * m], axis=1))
    assert -12 < partly < -6                                                      # (1 - 0.5)^2 / ((1 + 0.25) * 2) -> about -9.0 dB


# ------------------------------------------------------------------ dropouts
def with_gap(x, start_s, ms, both=True):
    y = x.copy()
    a, b = int(start_s * SR), int(start_s * SR + ms / 1000 * SR)
    if y.ndim == 1:
        y[a:b] = 0.0
    else:
        y[a:b, :] = 0.0 if both else y[a:b, :] * [0.0, 1.0]
    return y


def test_a_stretch_of_exact_zeros_in_the_middle_of_the_music_is_a_dropout():
    d = mc.find_dropouts(with_gap(music(), 1.2, 50), SR)
    assert len(d) == 1 and d[0][0] == pytest.approx(1.2, abs=0.002) and d[0][1] == pytest.approx(50, abs=1)


def test_short_gaps_edges_and_quiet_passages_are_not_dropouts():
    assert mc.find_dropouts(with_gap(music(), 1.2, 10), SR) == []                 # shorter than 20 ms
    assert mc.find_dropouts(with_gap(music(), 1.2, 15), SR, min_ms=10)[0][1] == pytest.approx(15, abs=1)   # the limit is a setting
    edge = music()
    edge[:3000] = 0.0
    edge[-3000:] = 0.0
    assert mc.find_dropouts(edge, SR) == []                                       # silence at the start or end of a file
    quiet = music()
    quiet[int(1.0 * SR):int(1.5 * SR)] = np.random.RandomState(0).randn(int(0.5 * SR), 2).astype(np.float32) * 1e-4   # a noise floor, not zeros
    assert mc.find_dropouts(quiet, SR) == []


def test_zeros_are_not_a_dropout_unless_there_is_sound_on_both_sides():
    rng = np.random.RandomState(0)
    y = np.zeros((SR * 3, 2), dtype=np.float32)
    y[int(0.1 * SR):int(0.3 * SR)] = 0.2 * rng.randn(int(0.2 * SR), 2)                 # sound, then zeros ...
    y[int(2.0 * SR):] = 1e-4 * rng.randn(SR, 2)                                         # ... then only a noise floor: not music
    assert mc.find_dropouts(y, SR) == []
    y[int(2.0 * SR):int(2.2 * SR)] = 0.2 * rng.randn(int(0.2 * SR), 2)                  # sound again within 200 ms of the end of the zeros
    d = mc.find_dropouts(y, SR)
    assert len(d) == 1 and d[0][0] == pytest.approx(0.3, abs=0.002) and d[0][1] == pytest.approx(1700, abs=2)   # now it is a gap between sounds


def test_only_a_gap_in_all_channels_counts_and_mono_works():
    assert mc.find_dropouts(with_gap(music(), 1.2, 50, both=False), SR) == []
    m = with_gap(music(stereo=False), 1.0, 30)
    assert len(mc.find_dropouts(m, SR)) == 1


def test_two_gaps_are_both_reported_in_order():
    x = with_gap(with_gap(music(), 0.8, 40), 2.5, 80)
    d = mc.find_dropouts(x, SR)
    assert [round(a, 2) for a, _ in d] == [0.8, 2.5] and [round(b) for _, b in d] == [40, 80]


# ------------------------------------------------------------------ abrupt edges
def _tone(n=SR * 2, phase=1.0):
    t = np.arange(n) / SR
    return (0.5 * np.sin(2 * np.pi * 440 * t + phase)).astype(np.float32)


def test_a_cut_through_a_loud_sound_is_an_abrupt_edge_but_a_fade_is_not():
    assert mc.abrupt_edges(_tone()) == (True, True)
    assert mc.abrupt_edges(music()) == (False, False)
    fade = np.minimum(1.0, np.arange(SR * 2)[::-1] / (0.1 * SR)).astype(np.float32)
    assert mc.abrupt_edges(_tone() * fade) == (True, False)                            # faded end, abrupt start
    assert mc.abrupt_edges(_tone() * fade[::-1]) == (False, True)                      # faded start, abrupt end


def test_edge_checks_ignore_silence_and_tiny_inputs():
    assert mc.abrupt_edges(np.zeros(SR, dtype=np.float32)) == (False, False)
    assert mc.abrupt_edges(np.ones(8, dtype=np.float32)) == (False, False)
    assert mc.abrupt_edges(np.stack([_tone(), _tone()], axis=1)) == (True, True)       # stereo works the same way


# ------------------------------------------------------------------ in the quality check and the pipeline
@pytest.fixture
def music_profile():
    return PROFILES["music_gen"]


def test_a_clean_music_file_has_no_issues_and_no_music_warnings(tmp_path, music_profile):
    r = analyze_file(write(tmp_path / "a.wav", music(), SR), music_profile)
    assert r.issues == [] and not [w for w in r.warnings if w.split(":")[0] in ("FAKE_STEREO", "MONO_CANCELLATION", "DIGITAL_DROPOUT", "ABRUPT_EDGE")]
    assert r.dropout_count == 0 and -6 < r.mono_sum_loss_db < 0.1


def test_dead_channel_is_an_issue_and_the_other_music_findings_are_warnings(tmp_path, music_profile):
    x = music()
    x[:, 1] = 0.0
    r = analyze_file(write(tmp_path / "dead.wav", x, SR), music_profile)
    assert any(i.startswith("DEAD_CHANNEL: channel 2") for i in r.issues)
    m = music(stereo=False)
    r = analyze_file(write(tmp_path / "dual.wav", np.stack([m, m], axis=1), SR), music_profile)
    assert any(w.startswith("FAKE_STEREO") for w in r.warnings) and not any(i.startswith("DEAD") for i in r.issues)
    r = analyze_file(write(tmp_path / "anti.wav", np.stack([m, -0.9 * m], axis=1), SR), music_profile)
    assert any(w.startswith("MONO_CANCELLATION") for w in r.warnings) and r.mono_sum_loss_db < -6
    r = analyze_file(write(tmp_path / "gap.wav", with_gap(music(), 1.2, 60), SR), music_profile)
    assert any(w.startswith("DIGITAL_DROPOUT: 1 stretch") for w in r.warnings) and r.dropout_count == 1
    t = np.arange(SR * 3) / SR
    cut = np.stack([0.4 * np.sin(2 * np.pi * 330 * t + 1.0)] * 2, axis=1).astype(np.float32)
    cut[:, 1] *= 0.7
    r = analyze_file(write(tmp_path / "cut.wav", cut, SR), music_profile)
    assert any(w.startswith("ABRUPT_EDGE: the file starts and ends") for w in r.warnings)


def test_mono_cancellation_limit_is_a_setting(tmp_path):
    m = music(stereo=False)
    p = write(tmp_path / "x.wav", np.stack([m, -0.5 * m], axis=1), SR)           # about -9 dB
    strict = dataclasses.replace(PROFILES["music_gen"], music_min_mono_sum_db=-12.0)
    assert not any(w.startswith("MONO_CANCELLATION") for w in analyze_file(p, strict).warnings)
    assert any(w.startswith("MONO_CANCELLATION") for w in analyze_file(p, PROFILES["music_gen"]).warnings)


def test_only_the_music_profile_runs_the_music_checks(tmp_path):
    x = music()
    x[:, 1] = 0.0
    p = write(tmp_path / "dead.wav", x, SR)
    assert [k for k, v in PROFILES.items() if v.enable_music_checks] == ["music_gen"]
    for key in ("asr", "tts", "speaker_id", "custom"):
        assert not any(i.startswith("DEAD_CHANNEL") for i in analyze_file(p, PROFILES[key]).issues)


def test_pipeline_repairs_a_dead_channel_and_verifies_it(sandbox, run_pipeline):
    tmp, inp = sandbox
    x = music()
    x[:, 0] = 0.0
    write(inp / "dead.wav", x, SR)
    write(inp / "good.wav", music(seed=5), SR)
    base, audit = run_pipeline(inp, purpose="music_gen")
    rec = audit["dead.wav"]
    assert rec["status"] == "fixed" and any("filled empty channel 1 from channel 2" in f for f in rec["fixes_applied"])
    import soundfile as sf
    out, sr = sf.read(rec["final_destination"])
    assert out.shape[1] == 2 and np.abs(out[:, 0] - out[:, 1]).max() < 1e-3 and np.abs(out).max() > 0.01
    assert audit["good.wav"]["status"] in ("passed", "fixed")


def test_pipeline_keeps_files_with_music_warnings_and_lists_the_warnings(sandbox, run_pipeline):
    tmp, inp = sandbox
    m = music(stereo=False)
    write(inp / "dual.wav", np.stack([m, m], axis=1), SR)
    write(inp / "gap.wav", with_gap(music(seed=7), 1.2, 60), SR)
    base, audit = run_pipeline(inp, purpose="music_gen")
    assert audit["dual.wav"]["status"] in ("passed", "fixed") and any(w.startswith("FAKE_STEREO") for w in audit["dual.wav"]["warnings"])
    assert audit["gap.wav"]["status"] in ("passed", "fixed") and any(w.startswith("DIGITAL_DROPOUT") for w in audit["gap.wav"]["warnings"])


# ------------------------------------------------------------------ threshold edges (found by planting bugs)
def test_a_quiet_file_with_one_empty_channel_is_not_called_dead():
    x = music() * 0.002                       # whole file far below -60 dBFS: nothing is "live" next to the empty channel
    x[:, 1] = 0.0
    assert mc.dead_channels(x) == []


def test_channels_that_differ_a_little_are_not_dual_mono():
    left = music(stereo=False)
    peak = float(np.abs(left).max())
    assert mc.is_dual_mono(np.stack([left, left], axis=1))
    assert mc.is_dual_mono(np.stack([left, left + 1e-5 * peak], axis=1))                # float noise only
    assert not mc.is_dual_mono(np.stack([left, left + 1e-2 * peak * np.sign(left)], axis=1))   # a real 1% difference


def test_sound_within_200ms_counts_as_context_even_if_the_last_few_ms_are_a_fade():
    rng = np.random.RandomState(1)
    y = 1e-4 * rng.randn(SR * 2, 2).astype(np.float32)                                  # noise floor everywhere ...
    y[int(0.1 * SR):int(0.2 * SR)] = 0.2 * rng.randn(int(0.1 * SR), 2)                  # ... loud sound, then 100 ms of floor ...
    y[int(0.3 * SR):int(0.4 * SR)] = 0.0                                                # ... a gap of exact zeros ...
    y[int(0.4 * SR):int(0.5 * SR)] = 1e-4 * rng.randn(int(0.1 * SR), 2)
    y[int(0.5 * SR):int(0.6 * SR)] = 0.2 * rng.randn(int(0.1 * SR), 2)                  # ... and loud sound again 100 ms later
    d = mc.find_dropouts(y, SR)
    assert len(d) == 1 and d[0][0] == pytest.approx(0.3, abs=0.002)
