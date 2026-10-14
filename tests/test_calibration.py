"""Guard rails for heuristics that are known to be weak.

The blind RT60 (reverb) estimate was measured against synthetic rooms of known reverb time and found
unable to separate dry from reverberant audio, so it must never reject a file on its own."""
import numpy as np

from conftest import speechlike, write


def test_reverb_estimate_never_rejects_a_clean_file(sandbox, run_pipeline):
    _, inp = sandbox
    for i in range(5):
        write(inp / f"clean{i}.wav", speechlike(seed=20 + i, f0=120 + 30 * i))
    for purpose in ("asr", "tts"):
        _, audit = run_pipeline(inp, purpose=purpose, name=f"rev_{purpose}")
        for name, row in audit.items():
            assert not any("HIGH_REVERB" in r for r in row["unfixable_reasons"]), (purpose, name)
