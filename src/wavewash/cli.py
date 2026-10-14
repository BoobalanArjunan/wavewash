"""wavewash command line entry point.

    wavewash app      # launch the web UI on http://127.0.0.1:8420
    wavewash run ...  # run the QC pipeline from the terminal (same flags as before)
    wavewash doctor   # check that the install works and show optional extras
"""
import argparse
import importlib
import importlib.util
import platform
import sys

from . import __version__

_REQUIRED = ["numpy", "scipy", "soundfile", "librosa", "pyloudnorm", "sklearn", "tqdm",
             "fastapi", "uvicorn", "pydantic"]
_OPTIONAL = {
    "langdetect": "language check for transcripts",
    "matplotlib": "plots in reports",
    "torch": "Silero VAD (--use-real-vad)",
    "whisper": "Whisper WER check (--asr-model whisper)",
}


def _doctor() -> int:
    print(f"wavewash {__version__}  |  Python {platform.python_version()}  |  {platform.platform()}")
    bad = 0
    print("\nRequired:")
    for m in _REQUIRED:
        try:
            mod = importlib.import_module(m)
            print(f"  ok       {m} {getattr(mod, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"  MISSING  {m}  ({e})")
    print("\nOptional extras (not needed for the default checks):")
    for m, why in _OPTIONAL.items():
        found = importlib.util.find_spec(m) is not None
        print(f"  {'ok     ' if found else 'absent '}  {m:<11} {why}")
    try:
        import soundfile as sf
        print(f"\nlibsndfile: {sf.__libsndfile_version__}")
    except Exception:  # noqa: BLE001
        pass
    from .server import FRONTEND_DIST
    ui = (FRONTEND_DIST / "index.html").exists()
    print(f"Bundled web UI: {'present' if ui else 'MISSING'}")
    bad += 0 if ui else 1
    print("\nResult:", "all good" if not bad else f"{bad} problem(s) found")
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="wavewash", description="WaveWash — audio QC & data prep for ML training data")
    parser.add_argument("--version", action="version", version=f"wavewash {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    pa = sub.add_parser("app", help="launch the web UI (local only)")
    pa.add_argument("--port", type=int, default=8420)
    pa.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    sub.add_parser("run", help="run the QC pipeline (pass --help after 'run' for its flags)", add_help=False)
    sub.add_parser("doctor", help="check the installation")

    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    if argv[0] == "run":
        from .pipeline import main as run_main
        run_main(argv[1:])
        return 0
    args = parser.parse_args(argv)
    if args.cmd == "doctor":
        return _doctor()
    if args.cmd == "app":
        from .server import serve
        serve(port=args.port, open_browser=not args.no_browser)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
