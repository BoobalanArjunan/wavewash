"""
audio_scanner.py
-----------------
Locates audio files no matter how the user points at them:
  - a single .wav file
  - a flat folder full of .wav files
  - a "main folder" containing many subfolders, each with .wav files, to any depth

Usage is deliberately forgiving: pass any path and it figures out what it is.
"""

from pathlib import Path
from typing import List
from .config import SUPPORTED_EXTENSIONS, EXTRA_READABLE_EXTENSIONS


def find_audio_files(input_path: str, include_extra_formats: bool = False) -> List[Path]:
    """
    Returns a sorted list of audio file paths found at input_path.

    - If input_path is a single file, returns [that file] (if it's a supported type).
    - If input_path is a directory, walks it recursively (any depth of subfolders)
      and returns every matching audio file found.
    """
    p = Path(input_path).expanduser().resolve()
    exts = set(SUPPORTED_EXTENSIONS)
    if include_extra_formats:
        exts |= EXTRA_READABLE_EXTENSIONS

    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")

    if p.is_file():
        if p.suffix.lower() in exts:
            return [p]
        raise ValueError(f"'{p}' is not a supported audio file "
                          f"(supported: {sorted(exts)})")

    if p.is_dir():
        found = []
        for ext in exts:
            found.extend(p.rglob(f"*{ext}"))
            found.extend(p.rglob(f"*{ext.upper()}"))
        # de-dup (case-insensitive glob can double up on some filesystems) and sort;
        # skip macOS resource-fork junk ("._name.wav", __MACOSX/) that unzipping leaves behind --
        # those aren't audio and would otherwise show up as "corrupt" files.
        found = sorted(f for f in set(found)
                       if not f.name.startswith("._") and "__MACOSX" not in f.parts)
        return found

    raise ValueError(f"Unrecognized path type: {p}")


def describe_input(files: List[Path], input_path: str) -> str:
    """Small human-readable summary of what was found, for logging."""
    p = Path(input_path)
    if p.is_file():
        return f"Single file provided: {p.name}"
    subfolders = {f.parent for f in files}
    return (f"Folder scan of '{p}': found {len(files)} audio file(s) "
            f"across {len(subfolders)} folder(s) (including subfolders).")
