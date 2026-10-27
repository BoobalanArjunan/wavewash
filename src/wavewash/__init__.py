"""WaveWash — audio QC & data prep for ML training data."""

from .config import PROFILES, PurposeProfile
from .quality_checker import QualityReport, analyze_file
from .audio_fixer import apply_fix_pipeline, fix_declip, fix_dc_offset, fix_dead_channel
from .pipeline import process_file, load_audio, save_audio, find_audio_files

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "PROFILES",
    "PurposeProfile",
    "QualityReport",
    "analyze_file",
    "apply_fix_pipeline",
    "fix_declip",
    "fix_dc_offset",
    "fix_dead_channel",
    "process_file",
    "load_audio",
    "save_audio",
    "find_audio_files",
]
