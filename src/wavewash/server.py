"""
server.py
----------
Local backend for the WaveWash app. Runs on your machine, gives the
React frontend real access to your filesystem and the real pipeline.py --
nothing here is a browser sandbox. This is NOT meant to be exposed on the
open internet; it binds to localhost only and has no auth, by design, for
a single-user local tool.
"""

import os
import sys
import json
import shutil
import subprocess
import threading
import time
import uuid
import webbrowser
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import PROFILES, PurposeProfile
import dataclasses
import re

from starlette.requests import Request
from starlette.responses import JSONResponse

DEFAULT_PORT = 8420
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1", "testserver"}
# Dev-server origins (Vite) and the app's own origin(s) are the only ones allowed.
ALLOWED_ORIGIN_RE = re.compile(r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$")
_PROTECTED_DIRS = {"/etc", "/usr", "/bin", "/sbin", "/System", "/Library", "/var", "/boot", "/dev", "/proc", "/sys",
                   "C:\\Windows", "C:\\Program Files"}
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

app = FastAPI(title="WaveWash Backend")


@app.middleware("http")
async def local_only_guard(request: Request, call_next):
    """Defence for a no-auth localhost tool: reject requests whose Host header
    isn't loopback (DNS-rebinding) and state-changing requests from foreign
    web pages (CSRF). A random website open in your browser can otherwise
    talk to 127.0.0.1 and browse/read your disk through this API."""
    host = request.headers.get("host", "").rsplit(":", 1)[0] if not request.headers.get("host", "").startswith("[") \
        else request.headers.get("host", "").split("]")[0] + "]"
    if host not in ALLOWED_HOSTS:
        return JSONResponse({"detail": "Forbidden host"}, status_code=403)
    origin = request.headers.get("origin")
    if origin and not ALLOWED_ORIGIN_RE.match(origin):
        return JSONResponse({"detail": "Forbidden origin"}, status_code=403)
    if request.headers.get("sec-fetch-site") == "cross-site":
        return JSONResponse({"detail": "Cross-site requests are not allowed"}, status_code=403)
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=ALLOWED_ORIGIN_RE.pattern,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

PACKAGE_DIR = Path(__file__).resolve().parent
AUDIO_EXTS = {".wav", ".wave"}
EXTRA_AUDIO_EXTS = {".flac", ".mp3", ".m4a", ".ogg"}
JOBS_ROOT = Path(os.environ.get("WAVEWASH_HOME", str(Path.home() / ".wavewash"))) / "jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

# in-memory job registry -- fine for a local single-user app; nothing here
# needs to survive a server restart in a meaningful way (job folders on disk do)
JOBS: dict = {}
JOBS_LOCK = threading.Lock()


# ---------- filesystem browsing ----------

@app.get("/api/browse")
def browse(path: Optional[str] = None):
    """Lists a directory's contents so the frontend can offer a real folder
    picker without relying on a browser file-input dialog (which can't
    navigate arbitrary paths). Only ever reads -- never writes here."""
    target = Path(path).expanduser() if path else Path.home()
    if not target.exists():
        raise HTTPException(404, f"Path does not exist: {target}")
    if not target.is_dir():
        raise HTTPException(400, f"Not a directory: {target}")

    entries = []
    try:
        for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if item.name.startswith("."):
                continue
            try:
                is_dir = item.is_dir()
                wav_count = None
                if is_dir:
                    try:   # direct children only -- keeps browsing fast on big folders
                        wav_count = sum(1 for c in item.iterdir()
                                        if c.is_file() and c.suffix.lower() in AUDIO_EXTS)
                    except Exception:
                        wav_count = None
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "is_dir": is_dir,
                    "wav_count": wav_count,
                    "ext": item.suffix.lower() if not is_dir else None,
                })
            except PermissionError:
                continue
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {target}")

    return {
        "current_path": str(target),
        "parent_path": str(target.parent) if target.parent != target else None,
        "entries": entries,
    }


@app.get("/api/wav-count")
def wav_count(path: str, extra: bool = False):
    """Recursive audio-file count for a chosen input path (folder or single file)."""
    p = Path(path).expanduser()
    if not p.exists():
        raise HTTPException(404, "That path doesn't exist.")
    exts = AUDIO_EXTS | (EXTRA_AUDIO_EXTS if extra else set())
    if p.is_file():
        return {"count": 1 if p.suffix.lower() in exts else 0, "is_file": True}
    count = 0
    for root, dirs, files in os.walk(p):
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        count += sum(1 for f in files if not f.startswith("._") and Path(f).suffix.lower() in exts)
    return {"count": count, "is_file": False}


@app.post("/api/open-folder")
def open_folder(body: dict):
    """Reveal a folder in Finder / Explorer / the file manager (local convenience only)."""
    p = Path(str(body.get("path", ""))).expanduser()
    if not p.exists():
        raise HTTPException(404, "Folder doesn't exist.")
    target = p if p.is_dir() else p.parent
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        elif sys.platform.startswith("win"):
            os.startfile(str(target))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception as e:
        raise HTTPException(500, f"Could not open the folder: {e}")
    return {"opened": str(target)}


# ---------- purpose profiles ----------

@app.get("/api/purposes")
def get_purposes():
    result = []
    for key, profile in PROFILES.items():
        d = dataclasses.asdict(profile)
        result.append({"key": key, **d})
    return result


class ThresholdOverride(BaseModel):
    purpose: str
    overrides: dict


@app.post("/api/purposes/preview")
def preview_profile(body: ThresholdOverride):
    """Returns what a profile would look like with the given overrides applied,
    without saving anything -- lets the frontend show a live preview in the
    advanced-settings panel before a run starts."""
    if body.purpose not in PROFILES:
        raise HTTPException(404, "Unknown purpose")
    base = dataclasses.asdict(PROFILES[body.purpose])
    base.update(body.overrides)
    return base


# ---------- running jobs ----------

class RunRequest(BaseModel):
    input_path: str
    purpose: str
    output_path: Optional[str] = None
    include_extra_formats: bool = False
    skip_dedup: bool = False
    dedup_mode: str = "near"          # off | exact | near
    use_real_vad: bool = False
    review_sample_size: int = 5
    transcripts_path: Optional[str] = None
    expected_language: Optional[str] = None
    asr_model: str = "none"
    metadata_path: Optional[str] = None
    make_split: bool = False
    train_ratio: float = 0.9
    threshold_overrides: Optional[dict] = None


def _write_overrides_file(purpose: str, overrides: dict, job_dir: Path) -> Path:
    """Writes threshold overrides to JSON; config.py (patched below) reads this
    via an env var and applies it to the chosen profile before pipeline.main() runs."""
    overrides_path = job_dir / "_overrides.json"
    overrides_path.write_text(json.dumps({"purpose": purpose, "overrides": overrides}))
    return overrides_path


def _run_job(job_id: str, req: RunRequest):
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(req.output_path).expanduser() if req.output_path else job_dir
    output_path = str(base_dir / "clean")   # siblings _error/ _duplicates/ _logs/ land in base_dir

    cmd = [sys.executable, "-m", "wavewash.pipeline",
           "--input", req.input_path,
           "--output", output_path,
           "--purpose", req.purpose,
           "--review-sample-size", str(req.review_sample_size),
           "--dedup-mode", req.dedup_mode if not req.skip_dedup else "off"]

    if req.include_extra_formats:
        cmd.append("--include-extra-formats")
    if req.skip_dedup:
        cmd.append("--skip-dedup")
    if req.use_real_vad:
        cmd.append("--use-real-vad")
    if req.transcripts_path:
        cmd += ["--transcripts", req.transcripts_path]
    if req.expected_language:
        cmd += ["--expected-language", req.expected_language]
    if req.asr_model != "none":
        cmd += ["--asr-model", req.asr_model]
    if req.metadata_path:
        cmd += ["--metadata", req.metadata_path]
    if req.make_split:
        cmd += ["--make-split", "--train-ratio", str(req.train_ratio)]

    env = os.environ.copy()
    if req.threshold_overrides:
        overrides_path = _write_overrides_file(req.purpose, req.threshold_overrides, job_dir)
        env["WAVEWASH_CONFIG_OVERRIDES"] = str(overrides_path)

    log_path = job_dir / "stdout.log"

    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["output_path"] = output_path
        JOBS[job_id]["base_dir"] = str(base_dir)
        JOBS[job_id]["log_path"] = str(log_path)

    try:
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                cmd, cwd=str(job_dir),
                stdout=logf, stderr=subprocess.STDOUT, text=True, env=env,
            )
            with JOBS_LOCK:
                JOBS[job_id]["pid"] = proc.pid
            proc.wait()
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done" if proc.returncode == 0 else "error"
            JOBS[job_id]["return_code"] = proc.returncode
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)


@app.post("/api/run")
def run_pipeline(req: RunRequest):
    inp = Path(req.input_path).expanduser().resolve()
    req.input_path = str(inp)
    for label, opt in (("transcripts_path", req.transcripts_path), ("metadata_path", req.metadata_path)):
        if opt:
            setattr(req, label, str(Path(opt).expanduser().resolve()))
    if not inp.exists():
        raise HTTPException(400, f"Input path doesn't exist: {inp}")
    if req.purpose not in PROFILES:
        raise HTTPException(400, f"Unknown purpose: {req.purpose}")
    if req.dedup_mode not in ("off", "exact", "near"):
        raise HTTPException(400, "dedup_mode must be off, exact or near")
    for label, opt in (("transcripts", req.transcripts_path), ("metadata", req.metadata_path)):
        if opt and not Path(opt).expanduser().is_file():
            raise HTTPException(400, f"The {label} file doesn't exist: {opt}")
    if req.output_path:
        out = Path(req.output_path).expanduser().resolve()
        req.output_path = str(out)
        if out == Path(out.anchor) or str(out) in _PROTECTED_DIRS:
            raise HTTPException(400, "Choose a normal data folder for the output, not a system folder.")
        inp_r = inp
        if out == inp_r or inp_r in out.parents:
            raise HTTPException(400, "The output folder can't be the input folder or inside it "
                                     "(the next run would re-scan its own results). "
                                     "Pick a different output folder.")
        if out in inp_r.parents:
            raise HTTPException(400, "The output folder can't contain the input folder.")
    job_id = str(uuid.uuid4())[:8]
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "starting", "request": (req.model_dump() if hasattr(req, "model_dump") else req.dict()), "created": time.time()}

    thread = threading.Thread(target=_run_job, args=(job_id, req), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")

    result = dict(job)

    log_path = job.get("log_path")
    log_tail = ""
    file_count_seen = 0
    if log_path and Path(log_path).exists():
        text = Path(log_path).read_text(errors="ignore")
        lines = text.splitlines()
        log_tail = "\n".join(lines[-30:])
        file_count_seen = sum(1 for l in lines if "PASSED" in l or "FIXED" in l or "FAILED" in l)

    result["log_tail"] = log_tail
    result["files_processed_so_far"] = file_count_seen
    return result


@app.get("/api/jobs/{job_id}/results")
def get_job_results(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    if job.get("status") not in ("done", "error"):
        raise HTTPException(202, "Job still running")

    output_path = Path(job["output_path"])
    logs_dir = output_path.parent / "_logs"

    audit_rows = []
    csv_files = sorted(logs_dir.glob("audit_log_*.csv")) if logs_dir.exists() else []
    if csv_files:
        import csv as csv_module
        with open(csv_files[-1]) as f:
            audit_rows = list(csv_module.DictReader(f))

    summary_files = sorted(logs_dir.glob("summary_*.md")) if logs_dir.exists() else []
    summary_text = summary_files[-1].read_text() if summary_files else ""

    diversity_path = logs_dir / "diversity_report.md"
    diversity_text = diversity_path.read_text() if diversity_path.exists() else ""

    status_counts = {}
    for row in audit_rows:
        s = row.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    return {
        "status_counts": status_counts,
        "files": audit_rows,
        "summary_markdown": summary_text,
        "diversity_markdown": diversity_text,
        "output_path": str(output_path),
        "base_dir": str(output_path.parent),
    }


@app.get("/api/jobs/{job_id}/download")
def download_results(job_id: str):
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(400, "Bad job id")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")

    job_dir = JOBS_ROOT / job_id
    zip_path = job_dir / "results.zip"
    output_path = Path(job["output_path"])
    base_dir = output_path.parent

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder_name in ["clean", "_error", "_duplicates", "_logs", "_review_sample"]:
            folder = base_dir / folder_name if folder_name != "clean" else output_path
            if folder.exists():
                for f in folder.rglob("*"):
                    if f.is_file():
                        zf.write(f, arcname=f.relative_to(base_dir.parent))

    return FileResponse(zip_path, filename=f"wavewash_results_{job_id}.zip")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------- serve the bundled frontend (one command, one port) ----------
FRONTEND_DIST = PACKAGE_DIR / "static"
if (FRONTEND_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT, open_browser: bool = True):
    import uvicorn
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("Refusing to bind to a non-loopback address: this tool has no "
              "authentication and can read/write your disk.")
        raise SystemExit(2)
    if open_browser and (FRONTEND_DIST / "index.html").exists():
        def _open():
            time.sleep(1.2)
            webbrowser.open(f"http://127.0.0.1:{port}")
        threading.Thread(target=_open, daemon=True).start()
    elif not (FRONTEND_DIST / "index.html").exists():
        print("Note: bundled UI not found (source checkout without a frontend build). "
              "API still runs; see README for building the UI.")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    serve()
