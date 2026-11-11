#!/usr/bin/env python3
"""Fetch daily download and engagement stats across the 10-tool ComposerMind PyPI suite,
Hugging Face, Zenodo, and GitHub, then send a consolidated HTML digest via Resend.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

TARGET_EMAIL = os.environ.get("NOTIFICATION_EMAIL", "boobalan_arjunan@composermind.com")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", "ComposerMind Suite Stats <onboarding@resend.dev>")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
USER_AGENT = "ComposerMind-Suite-Analytics/2.0 (https://github.com/BoobalanArjunan/wavewash)"

# All 10 live PyPI packages in the ComposerMind Open Audio ML Suite
SUITE_PACKAGES: list[tuple[str, str]] = [
    ("wavewash", "BoobalanArjunan/wavewash"),
    ("splitsafe", "BoobalanArjunan/splitsafe"),
    ("bandcheck", "BoobalanArjunan/Bandcheck"),
    ("stemcheck", "BoobalanArjunan/Stemcheck"),
    ("phasecheck", "BoobalanArjunan/Phasecheck"),
    ("driftcheck-audio", "BoobalanArjunan/Driftcheck"),
    ("mixforge-audio", "BoobalanArjunan/Mix-Forge"),
    ("tonicfind", "BoobalanArjunan/Tonicfind"),
    ("tempocheck", "BoobalanArjunan/Tempocheck"),
    ("turnscope", "BoobalanArjunan/Turnscope"),
]

HF_AUTHOR = "boobalanit"
HF_DATASET_ID = "boobalanit/ComposerMind-OpenAudio-48k"
HF_SPACES = [
    ("boobalanit/wavewash", "WaveWash Space"),
    ("boobalanit/splitsafe", "SplitSafe Space"),
]
HF_MODELS = [
    ("boobalanit/ACE-Step-1.5-SouthIndiaParaiLoRA", "SouthIndiaParai LoRA"),
]
ZENODO_RECORD_ID = "24050283"


def http_get_json(
    url: str,
    timeout: int = 20,
    extra_headers: dict[str, str] | None = None,
    retries: int = 2,
) -> tuple[int, Any, str | None]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return resp.status, json.loads(raw), None
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504) and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:180]
            except Exception:
                pass
            return e.code, None, f"HTTP {e.code}: {body}"
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), socket.gaierror):
                return -1, None, str(e)
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return -1, None, str(e)
        except Exception as e:
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return -1, None, str(e)
    return -1, None, "Unreachable"


def fetch_pypi_package_stats(pkg: str) -> dict[str, Any]:
    """Fetch version metadata, overall history, and recent counts for one PyPI package."""
    version = "-"
    first_upload = "-"
    latest_upload = "-"
    code, pypi_meta, err = http_get_json(f"https://pypi.org/pypi/{pkg}/json")
    if code == 200 and isinstance(pypi_meta, dict):
        info = pypi_meta.get("info", {})
        version = str(info.get("version") or "-")
        releases = pypi_meta.get("releases", {})
        all_times: list[str] = []
        latest_times: list[str] = []
        for ver, files in releases.items():
            for f in files:
                ut = f.get("upload_time_iso_8601") or f.get("upload_time")
                if ut:
                    all_times.append(str(ut))
                    if ver == version:
                        latest_times.append(str(ut))
        if all_times:
            first_upload = min(all_times)[:10]
        if latest_times:
            latest_upload = min(latest_times)[:10]
        elif all_times:
            latest_upload = max(all_times)[:10]
    else:
        print(f"[WARN] PyPI metadata failed for {pkg}: {err}")

    time.sleep(0.25)

    total_without_mirrors = 0
    total_with_mirrors = 0
    without_by_date: dict[str, int] = {}
    with_by_date: dict[str, int] = {}
    max_date: str | None = None
    min_date: str | None = None
    peak_without_date = "-"
    peak_without_val = 0

    code_o, overall, err_o = http_get_json(f"https://pypistats.org/api/packages/{pkg}/overall")
    if code_o == 200 and isinstance(overall, dict):
        rows = overall.get("data", [])
        wo_rows = [r for r in rows if r.get("category") == "without_mirrors"]
        wm_rows = [r for r in rows if r.get("category") == "with_mirrors"]
        for r in wo_rows:
            d = r.get("date")
            dl = int(r.get("downloads") or 0)
            if d:
                without_by_date[d] = without_by_date.get(d, 0) + dl
            total_without_mirrors += dl
        for r in wm_rows:
            d = r.get("date")
            dl = int(r.get("downloads") or 0)
            if d:
                with_by_date[d] = with_by_date.get(d, 0) + dl
            total_with_mirrors += dl
        if without_by_date:
            all_dates = sorted(without_by_date.keys())
            min_date, max_date = all_dates[0], all_dates[-1]
            peak_without_date = max(without_by_date, key=lambda k: without_by_date[k])
            peak_without_val = without_by_date[peak_without_date]
    else:
        print(f"[WARN] PyPI overall stats failed for {pkg}: {err_o}")

    time.sleep(0.35)

    last_day = 0
    last_week = 0
    code_r, recent, err_r = http_get_json(f"https://pypistats.org/api/packages/{pkg}/recent")
    if code_r == 200 and isinstance(recent, dict):
        rdata = recent.get("data", {})
        last_day = int(rdata.get("last_day") or 0)
        last_week = int(rdata.get("last_week") or 0)
    else:
        print(f"[WARN] PyPI recent stats failed for {pkg}: {err_r}")
        if max_date and without_by_date:
            last_day = without_by_date.get(max_date, 0)
            cutoff = (datetime.strptime(max_date, "%Y-%m-%d") - timedelta(days=6)).strftime("%Y-%m-%d")
            last_week = sum(v for d, v in without_by_date.items() if d >= cutoff)

    return {
        "pkg": pkg,
        "version": version,
        "first_upload": first_upload,
        "latest_upload": latest_upload,
        "without_mirrors_total": total_without_mirrors,
        "with_mirrors_total": total_with_mirrors,
        "last_day": last_day,
        "last_week": last_week,
        "min_date": min_date,
        "max_date": max_date,
        "peak_date": peak_without_date,
        "peak_direct": peak_without_val,
        "without_by_date": without_by_date,
        "with_by_date": with_by_date,
    }


def fetch_github_stats() -> dict[str, dict[str, Any]]:
    gh_headers: dict[str, str] = {}
    if GITHUB_TOKEN:
        gh_headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    code, repos, err = http_get_json(
        "https://api.github.com/users/BoobalanArjunan/repos?per_page=100",
        extra_headers=gh_headers,
    )
    out: dict[str, dict[str, Any]] = {}
    if code == 200 and isinstance(repos, list):
        for r in repos:
            full = str(r.get("full_name") or "").lower()
            out[full] = {
                "stars": int(r.get("stargazers_count") or 0),
                "forks": int(r.get("forks_count") or 0),
                "open_issues": int(r.get("open_issues_count") or 0),
                "pushed_at": str(r.get("pushed_at") or "")[:10],
            }
    else:
        print(f"[WARN] GitHub repos API failed: {err}")
    return out


def fetch_huggingface_stats() -> dict[str, Any]:
    dataset_stats = {
        "id": HF_DATASET_ID,
        "downloads_30d": 0,
        "downloads_all_time": 0,
        "likes": 0,
        "ok": False,
    }
    code, ds, err = http_get_json(
        f"https://huggingface.co/api/datasets/{HF_DATASET_ID}"
        "?expand[]=downloads&expand[]=downloadsAllTime&expand[]=likes&expand[]=trendingScore"
    )
    if code == 200 and isinstance(ds, dict):
        dataset_stats.update({
            "downloads_30d": int(ds.get("downloads") or 0),
            "downloads_all_time": int(ds.get("downloadsAllTime") or ds.get("downloads") or 0),
            "likes": int(ds.get("likes") or 0),
            "ok": True,
        })
    else:
        print(f"[WARN] HF dataset API failed: {err}")

    spaces: list[dict[str, Any]] = []
    for sid, label in HF_SPACES:
        sc, sdata, serr = http_get_json(f"https://huggingface.co/api/spaces/{sid}")
        if sc == 200 and isinstance(sdata, dict):
            spaces.append({
                "id": sid,
                "label": label,
                "likes": int(sdata.get("likes") or 0),
                "stage": str((sdata.get("runtime") or {}).get("stage") or "Unknown"),
                "trending": int(sdata.get("trendingScore") or 0),
            })
        else:
            print(f"[WARN] HF space API failed for {sid}: {serr}")
            spaces.append({"id": sid, "label": label, "likes": 0, "stage": "Unavailable", "trending": 0})

    models: list[dict[str, Any]] = []
    for mid, label in HF_MODELS:
        mc, mdata, merr = http_get_json(
            f"https://huggingface.co/api/models/{mid}"
            "?expand[]=downloads&expand[]=downloadsAllTime&expand[]=likes"
        )
        if mc == 200 and isinstance(mdata, dict):
            models.append({
                "id": mid,
                "label": label,
                "downloads_30d": int(mdata.get("downloads") or 0),
                "downloads_all_time": int(mdata.get("downloadsAllTime") or mdata.get("downloads") or 0),
                "likes": int(mdata.get("likes") or 0),
            })
        else:
            print(f"[WARN] HF model API failed for {mid}: {merr}")

    return {"dataset": dataset_stats, "spaces": spaces, "models": models}


def fetch_zenodo_stats() -> dict[str, Any]:
    code, zdata, err = http_get_json(f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}")
    if code == 200 and isinstance(zdata, dict):
        s = zdata.get("stats") or {}
        return {
            "ok": True,
            "record_id": ZENODO_RECORD_ID,
            "downloads": int(s.get("downloads") or 0),
            "unique_downloads": int(s.get("unique_downloads") or 0),
            "views": int(s.get("views") or 0),
            "unique_views": int(s.get("unique_views") or 0),
        }
    print(f"[WARN] Zenodo API failed: {err}")
    return {
        "ok": False,
        "record_id": ZENODO_RECORD_ID,
        "downloads": 0,
        "unique_downloads": 0,
        "views": 0,
        "unique_views": 0,
    }


def fetch_stats() -> dict[str, Any]:
    gh_map = fetch_github_stats()

    tools: list[dict[str, Any]] = []
    daily_suite_direct: dict[str, int] = {}
    daily_suite_mirrors: dict[str, int] = {}
    wavewash_daily_direct: dict[str, int] = {}
    all_max_dates: list[str] = []

    for pkg, repo in SUITE_PACKAGES:
        p = fetch_pypi_package_stats(pkg)
        gh = gh_map.get(repo.lower(), {"stars": 0, "forks": 0, "open_issues": 0, "pushed_at": "-"})
        p["repo"] = repo
        p["github"] = gh
        tools.append(p)

        if p["max_date"]:
            all_max_dates.append(p["max_date"])
        for d, val in p["without_by_date"].items():
            daily_suite_direct[d] = daily_suite_direct.get(d, 0) + val
            if pkg == "wavewash":
                wavewash_daily_direct[d] = val
        for d, val in p["with_by_date"].items():
            daily_suite_mirrors[d] = daily_suite_mirrors.get(d, 0) + val
        time.sleep(0.25)

    # Sort leaderboard by direct installs descending, then with_mirrors descending
    tools_sorted = sorted(
        tools,
        key=lambda x: (x["without_mirrors_total"], x["with_mirrors_total"]),
        reverse=True,
    )

    wavewash_stats = next((t for t in tools if t["pkg"] == "wavewash"), tools[0])
    all_seen_dates = sorted(set(daily_suite_direct.keys()) | set(daily_suite_mirrors.keys()) | set(all_max_dates))
    pypi_cutoff_date = all_seen_dates[-1] if all_seen_dates else "-"
    if pypi_cutoff_date != "-":
        cutoff_dt = datetime.strptime(pypi_cutoff_date, "%Y-%m-%d")
        recent_dates = [(cutoff_dt - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    else:
        recent_dates = []
    trend_7d = [
        {
            "date": d,
            "wavewash_direct": wavewash_daily_direct.get(d, 0),
            "suite_direct": daily_suite_direct.get(d, 0),
            "suite_mirrors": daily_suite_mirrors.get(d, 0),
        }
        for d in recent_dates
    ]

    hf_stats = fetch_huggingface_stats()
    zenodo_stats = fetch_zenodo_stats()

    suite_without_total = sum(t["without_mirrors_total"] for t in tools)
    suite_with_total = sum(t["with_mirrors_total"] for t in tools)
    suite_last_day = sum(t["last_day"] for t in tools)
    suite_last_week = sum(t["last_week"] for t in tools)
    suite_gh_stars = sum(t["github"]["stars"] for t in tools)
    suite_gh_forks = sum(t["github"]["forks"] for t in tools)

    now_utc = datetime.now(timezone.utc)
    now_ist = now_utc + timedelta(hours=5, minutes=30)

    return {
        "tools": tools_sorted,
        "wavewash": wavewash_stats,
        "suite": {
            "without_mirrors_total": suite_without_total,
            "with_mirrors_total": suite_with_total,
            "last_day": suite_last_day,
            "last_week": suite_last_week,
            "gh_stars": suite_gh_stars,
            "gh_forks": suite_gh_forks,
            "pypi_cutoff_date": pypi_cutoff_date,
        },
        "trend_7d": trend_7d,
        "hf": hf_stats,
        "zenodo": zenodo_stats,
        "date_utc": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "date_ist": now_ist.strftime("%Y-%m-%d %H:%M IST"),
    }


def build_html_email(stats: dict[str, Any]) -> str:
    suite = stats["suite"]
    ww = stats["wavewash"]
    hf_ds = stats["hf"]["dataset"]
    zen = stats["zenodo"]

    tool_rows = ""
    for rank, t in enumerate(stats["tools"], start=1):
        pkg_esc = html.escape(t["pkg"])
        ver_esc = html.escape(t["version"])
        repo_esc = html.escape(t["repo"])
        peak_str = f"{t['peak_date']} ({t['peak_direct']})" if t["peak_date"] != "-" else "-"
        highlight_bg = "#1e293b" if t["pkg"] == "wavewash" else "#18181b"
        badge = " <span style='font-size:10px;background:#0284c7;color:#fff;padding:1px 5px;border-radius:4px;'>FLAGSHIP</span>" if t["pkg"] == "wavewash" else ""
        tool_rows += f"""
        <tr style="background: {highlight_bg};">
          <td style="padding: 9px 8px; border-bottom: 1px solid #27272a; text-align: center; color: #94a3b8; font-weight: 600;">{rank}</td>
          <td style="padding: 9px 8px; border-bottom: 1px solid #27272a;">
            <a href="https://pypi.org/project/{pkg_esc}/" style="color: #38bdf8; text-decoration: none; font-weight: 700;">{pkg_esc}</a>{badge}
            <div style="font-size: 10px; color: #71717a;">v{ver_esc} &bull; <a href="https://github.com/{repo_esc}" style="color: #94a3b8; text-decoration: none;">GH {t['github']['stars']}&#9733;/{t['github']['forks']}&#9913;</a></div>
          </td>
          <td style="padding: 9px 8px; border-bottom: 1px solid #27272a; text-align: right; font-weight: 700; color: #f4f4f5; font-family: monospace;">{t['without_mirrors_total']:,}</td>
          <td style="padding: 9px 8px; border-bottom: 1px solid #27272a; text-align: right; color: #a1a1aa; font-family: monospace;">{t['with_mirrors_total']:,}</td>
          <td style="padding: 9px 8px; border-bottom: 1px solid #27272a; text-align: right; color: #4ade80; font-weight: 600; font-family: monospace;">+{t['last_day']:,}</td>
          <td style="padding: 9px 8px; border-bottom: 1px solid #27272a; text-align: right; color: #c084fc; font-family: monospace;">{t['last_week']:,}</td>
          <td style="padding: 9px 8px; border-bottom: 1px solid #27272a; text-align: right; color: #94a3b8; font-size: 11px; font-family: monospace;">{html.escape(peak_str)}</td>
        </tr>
        """

    trend_rows = ""
    for row in reversed(stats["trend_7d"]):
        trend_rows += f"""
        <tr>
          <td style="padding: 8px 12px; border-bottom: 1px solid #27272a; color: #a1a1aa; font-family: monospace;">{html.escape(row['date'])}</td>
          <td style="padding: 8px 12px; border-bottom: 1px solid #27272a; text-align: right; font-weight: 700; color: #38bdf8; font-family: monospace;">{row['wavewash_direct']:,}</td>
          <td style="padding: 8px 12px; border-bottom: 1px solid #27272a; text-align: right; font-weight: 700; color: #4ade80; font-family: monospace;">{row['suite_direct']:,}</td>
          <td style="padding: 8px 12px; border-bottom: 1px solid #27272a; text-align: right; color: #94a3b8; font-family: monospace;">{row['suite_mirrors']:,}</td>
        </tr>
        """

    space_items = ""
    for sp in stats["hf"]["spaces"]:
        sid_esc = html.escape(sp["id"])
        lbl_esc = html.escape(sp["label"])
        st_esc = html.escape(sp["stage"])
        space_items += f"<div style='margin-top:4px;font-size:12px;color:#d4d4d8;'><a href='https://huggingface.co/spaces/{sid_esc}' style='color:#fbbf24;text-decoration:none;font-weight:600;'>{lbl_esc}</a>: {sp['likes']} &#10084;&#65039; &bull; {st_esc}</div>"

    model_items = ""
    for m in stats["hf"]["models"]:
        mid_esc = html.escape(m["id"])
        lbl_esc = html.escape(m["label"])
        model_items += f"<div style='margin-top:4px;font-size:11px;color:#a1a1aa;'><a href='https://huggingface.co/models/{mid_esc}' style='color:#94a3b8;text-decoration:none;'>{lbl_esc}</a>: {m['downloads_all_time']} dl / {m['likes']} &#10084;&#65039;</div>"

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ComposerMind 10-Tool Suite &amp; WaveWash Daily Analytics</title>
</head>
<body style="margin: 0; padding: 20px; background-color: #09090b; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #f4f4f5;">
  <div style="max-width: 720px; margin: 0 auto; background: #18181b; border: 1px solid #27272a; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">

    <!-- Header -->
    <div style="background: linear-gradient(135deg, #0284c7, #4f46e5); padding: 24px; text-align: center;">
      <h1 style="margin: 0; font-size: 22px; font-weight: 800; letter-spacing: -0.4px; color: #ffffff;">&#127754; WaveWash &amp; ComposerMind Suite Analytics</h1>
      <p style="margin: 6px 0 0 0; font-size: 13px; color: rgba(255,255,255,0.88);">Consolidated Daily Report &bull; 10 PyPI Tools + Hugging Face + Zenodo</p>
      <div style="display: inline-block; margin-top: 10px; background: rgba(0,0,0,0.28); padding: 4px 12px; border-radius: 20px; font-size: 11px; color: #ffffff; font-family: monospace;">
        {html.escape(stats['date_utc'])} &bull; {html.escape(stats['date_ist'])} &bull; PyPI Cutoff: {html.escape(suite['pypi_cutoff_date'])}
      </div>
    </div>

    <div style="padding: 22px;">

      <!-- Top Summary Metric Cards -->
      <table style="width: 100%; border-collapse: separate; border-spacing: 10px; margin: -10px -10px 14px -10px;">
        <tr>
          <td style="background: #27272a; border-radius: 8px; padding: 14px; text-align: center; width: 50%; border: 1px solid #3f3f46;">
            <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px;">10-Tool PyPI Direct Installs</div>
            <div style="font-size: 30px; font-weight: 800; color: #38bdf8; margin: 4px 0;">{suite['without_mirrors_total']:,}</div>
            <div style="font-size: 11px; color: #a1a1aa;">+{suite['last_day']:,} yesterday &bull; {suite['last_week']:,} last 7d &bull; <strong>{suite['with_mirrors_total']:,}</strong> w/ mirrors</div>
          </td>
          <td style="background: #27272a; border-radius: 8px; padding: 14px; text-align: center; width: 50%; border: 1px solid #3f3f46;">
            <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px;">WaveWash Flagship (v{html.escape(ww['version'])})</div>
            <div style="font-size: 30px; font-weight: 800; color: #4ade80; margin: 4px 0;">{ww['without_mirrors_total']:,}</div>
            <div style="font-size: 11px; color: #a1a1aa;">+{ww['last_day']:,} yesterday &bull; {ww['last_week']:,} last 7d &bull; <strong>{ww['with_mirrors_total']:,}</strong> w/ mirrors</div>
          </td>
        </tr>
        <tr>
          <td style="background: #27272a; border-radius: 8px; padding: 14px; text-align: center; width: 50%; border: 1px solid #3f3f46;">
            <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px;">Hugging Face OpenAudio-48k</div>
            <div style="font-size: 26px; font-weight: 800; color: #fbbf24; margin: 4px 0;">{hf_ds['downloads_all_time']:,} dl</div>
            <div style="font-size: 11px; color: #a1a1aa;">30d: {hf_ds['downloads_30d']:,} &bull; {hf_ds['likes']} &#10084;&#65039; &bull; <a href="https://huggingface.co/datasets/{html.escape(hf_ds['id'])}" style="color: #fbbf24; text-decoration: none;">Dataset Card</a></div>
          </td>
          <td style="background: #27272a; border-radius: 8px; padding: 14px; text-align: center; width: 50%; border: 1px solid #3f3f46;">
            <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px;">Zenodo 24050283 &amp; HF Spaces</div>
            <div style="font-size: 22px; font-weight: 800; color: #c084fc; margin: 4px 0;">{zen['downloads']:,} dl / {zen['views']:,} views</div>
            <div style="font-size: 11px; color: #a1a1aa;">Unique: {zen['unique_downloads']:,} dl &bull; {zen['unique_views']:,} views</div>
            {space_items}
            {model_items}
          </td>
        </tr>
      </table>

      <!-- All 10 PyPI Tools Ranked Table -->
      <h3 style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.6px; color: #cbd5e1; margin: 20px 0 8px 0; font-weight: 700;">All 10 PyPI Tools Leaderboard (Lifetime &amp; Recent)</h3>
      <table style="width: 100%; border-collapse: collapse; background: #18181b; border-radius: 8px; overflow: hidden; border: 1px solid #27272a; font-size: 12px;">
        <thead>
          <tr style="background: #27272a;">
            <th style="padding: 9px 8px; text-align: center; color: #94a3b8; font-size: 10px; text-transform: uppercase;">#</th>
            <th style="padding: 9px 8px; text-align: left; color: #94a3b8; font-size: 10px; text-transform: uppercase;">Package</th>
            <th style="padding: 9px 8px; text-align: right; color: #94a3b8; font-size: 10px; text-transform: uppercase;">Direct</th>
            <th style="padding: 9px 8px; text-align: right; color: #94a3b8; font-size: 10px; text-transform: uppercase;">w/ Mirrors</th>
            <th style="padding: 9px 8px; text-align: right; color: #94a3b8; font-size: 10px; text-transform: uppercase;">Yday</th>
            <th style="padding: 9px 8px; text-align: right; color: #94a3b8; font-size: 10px; text-transform: uppercase;">7d</th>
            <th style="padding: 9px 8px; text-align: right; color: #94a3b8; font-size: 10px; text-transform: uppercase;">Peak (Direct)</th>
          </tr>
        </thead>
        <tbody>
          {tool_rows}
          <tr style="background: #27272a; font-weight: 800;">
            <td style="padding: 10px 8px; text-align: center; color: #e4e4e7;">&#931;</td>
            <td style="padding: 10px 8px; color: #e4e4e7;">Suite Total (10 Tools) &bull; GH {suite['gh_stars']}&#9733;/{suite['gh_forks']}&#9913;</td>
            <td style="padding: 10px 8px; text-align: right; color: #38bdf8; font-family: monospace;">{suite['without_mirrors_total']:,}</td>
            <td style="padding: 10px 8px; text-align: right; color: #e4e4e7; font-family: monospace;">{suite['with_mirrors_total']:,}</td>
            <td style="padding: 10px 8px; text-align: right; color: #4ade80; font-family: monospace;">+{suite['last_day']:,}</td>
            <td style="padding: 10px 8px; text-align: right; color: #c084fc; font-family: monospace;">{suite['last_week']:,}</td>
            <td style="padding: 10px 8px; text-align: right; color: #94a3b8; font-family: monospace;">-</td>
          </tr>
        </tbody>
      </table>

      <!-- Last 7 Days Daily Breakdown -->
      <h3 style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.6px; color: #cbd5e1; margin: 22px 0 8px 0; font-weight: 700;">Last 7 Finalized Days Trend</h3>
      <table style="width: 100%; border-collapse: collapse; background: #202023; border-radius: 8px; overflow: hidden; border: 1px solid #27272a; font-size: 12px;">
        <thead>
          <tr style="background: #27272a;">
            <th style="padding: 9px 12px; text-align: left; color: #94a3b8; font-size: 10px; text-transform: uppercase;">Date (UTC)</th>
            <th style="padding: 9px 12px; text-align: right; color: #94a3b8; font-size: 10px; text-transform: uppercase;">WaveWash Direct</th>
            <th style="padding: 9px 12px; text-align: right; color: #94a3b8; font-size: 10px; text-transform: uppercase;">All 10 Direct</th>
            <th style="padding: 9px 12px; text-align: right; color: #94a3b8; font-size: 10px; text-transform: uppercase;">All 10 w/ Mirrors</th>
          </tr>
        </thead>
        <tbody>
          {trend_rows}
        </tbody>
      </table>

      <!-- Quick Links -->
      <div style="margin-top: 22px; padding-top: 16px; border-top: 1px solid #27272a; text-align: center;">
        <a href="https://pypi.org/project/wavewash/" style="display: inline-block; margin: 4px; padding: 8px 12px; background: #0284c7; color: #ffffff; text-decoration: none; border-radius: 6px; font-size: 12px; font-weight: 600;">PyPI WaveWash</a>
        <a href="https://pypi.org/project/splitsafe/" style="display: inline-block; margin: 4px; padding: 8px 12px; background: #27272a; color: #e4e4e7; text-decoration: none; border-radius: 6px; font-size: 12px; font-weight: 600; border: 1px solid #3f3f46;">PyPI SplitSafe</a>
        <a href="https://huggingface.co/datasets/boobalanit/ComposerMind-OpenAudio-48k" style="display: inline-block; margin: 4px; padding: 8px 12px; background: #27272a; color: #fbbf24; text-decoration: none; border-radius: 6px; font-size: 12px; font-weight: 600; border: 1px solid #3f3f46;">HF Dataset</a>
        <a href="https://zenodo.org/records/24050283" style="display: inline-block; margin: 4px; padding: 8px 12px; background: #27272a; color: #c084fc; text-decoration: none; border-radius: 6px; font-size: 12px; font-weight: 600; border: 1px solid #3f3f46;">Zenodo Corpus</a>
        <a href="https://github.com/BoobalanArjunan" style="display: inline-block; margin: 4px; padding: 8px 12px; background: #27272a; color: #e4e4e7; text-decoration: none; border-radius: 6px; font-size: 12px; font-weight: 600; border: 1px solid #3f3f46;">GitHub</a>
      </div>

    </div>

    <!-- Footer -->
    <div style="background: #141416; padding: 14px; text-align: center; border-top: 1px solid #27272a; font-size: 11px; color: #71717a;">
      Sent automatically via GitHub Actions Daily Analytics Cron &bull; ComposerMind Open Audio ML Suite
    </div>

  </div>
</body>
</html>
"""
    return html_doc


def build_subject(stats: dict[str, Any]) -> str:
    suite = stats["suite"]
    hf_dl = stats["hf"]["dataset"]["downloads_all_time"]
    zen_dl = stats["zenodo"]["downloads"]
    return (
        f"Audio ML Suite Daily Stats: {suite['without_mirrors_total']:,} Direct / "
        f"{suite['with_mirrors_total']:,} PyPI Total (+{suite['last_day']:,} yday) | "
        f"HF {hf_dl:,} | Zenodo {zen_dl:,}"
    )


def send_email_resend(html_content: str, stats: dict[str, Any], dry_run: bool = False) -> bool:
    suite = stats["suite"]
    hf_dl = stats["hf"]["dataset"]["downloads_all_time"]
    zen_dl = stats["zenodo"]["downloads"]

    if dry_run or not RESEND_API_KEY:
        print("[INFO] RESEND_API_KEY not configured (or --dry-run set). Running in preview mode.")
        print(f"[INFO] Would send to: {TARGET_EMAIL}")
        print(f"[INFO] Subject: {build_subject(stats)}")
        print(
            f"[INFO] Suite Summary: {suite['without_mirrors_total']} direct / "
            f"{suite['with_mirrors_total']} w/ mirrors (+{suite['last_day']} yesterday), "
            f"HF Dataset={hf_dl}, Zenodo={zen_dl}"
        )
        return True

    payload = {
        "from": FROM_EMAIL,
        "to": [TARGET_EMAIL],
        "subject": build_subject(stats),
        "html": html_content,
    }

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp_data = json.loads(resp.read().decode("utf-8", errors="replace"))
            print(f"[SUCCESS] Email dispatched successfully! Resend ID: {resp_data.get('id')}")
            return True
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"[ERROR] Resend API error ({e.code}): {err_body}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Send consolidated daily analytics email for all 10 PyPI tools + HF + Zenodo.")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without sending an email via Resend.")
    parser.add_argument("--preview-html", type=str, default="", help="Write rendered HTML digest to a local file.")
    args = parser.parse_args()

    print("=== ComposerMind 10-Tool Suite & WaveWash Daily Analytics Reporter ===")
    stats = fetch_stats()
    html_doc = build_html_email(stats)

    if args.preview_html:
        with open(args.preview_html, "w", encoding="utf-8") as f:
            f.write(html_doc)
        print(f"[INFO] Wrote HTML preview to {args.preview_html}")

    success = send_email_resend(html_doc, stats, dry_run=args.dry_run)
    if not success and RESEND_API_KEY and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
