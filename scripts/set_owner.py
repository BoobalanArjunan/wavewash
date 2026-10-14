#!/usr/bin/env python3
"""Put YOUR name and GitHub username into WaveWash (run once, before building for PyPI).

    python scripts/set_owner.py --name "Your Full Name" --github your-github-name [--email you@example.com] [--hf-dir ../huggingface_space]

What it changes:
  pyproject.toml : authors + maintainers (shown on the PyPI page), project links
  LICENSE        : the copyright holder
  README.md      : GitHub username in links/badge, screenshot link made absolute (PyPI needs that)
  --hf-dir       : the same username in the Hugging Face Space files

The email is optional. Anything you put in pyproject.toml is PUBLIC on PyPI. If you don't want your
real email shown, leave --email out, or use the private address GitHub gives you
(Settings > Emails > "Keep my email addresses private").
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD_NAME = "WaveWash contributors"
PLACEHOLDER = "YOUR-GITHUB-NAME"


def toml_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="your full name, as it should appear")
    ap.add_argument("--github", required=True, help="your GitHub username")
    ap.add_argument("--email", default=None, help="optional public contact email")
    ap.add_argument("--hf-dir", default=None, help="path to the huggingface_space folder (optional)")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", a.github):
        sys.exit("That doesn't look like a GitHub username (letters, numbers and hyphens only).")
    if a.email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", a.email):
        sys.exit("That email address doesn't look valid.")
    repo = f"https://github.com/{a.github}/wavewash"

    # --- pyproject.toml
    p = ROOT / "pyproject.toml"
    s = p.read_text(encoding="utf-8")
    person = f'name = "{toml_str(a.name)}"' + (f', email = "{a.email}"' if a.email else "")
    s = re.sub(r"^maintainers = .*\n", "", s, flags=re.M)             # safe to run more than once
    s, n = re.subn(r"^authors = .*$", f"authors = [{{ {person} }}]\nmaintainers = [{{ {person} }}]", s, count=1, flags=re.M)
    if n != 1:
        sys.exit("Could not find the 'authors' line in pyproject.toml.")
    s = re.sub(r"\n\[project\.urls\]\n(?:.*\n)*?(?=\n\[|\Z)", "\n", s)       # drop old links block, re-add below
    urls = f'\n[project.urls]\nHomepage = "{repo}"\nSource = "{repo}"\nIssues = "{repo}/issues"\n'
    s = s.replace("\n[project.scripts]", urls + "\n[project.scripts]", 1)
    p.write_text(s, encoding="utf-8")

    # --- LICENSE
    lic = ROOT / "LICENSE"
    t = lic.read_text(encoding="utf-8")
    t = re.sub(r"(Copyright \(c\) \d{4}) .*", rf"\1 {a.name}", t, count=1)
    lic.write_text(t, encoding="utf-8")

    # --- README and docs (every .md file in the project)
    raw = f"https://raw.githubusercontent.com/{a.github}/wavewash/main/docs/screenshot-setup.png"
    for f in ROOT.rglob("*.md"):
        if "node_modules" in f.parts:
            continue
        t = f.read_text(encoding="utf-8")
        t2 = t.replace(PLACEHOLDER, a.github).replace("](docs/screenshot-setup.png)", f"]({raw})")
        if t2 != t:
            f.write_text(t2, encoding="utf-8")

    # --- Hugging Face Space files
    if a.hf_dir:
        for f in ("README.md", "requirements.txt", "app.py"):
            fp = Path(a.hf_dir) / f
            if fp.exists():
                fp.write_text(fp.read_text(encoding="utf-8").replace(PLACEHOLDER, a.github), encoding="utf-8")

    print(f"Done. Author/maintainer: {a.name}" + (f" <{a.email}>" if a.email else " (no email)"))
    print(f"Project links point to: {repo}")
    left = [str(f.relative_to(ROOT)) for f in ROOT.rglob("*")
            if f.is_file() and f.suffix in {".md", ".toml", ".py", ".txt", ".yml"} and "node_modules" not in f.parts
            and PLACEHOLDER in f.read_text(encoding="utf-8", errors="ignore") and f.name != "set_owner.py"]
    if left:
        print("Still contains the placeholder (check by hand):", ", ".join(left))


if __name__ == "__main__":
    main()
