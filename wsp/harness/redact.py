"""Replace machine-specific paths in run logs with placeholders (<repo>, <home>) before they are shared.

Usage:  python -m wsp.harness.redact experiments     # rewrites text logs under every raw/ folder in place
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".log"}


def redact(text: str) -> str:
    return text.replace(str(REPO), "<repo>").replace(str(Path.home()), "<home>")


def redact_tree(root: Path) -> int:
    """Rewrite every text log under `root` (skipping work/ and jobs/ folders). Returns files changed."""
    changed = 0
    for path in root.rglob("*"):
        parts = path.relative_to(root).parts
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES or "work" in parts or "jobs" in parts:
            continue  # work/ is never shared; jobs/ holds files that running workers still read
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        cleaned = redact(text)
        if cleaned != text:
            path.write_text(cleaned)
            changed += 1
    return changed


if __name__ == "__main__":
    total = 0
    for raw in Path(sys.argv[1]).rglob("raw"):
        if raw.is_dir():
            total += redact_tree(raw)
    print(f"redacted {total} files")
