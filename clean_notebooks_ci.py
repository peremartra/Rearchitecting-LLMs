#!/usr/bin/env python3
"""
Removes metadata.widgets from all notebooks in the repo.
Only files that are actually modified get staged, so the amend
only touches notebooks that had the widget metadata issue.
"""
import json, subprocess
from pathlib import Path

modified = []

for path in Path(".").rglob("*.ipynb"):
    # Skip hidden directories like .ipynb_checkpoints
    if any(part.startswith(".") for part in path.parts):
        continue

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"  ⚠️  Skipping (invalid JSON or missing file): {path}")
        continue

    if "widgets" in nb.get("metadata", {}):
        del nb["metadata"]["widgets"]
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        subprocess.run(["git", "add", str(path)])
        modified.append(path)
        print(f"  ✔ metadata.widgets removed: {path}")

if not modified:
    print("  ✔ No changes needed.")
