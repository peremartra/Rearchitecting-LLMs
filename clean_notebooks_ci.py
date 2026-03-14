#!/usr/bin/env python3
"""
Removes metadata.widgets only from notebooks changed in the current push.
"""
import json, subprocess
from pathlib import Path

# Get only notebooks changed in this push (requires fetch-depth: 2)
result = subprocess.run(
    ["git", "diff", "HEAD~1", "HEAD", "--name-only", "--diff-filter=ACM"],
    capture_output=True, text=True
)
notebooks = [Path(p) for p in result.stdout.splitlines() if p.endswith(".ipynb")]

for path in notebooks:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"  ⚠️  Skipping (invalid JSON or missing file): {path}")
        continue

    if "widgets" in nb.get("metadata", {}):
        del nb["metadata"]["widgets"]
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        subprocess.run(["git", "add", str(path)])
        print(f"  ✔ metadata.widgets removed: {path}")
    else:
        print(f"  ✔ No changes needed: {path}")
