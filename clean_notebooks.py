python3 << 'PYEOF'
script = """#!/usr/bin/env python3
import json, subprocess, sys

def get_staged_notebooks():
    result = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"], capture_output=True, text=True)
    return [f for f in result.stdout.splitlines() if f.endswith(".ipynb")]

def clean_notebook(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  WARNING JSON invalido, se omite: {path} ({e})")
        return
    if "widgets" in nb.get("metadata", {}):
        del nb["metadata"]["widgets"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\\n")
        subprocess.run(["git", "add", path])
        print(f"  OK metadata.widgets eliminado: {path}")
    else:
        print(f"  OK Sin cambios necesarios: {path}")

def main():
    notebooks = get_staged_notebooks()
    if not notebooks:
        sys.exit(0)
    print("Limpiando notebooks staged...")
    for nb_path in notebooks:
        clean_notebook(nb_path)
    print("Listo.")

if __name__ == "__main__":
    main()
"""
with open("clean_notebooks.py", "w") as f:
    f.write(script)
print("Archivo actualizado.")
PYEOF
