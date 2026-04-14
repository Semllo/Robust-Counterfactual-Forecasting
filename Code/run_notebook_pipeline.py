from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(r"d:\5_Article_Rapit_Manuel\Code\Datos")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Use glob patterns for notebooks whose filenames contain broken accents.
NOTEBOOK_PATTERNS = [
    "1_*.ipynb",
    "2_*.ipynb",
    "3_*.ipynb",
    "4_*.ipynb",
    "7_*Distritos.ipynb",
    "8_*.ipynb",
    "9_*.ipynb",
    "10_*.ipynb",
    "12_*.ipynb",
    "13_*.ipynb",
    "14_*.ipynb",
    "15_*.ipynb",
]


def resolve_notebooks() -> list[Path]:
    notebooks: list[Path] = []
    for pattern in NOTEBOOK_PATTERNS:
        matches = sorted(ROOT.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"Pattern {pattern!r} resolved to {len(matches)} notebooks: {matches}")
        notebooks.extend(matches)
    return notebooks


def execute_notebook(path: Path) -> None:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        data = json.load(fh)

    code_cells = [cell for cell in data.get("cells", []) if cell.get("cell_type") == "code"]
    env = {"__name__": "__main__", "__file__": str(path)}

    old_cwd = Path.cwd()
    os.chdir(path.parent)
    try:
        for idx, cell in enumerate(code_cells, start=1):
            source = "".join(cell.get("source", []))
            if not source.strip():
                continue
            if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
                raise RuntimeError(f"Unsupported Jupyter magic in {path.name} cell {idx}")
            code = compile(source, f"{path.name}::cell_{idx}", "exec")
            exec(code, env)
    finally:
        os.chdir(old_cwd)


def main() -> int:
    notebooks = resolve_notebooks()
    print("Notebook pipeline:")
    for nb in notebooks:
        print(f" - {nb.name}")

    for nb in notebooks:
        print(f"\n=== RUN {nb.name} ===")
        started = time.time()
        try:
            execute_notebook(nb)
        except Exception:
            print(traceback.format_exc())
            print(f"FAILED: {nb.name}")
            return 1
        elapsed = time.time() - started
        print(f"OK: {nb.name} ({elapsed:.1f}s)")

    print("\nPipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
