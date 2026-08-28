"""Execute and validate the repository's research notebooks.

The script intentionally fails on the first cell error.  It can execute every
notebook in ``notebooks/`` or an explicit subset supplied on the command line.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def notebook_paths(arguments: list[str]) -> list[Path]:
    """Resolve explicit notebook arguments or the complete ordered curriculum."""
    if arguments:
        paths = [Path(value).resolve() for value in arguments]
    else:
        paths = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"notebook files not found: {missing}")
    return paths


def execute_notebook(path: Path, timeout: int) -> float:
    """Run one notebook from a fresh kernel and persist verified outputs."""
    notebook = nbformat.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        cell.setdefault("id", f"cell-{index:03d}")
    nbformat.validate(notebook)

    started = time.perf_counter()
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Optional .ipynb paths")
    parser.add_argument("--timeout", type=int, default=240, help="Per-cell timeout in seconds")
    args = parser.parse_args()

    paths = notebook_paths(args.notebooks)
    if not paths:
        raise RuntimeError(f"no notebooks found in {NOTEBOOK_DIR}")
    for path in paths:
        elapsed = execute_notebook(path, args.timeout)
        relative = path.relative_to(ROOT)
        print(f"PASS {relative} ({elapsed:.2f}s)", flush=True)
    print(f"Executed {len(paths)} notebooks successfully.")


if __name__ == "__main__":
    main()
