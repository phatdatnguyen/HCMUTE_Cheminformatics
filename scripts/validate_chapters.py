"""Execute Chapters 1-12 in fresh kernels using this Python environment.

Run from any directory: python scripts/validate_chapters.py
By default executed copies and a JSON report go to outputs/validation/.
Use --inplace to update the source notebooks with verified outputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = (
    "Chapter01_Part1.ipynb",
    "Chapter01_Part2.ipynb",
    "Chapter01_Part3.ipynb",
    "Chapter02_Part1.ipynb",
    "Chapter02_Part2.ipynb",
    "Chapter03.ipynb",
    "Chapter04.ipynb",
    "Chapter05.ipynb",
    "Chapter06.ipynb",
    "Chapter07.ipynb",
    "Chapter08_Part1.ipynb",
    "Chapter08_Part2.ipynb",
    "Chapter08_Part3.ipynb",
    "Chapter08_Part4.ipynb",
    "Chapter08_Part5.ipynb",
    "Chapter08_Part6.ipynb",
    "Chapter08_Part7.ipynb",
    "Chapter09_Part1.ipynb",
    "Chapter09_Part2.ipynb",
    "Chapter09_Part3.ipynb",
    "Chapter09_Part4.ipynb",
    "Chapter10_Part1.ipynb",
    "Chapter10_Part2.ipynb",
    "Chapter11_Part1.ipynb",
    "Chapter11_Part2.ipynb",
    "Chapter11_Part3.ipynb",
    "Chapter11_Part4.ipynb",
    "Chapter12_Part1.ipynb",
    "Chapter12_Part2.ipynb",
    "Chapter12_Part3.ipynb",
    "Chapter12_Part4.ipynb",
    "Chapter12_Part5.ipynb",
    "Chapter12_Part6.ipynb",
    "Chapter12_Part7.ipynb",
    "Chapter12_Part8.ipynb",
)
PACKAGES = (
    "numpy", "pandas", "matplotlib", "seaborn", "openpyxl", "rdkit",
    "py3Dmol", "openmm", "jupyterlab", "ipykernel", "nbformat", "nbclient",
    "scipy", "psi4", "mopac", "optking", "qcelemental", "scikit-learn", "torch", "torch-geometric", "joblib",
)


def installed_version(package: str) -> str:
    """Include Conda-only packages, which may lack Python distribution metadata."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        for record_path in (Path(sys.prefix) / "conda-meta").glob(f"{package}-*.json"):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("name") == package:
                return record["version"]
        return "not installed"


def cell_timings(notebook) -> list[dict]:
    """Kernel busy-to-idle times include calculations and rendering outputs."""
    timings = []
    for index, cell in enumerate(notebook.cells):
        execution = cell.metadata.get("execution", {})
        start = execution.get("iopub.status.busy")
        end = execution.get("iopub.status.idle")
        if cell.cell_type == "code" and start and end:
            seconds = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
            timings.append({"cell_index": index, "seconds": round(seconds, 4)})
    return timings


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", metavar="NOTEBOOK",
                        help="Optional notebook filenames; default: all thirty-five.")
    parser.add_argument("--inplace", action="store_true")
    chapters = parser.add_mutually_exclusive_group()
    chapters.add_argument("--chapter8", action="store_true",
                          help="Select the seven Chapter 8 notebooks.")
    chapters.add_argument("--chapter9", action="store_true",
                          help="Select the four Chapter 9 notebooks.")
    chapters.add_argument("--chapter10", action="store_true",
                          help="Select the two Chapter 10 notebooks.")
    chapters.add_argument("--chapter11", action="store_true",
                          help="Select the four Chapter 11 notebooks.")
    chapters.add_argument("--chapter12", action="store_true",
                          help="Select the eight Chapter 12 notebooks.")
    chapters.add_argument("--chapters10-11", action="store_true",
                          help="Select all six notebooks in Chapters 10-11.")
    parser.add_argument("--cell-timeout", type=int, default=300, metavar="SECONDS",
                        help="Maximum execution time per cell (default: 300); use 10 for the Chapters 8-12 runtime checks.")
    args = parser.parse_args()
    if args.cell_timeout <= 0:
        parser.error("--cell-timeout must be positive.")
    selectors = {
        "chapter8": ("Chapter08_",), "chapter9": ("Chapter09_",),
        "chapter10": ("Chapter10_",), "chapter11": ("Chapter11_",),
        "chapter12": ("Chapter12_",),
        "chapters10_11": ("Chapter10_", "Chapter11_"),
    }
    prefixes = next((value for flag, value in selectors.items() if getattr(args, flag)), None)
    if prefixes and args.notebooks:
        parser.error("Use a chapter selector or explicit filenames, not both.")
    unknown = set(args.notebooks) - set(CHAPTERS)
    if unknown:
        parser.error(f"Choose filenames from Chapters 1-12: {', '.join(sorted(unknown))}")
    if prefixes:
        selected = [name for name in CHAPTERS if name.startswith(prefixes)]
    else:
        selected = args.notebooks or CHAPTERS
    output_dir = ROOT / "outputs" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"python": sys.version, "packages": {}, "cell_timeout_seconds": args.cell_timeout, "notebooks": []}
    for package in PACKAGES:
        report["packages"][package] = installed_version(package)

    # A private kernelspec selects this interpreter without changing user kernels.
    with tempfile.TemporaryDirectory(prefix="cheminformatics-kernel-") as temporary:
        kernels = Path(temporary) / "kernels"
        spec_dir = kernels / "course-validation"
        spec_dir.mkdir(parents=True)
        (spec_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Course validation", "language": "python",
            "env": {
                "MPLBACKEND": "module://matplotlib_inline.backend_inline",
                "MPLCONFIGDIR": str(Path(temporary) / "matplotlib"),
                "IPYTHONDIR": str(Path(temporary) / "ipython"),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "MKL_THREADING_LAYER": "SEQUENTIAL",
                "OPENBLAS_NUM_THREADS": "1",
            },
        }), encoding="utf-8")
        for filename in selected:
            start = time.monotonic()
            result = {"notebook": filename}
            print(f"Executing {filename} ...", flush=True)
            notebook = nbformat.read(ROOT / filename, as_version=4)
            try:
                nbformat.validate(notebook)
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        cell.outputs = []
                        cell.execution_count = None
                        cell.metadata.pop("execution", None)
                manager = KernelManager(
                    kernel_name="course-validation",
                    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
                )
                client = NotebookClient(
                    notebook, km=manager, timeout=args.cell_timeout, allow_errors=False,
                    force_raise_errors=True, record_timing=True,
                    resources={"metadata": {"path": str(ROOT)}},
                )
                client.execute(cleanup_kc=True)
                nbformat.validate(notebook)
                result["status"] = "passed"
                result["executed_cells"] = sum(
                    cell.cell_type == "code" and cell.execution_count is not None
                    for cell in notebook.cells
                )
                nbformat.write(notebook, ROOT / filename if args.inplace else output_dir / filename)
            except Exception as exc:
                result["status"] = "failed"
                result["error"] = str(exc)
                nbformat.write(notebook, output_dir / filename)
                print(f"  {exc}", file=sys.stderr, flush=True)
            result["seconds"] = round(time.monotonic() - start, 2)
            result["code_cell_seconds"] = cell_timings(notebook)
            result["slowest_cell_seconds"] = max(
                (cell["seconds"] for cell in result["code_cell_seconds"]), default=0.0,
            )
            report["notebooks"].append(result)
            print(f"  {result['status']} ({result['seconds']} s; "
                  f"slowest cell {result['slowest_cell_seconds']:.3f} s)", flush=True)

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return int(any(result["status"] != "passed" for result in report["notebooks"]))


if __name__ == "__main__":
    raise SystemExit(main())
