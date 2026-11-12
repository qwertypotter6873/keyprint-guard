"""
Creates and validates the final CyberCampus submission archive:
  기말-201935607-김대현.zip
containing:
  - 포트폴리오_201935607_김대현.pdf (identical to README.md)
  - README.md
  - submission.json
  - requirements.txt
  - notebooks/keyprint_training_pipeline.ipynb (with all executed cell outputs preserved)
  - src/*.py
  - models/keyprint_net.onnx, keyprint_net.pt, vanilla_1dcnn.pt, evaluation_report.json
  - web/ source files (excluding node_modules / heavy build caches)
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
import nbformat


REQUIRED_SUBMISSION_KEYS = [
    "student_id",
    "name",
    "title",
    "github_url",
    "demo_url",
    "data_type",
    "training",
    "models",
    "primary_metric",
]


def build_zip() -> None:
    root = Path(__file__).resolve().parent.parent
    sub_json_path = root / "submission.json"
    pdf_path = root / "포트폴리오_201935607_김대현.pdf"
    nb_path = root / "notebooks" / "keyprint_training_pipeline.ipynb"
    readme_path = root / "README.md"

    assert sub_json_path.exists(), "Missing submission.json"
    assert pdf_path.exists(), "Missing 포트폴리오_201935607_김대현.pdf"
    assert nb_path.exists(), "Missing executed notebook"
    assert readme_path.exists(), "Missing README.md"

    sub_data = json.loads(sub_json_path.read_text(encoding="utf-8"))
    for key in REQUIRED_SUBMISSION_KEYS:
        assert key in sub_data and sub_data[key] not in (None, ""), f"Missing key in submission.json: {key}"

    nb = nbformat.read(nb_path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    executed_cells = [c for c in code_cells if len(c.get("outputs", [])) > 0]
    assert len(executed_cells) == len(code_cells), (
        f"Notebook has unexecuted code cells: {len(executed_cells)}/{len(code_cells)}"
    )

    zip_path = root / "기말-201935607-김대현.zip"
    include_files = [
        pdf_path,
        readme_path,
        sub_json_path,
        root / "requirements.txt",
        nb_path,
        *sorted((root / "src").glob("*.py")),
        root / "models" / "keyprint_net.onnx",
        root / "models" / "keyprint_net.pt",
        root / "models" / "vanilla_1dcnn.pt",
        *sorted((root / "models" / "legacy").glob("*")),  # 9ch password encoder (.pt + .onnx)
        root / "models" / "evaluation_report.json",
        *sorted((root / "assets" / "figures").glob("*.png")),
        *sorted((root / "assets" / "screenshots").glob("*.png")),
        root / "web" / "package.json",
        root / "web" / "vite.config.js",
        root / "web" / "index.html",
        *sorted((root / "web" / "src").glob("*")),
        *sorted((root / "web" / "scripts").glob("*.mjs")),
        *sorted((root / "web" / "scripts" / "fixtures").glob("*.json")),  # parity_check.mjs input
        *sorted((root / "web" / "public" / "models").glob("*")),
        root / "data" / "processed" / "scaler_and_metadata.json",
        root / ".github" / "workflows" / "deploy.yml",
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in include_files:
            if fpath.exists() and fpath.is_file():
                arcname = fpath.relative_to(root).as_posix()
                zf.write(fpath, arcname=arcname)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    assert size_mb < 100.0, f"Archive exceeds 100MB limit: {size_mb:.2f} MB"
    print(f"[Package] Created {zip_path.name} ({size_mb:.2f} MB, {len(include_files)} files)")


if __name__ == "__main__":
    build_zip()
