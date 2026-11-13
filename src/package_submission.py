"""
Creates and clean-room validates the final CyberCampus submission archive:
  기말-201935607-김대현.zip

Clean-room 4-step gate (--verify-cleanroom, enabled by default):
  Step 1: Static, <100MB budget, executed error-free notebook, zero b"file://" in PDF,
          and 100% file-by-file SHA-256 parity on a fresh temporary directory extraction.
  Step 2: As-shipped Web & Parity gate BEFORE any Python script overwrites files:
          `node web/scripts/parity_check.mjs` -> `npm ci` -> `npm run build` -> `node scripts/offline_smoke.mjs`.
  Step 3: Standalone Python reproducibility gate AFTER Step 2 inside the isolated temp dir
          (`cwd=extracted_dir`, `PYTHONPATH=extracted_dir/src`, `KEYPRINT_REUSE_WEIGHTS=1`):
          re-runs `visualize_and_run.py`, `eval_password_calibration.py`, `eval_typing_style.py`
          and asserts key test metrics match shipped JSON snapshots within 1e-4.
  Step 4: Atomic `os.replace(tmp_zip, final_zip)` ONLY when Steps 1..3 all succeed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

import nbformat

MAX_ARCHIVE_BYTES = 100 * 1024 * 1024  # Strict < 100 MB budget
METRIC_ATOL = 1e-4

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

MANDATORY_ARCHIVE_FILES = [
    "포트폴리오_201935607_김대현.pdf",
    "README.md",
    "submission.json",
    "requirements.txt",
    "notebooks/keyprint_training_pipeline.ipynb",
    "data/raw/DSL-StrongPasswordData.csv",
    "data/raw/KeyRecs-free-text.csv",
    "data/raw/KeyRecs-fixed-text.csv",
    "web/package-lock.json",
    "web/public/models/keyprint_net.onnx",
    "web/public/models/keyprint_net_password.onnx",
    "web/public/models/model_config.json",
]

NON_METRIC_METADATA_KEYS = {
    "elapsed_seconds",
    "elapsed_sec",
    "device",
    "run_id",
    "timestamp",
    "created_at",
    "git_commit",
    "command",
    "cwd",
    "python_executable",
    "python_version",
    "output_sha256",
    "artifact_hashes",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_excluded_rel(rel_posix: str) -> bool:
    parts = rel_posix.split("/")
    excluded_names = {"__pycache__", ".DS_Store", ".pytest_cache", ".ipynb_checkpoints"}
    if any(p in excluded_names for p in parts):
        return True
    if rel_posix.endswith((".pyc", ".pyo", ".npz")):
        return True
    if rel_posix.startswith(("web/node_modules/", "web/dist/", "web/public/ort/")):
        return True
    if Path(rel_posix).name.startswith(".tmp_"):
        return True
    return False


def collect_archive_sources(root: Path, pdf_override: Path | None = None) -> dict[str, Path]:
    """Return deterministic mapping `{posix_arcname: source_path}` for the submission archive."""
    sources: dict[str, Path] = {}

    def add_file(fpath: Path, arcname: str | None = None) -> None:
        if not fpath.exists() or not fpath.is_file():
            return
        rel = unicodedata.normalize("NFC", arcname or fpath.relative_to(root).as_posix())
        if _is_excluded_rel(rel):
            return
        sources[rel] = fpath

    def add_tree(dir_path: Path, suffixes: tuple[str, ...] | None = None) -> None:
        if not dir_path.exists() or not dir_path.is_dir():
            return
        for item in sorted(dir_path.rglob("*")):
            if not item.is_file():
                continue
            rel = unicodedata.normalize("NFC", item.relative_to(root).as_posix())
            if _is_excluded_rel(rel):
                continue
            if suffixes is not None and item.suffix.lower() not in suffixes:
                continue
            sources[rel] = item

    # 1. Root files (+ CI workflow if present)
    pdf_src = pdf_override.resolve() if pdf_override is not None else (root / "포트폴리오_201935607_김대현.pdf")
    add_file(pdf_src, arcname="포트폴리오_201935607_김대현.pdf")
    for root_name in ("README.md", "submission.json", "requirements.txt", ".gitignore"):
        add_file(root / root_name)
    add_file(root / ".github" / "workflows" / "deploy.yml")

    # 2. Raw & processed datasets
    add_tree(root / "data" / "raw", suffixes=(".csv",))
    add_tree(root / "data" / "processed")

    # 3. Python source & optional test suites
    add_tree(root / "src", suffixes=(".py",))
    if (root / "tests").exists():
        add_tree(root / "tests", suffixes=(".py", ".mjs", ".js", ".json"))

    # 4. Executed Jupyter notebook
    add_file(root / "notebooks" / "keyprint_training_pipeline.ipynb")

    # 5. Trained weights, ONNX models, legacy 9ch weights, and JSON evaluation reports
    add_tree(root / "models", suffixes=(".pt", ".onnx", ".json"))

    # 6. Figures & E2E UI screenshots
    add_tree(root / "assets" / "figures", suffixes=(".png",))
    add_tree(root / "assets" / "screenshots", suffixes=(".png",))

    # 7. Web application source, lockfile, public models/assets, and verification scripts
    for web_file in ("package.json", "package-lock.json", "vite.config.js", "index.html"):
        add_file(root / "web" / web_file)
    add_tree(root / "web" / "src")
    add_tree(root / "web" / "public")
    add_tree(root / "web" / "scripts")

    return dict(sorted(sources.items()))


def write_archive(tmp_zip: Path, sources: dict[str, Path]) -> None:
    tmp_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for arcname, src_path in sources.items():
            zf.write(src_path, arcname=arcname)


def _run_checked(cmd: list[str], cwd: Path, env: dict[str, str] | None = None, step_label: str = "") -> None:
    prefix = f"[{step_label}] " if step_label else ""
    print(f"{prefix}$ (cd {cwd} && {' '.join(cmd)})", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{prefix}Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _assert_numeric_tree_close(
    expected: Any,
    actual: Any,
    path_prefix: str,
    atol: float = METRIC_ATOL,
) -> int:
    """Recursively assert numeric leaves in `actual` match `expected` within `atol`. Returns count checked."""
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return 0
    if isinstance(expected, (int, float)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise AssertionError(f"Metric type mismatch at {path_prefix}: expected numeric {expected!r}, got {actual!r}")
        exp_f = float(expected)
        act_f = float(actual)
        if math.isnan(exp_f) or math.isnan(act_f):
            raise AssertionError(f"NaN metric at {path_prefix}: expected={exp_f}, actual={act_f}")
        diff = abs(exp_f - act_f)
        if diff > atol:
            raise AssertionError(
                f"Metric drift > {atol} at {path_prefix}: shipped={exp_f:.6f}, cleanroom={act_f:.6f}, diff={diff:.6f}"
            )
        return 1
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"Expected dict at {path_prefix}, got {type(actual).__name__}")
        checked = 0
        for k, v in expected.items():
            if k in NON_METRIC_METADATA_KEYS:
                continue
            if k not in actual:
                raise AssertionError(f"Missing key {path_prefix}.{k} in cleanroom output")
            checked += _assert_numeric_tree_close(v, actual[k], f"{path_prefix}.{k}", atol=atol)
        return checked
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(
                f"List length mismatch at {path_prefix}: shipped={len(expected)}, cleanroom={len(actual) if isinstance(actual, list) else 'non-list'}"
            )
        checked = 0
        for idx, (ev, av) in enumerate(zip(expected, actual)):
            checked += _assert_numeric_tree_close(ev, av, f"{path_prefix}[{idx}]", atol=atol)
        return checked
    return 0


def step1_static_and_hash_gate(tmp_zip: Path, extracted_dir: Path, sources: dict[str, Path]) -> None:
    print("[Step 1/4] Running Static, Budget, Notebook, PDF & SHA-256 Parity Gate...", flush=True)
    archive_bytes = tmp_zip.stat().st_size
    if archive_bytes >= MAX_ARCHIVE_BYTES:
        raise AssertionError(
            f"Archive size {archive_bytes / (1024 * 1024):.2f} MB exceeds 100 MB limit ({MAX_ARCHIVE_BYTES} bytes)"
        )

    with zipfile.ZipFile(tmp_zip, "r") as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise AssertionError(f"Corrupted ZIP member detected: {bad_member}")
        zf.extractall(extracted_dir)

    # 1. Mandatory files check
    for rel_req in MANDATORY_ARCHIVE_FILES:
        p = extracted_dir / rel_req
        if not p.exists() or not p.is_file() or p.stat().st_size == 0:
            raise AssertionError(f"Missing or empty mandatory file in archive: {rel_req}")

    # 2. submission.json schema check
    sub_data = json.loads((extracted_dir / "submission.json").read_text(encoding="utf-8"))
    for key in REQUIRED_SUBMISSION_KEYS:
        if key not in sub_data or sub_data[key] in (None, ""):
            raise AssertionError(f"Missing or empty required key in submission.json: {key}")

    # 3. Executed notebook & zero error outputs check
    nb = nbformat.read(extracted_dir / "notebooks" / "keyprint_training_pipeline.ipynb", as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    executed_cells = [c for c in code_cells if len(c.get("outputs", [])) > 0]
    if len(executed_cells) < 1:
        raise AssertionError("Notebook must contain >= 1 executed code cell with non-empty outputs")
    error_cells = [
        idx
        for idx, c in enumerate(code_cells)
        if any(out.get("output_type") == "error" for out in c.get("outputs", []))
    ]
    if error_cells:
        raise AssertionError(f"Notebook contains cell(s) with output_type=='error' at code cell indices: {error_cells}")

    # 4. Zero local file:// URIs in rendered portfolio PDF
    pdf_bytes = (extracted_dir / "포트폴리오_201935607_김대현.pdf").read_bytes()
    file_uri_count = pdf_bytes.count(b"file://")
    if file_uri_count != 0:
        raise AssertionError(
            f"포트폴리오_201935607_김대현.pdf contains {file_uri_count} forbidden b'file://' occurrence(s)"
        )

    # 5. Exact 1-to-1 SHA-256 verification of EVERY extracted file vs workspace source
    extracted_files = {
        unicodedata.normalize("NFC", p.relative_to(extracted_dir).as_posix()): p
        for p in sorted(extracted_dir.rglob("*"))
        if p.is_file()
    }
    if set(extracted_files.keys()) != set(sources.keys()):
        missing = sorted(set(sources.keys()) - set(extracted_files.keys()))
        extra = sorted(set(extracted_files.keys()) - set(sources.keys()))
        raise AssertionError(f"Extracted file set mismatch: missing={missing}, extra={extra}")

    mismatches: list[str] = []
    for rel, ext_path in extracted_files.items():
        src_hash = sha256_file(sources[rel])
        ext_hash = sha256_file(ext_path)
        if src_hash != ext_hash:
            mismatches.append(f"{rel} (src={src_hash[:12]} != zip={ext_hash[:12]})")
    if mismatches:
        raise AssertionError(f"SHA-256 mismatch on {len(mismatches)} file(s): {mismatches[:5]}")

    print(
        f"[Step 1/4] PASSED ({archive_bytes / (1024 * 1024):.2f} MB, {len(extracted_files)} files SHA-256 verified, "
        f"{len(executed_cells)}/{len(code_cells)} executed notebook cells, 0 errors, 0 file:// URIs).",
        flush=True,
    )


def step2_as_shipped_web_and_parity_gate(extracted_dir: Path) -> None:
    print("[Step 2/4] Running As-Shipped Web & Parity Gate (before any Python regeneration)...", flush=True)
    web_dir = extracted_dir / "web"
    _run_checked(["npm", "ci"], cwd=web_dir, step_label="Step 2.1 npm-ci")
    _run_checked(["node", "web/scripts/parity_check.mjs"], cwd=extracted_dir, step_label="Step 2.2 parity")
    _run_checked(["npm", "run", "build"], cwd=web_dir, step_label="Step 2.3 build")
    _run_checked(["node", "scripts/offline_smoke.mjs"], cwd=web_dir, step_label="Step 2.4 smoke")
    _run_checked(["node", "scripts/adversarial_regression.mjs"], cwd=web_dir, step_label="Step 2.5 adversarial")
    print(
        "[Step 2/4] PASSED (npm ci + as-shipped parity_check + vite build + offline_smoke + adversarial_regression).",
        flush=True,
    )


def step3_python_reproducibility_gate(extracted_dir: Path, python_bin: Path) -> None:
    print("[Step 3/4] Running Standalone Clean-Room Python Reproducibility Gate...", flush=True)
    eval_path = extracted_dir / "models" / "evaluation_report.json"
    pw_path = extracted_dir / "models" / "password_calibration.json"
    style_path = extracted_dir / "models" / "typing_style_analysis.json"

    for report_p in (eval_path, pw_path, style_path):
        if not report_p.exists():
            raise AssertionError(f"Missing shipped JSON report prior to Step 3: {report_p.name}")

    # Snapshot shipped JSON reports BEFORE running Python scripts in the clean-room directory
    shipped_eval = json.loads(eval_path.read_text(encoding="utf-8"))
    shipped_pw = json.loads(pw_path.read_text(encoding="utf-8"))
    shipped_style = json.loads(style_path.read_text(encoding="utf-8"))

    clean_env = os.environ.copy()
    clean_env["PYTHONPATH"] = str(extracted_dir / "src")
    clean_env["PYTHONNOUSERSITE"] = "1"
    clean_env["MPLBACKEND"] = "Agg"

    reuse_env = {**clean_env, "KEYPRINT_REUSE_WEIGHTS": "1"}
    _run_checked(
        [str(python_bin), "src/visualize_and_run.py"],
        cwd=extracted_dir,
        env=reuse_env,
        step_label="Step 3.1 visualize_and_run",
    )
    _run_checked(
        [str(python_bin), "src/eval_password_calibration.py"],
        cwd=extracted_dir,
        env=clean_env,
        step_label="Step 3.2 eval_password_calibration",
    )
    _run_checked(
        [str(python_bin), "src/eval_typing_style.py"],
        cwd=extracted_dir,
        env=clean_env,
        step_label="Step 3.3 eval_typing_style",
    )

    repro_eval = json.loads(eval_path.read_text(encoding="utf-8"))
    repro_pw = json.loads(pw_path.read_text(encoding="utf-8"))
    repro_style = json.loads(style_path.read_text(encoding="utf-8"))

    # Explicit key presence assertions + 1e-4 numeric parity checks
    kp_shipped = shipped_eval["models"]["KeyPrintNet"]
    kp_repro = repro_eval["models"]["KeyPrintNet"]
    for req_metric in ("eer", "roc_auc"):
        if req_metric not in kp_shipped or req_metric not in kp_repro:
            raise AssertionError(f"Missing models.KeyPrintNet.{req_metric}")

    checked_leaves = 0
    checked_leaves += _assert_numeric_tree_close(kp_shipped, kp_repro, "evaluation_report.models.KeyPrintNet")
    checked_leaves += _assert_numeric_tree_close(
        shipped_eval["feature_upgrade"]["web_faithful_identification"],
        repro_eval["feature_upgrade"]["web_faithful_identification"],
        "evaluation_report.feature_upgrade.web_faithful_identification",
    )
    checked_leaves += _assert_numeric_tree_close(
        shipped_pw,
        repro_pw,
        "password_calibration",
    )
    checked_leaves += _assert_numeric_tree_close(
        shipped_style,
        repro_style,
        "typing_style_analysis",
    )

    print(
        f"[Step 3/4] PASSED ({checked_leaves} clean-room numeric metrics matched shipped JSON snapshots within {METRIC_ATOL}).",
        flush=True,
    )


def build_zip(
    out_path: Path | None = None,
    verify_cleanroom: bool = True,
    static_only: bool = False,
    pdf_override: Path | None = None,
) -> Path:
    root = Path(__file__).resolve().parent.parent
    final_zip = out_path.resolve() if out_path is not None else (root / "기말-201935607-김대현.zip")
    tmp_zip = final_zip.parent / f".{final_zip.stem}.tmp.zip"

    venv_python = root / ".venv" / "bin" / "python"
    python_bin = venv_python if venv_python.exists() else Path(sys.executable)

    sources = collect_archive_sources(root, pdf_override=pdf_override)
    if tmp_zip.exists():
        tmp_zip.unlink()

    try:
        write_archive(tmp_zip, sources)
        size_mb = tmp_zip.stat().st_size / (1024 * 1024)
        print(f"[Package] Built candidate {tmp_zip.name} ({size_mb:.2f} MB, {len(sources)} files)", flush=True)

        if verify_cleanroom or static_only:
            with tempfile.TemporaryDirectory(prefix="keyprint_cleanroom_") as tmp_dir_str:
                extracted_dir = Path(tmp_dir_str)
                step1_static_and_hash_gate(tmp_zip, extracted_dir, sources)
                if verify_cleanroom and not static_only:
                    step2_as_shipped_web_and_parity_gate(extracted_dir)
                    step3_python_reproducibility_gate(extracted_dir, python_bin)

        print(f"[Step 4/4] Atomically promoting {tmp_zip.name} -> {final_zip}", flush=True)
        os.replace(tmp_zip, final_zip)
        print(f"[Package] SUCCESS: {final_zip} ({size_mb:.2f} MB, {len(sources)} files)", flush=True)
        return final_zip
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package and clean-room verify the KeyPrint Guard submission ZIP archive."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output ZIP path (defaults to <repo_root>/기말-201935607-김대현.zip).",
    )
    parser.add_argument(
        "--verify-cleanroom",
        dest="verify_cleanroom",
        action="store_true",
        default=True,
        help="Run the full 4-step clean-room verification gate before atomic replace (enabled by default).",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        default=False,
        help="Run only Step 1 (Static/Hash/PDF/Notebook gate) + Step 4 atomic replace for fast dry-runs.",
    )
    parser.add_argument(
        "--pdf-override",
        type=Path,
        default=None,
        help="Optional path to a freshly rendered PDF to package instead of root 포트폴리오_201935607_김대현.pdf.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    build_zip(
        out_path=args.out,
        verify_cleanroom=args.verify_cleanroom,
        static_only=args.static_only,
        pdf_override=args.pdf_override,
    )
