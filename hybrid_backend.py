import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
R_SCRIPT = BASE_DIR / "rna_seq_backend.R"
DEFAULT_RSCRIPT_CANDIDATES = (
    Path("/home/ananya/miniconda3/envs/rnaseq/bin/Rscript"),
    Path("/home/ananya/miniconda3/pkgs/r-base-4.5.3-h1a7235e_0/bin/Rscript"),
)


def find_rscript() -> str | None:
    env_rscript = os.environ.get("TRANSCRIPTOMIC_RSCRIPT")
    candidates = [Path(env_rscript).expanduser()] if env_rscript else []
    path_rscript = shutil.which("Rscript")
    if path_rscript:
        candidates.append(Path(path_rscript))
    candidates.extend(DEFAULT_RSCRIPT_CANDIDATES)

    for candidate in candidates:
        if candidate and candidate.exists() and candidate.is_file():
            return str(candidate)
    return None


def r_backend_status() -> dict:
    rscript = find_rscript()
    return {
        "available": bool(rscript and R_SCRIPT.exists()),
        "rscript": rscript,
        "script": str(R_SCRIPT),
    }


def run_r_backend(count_df: pd.DataFrame, group_map: dict, label_a: str, label_b: str,
                  min_cpm: float, min_samples: int, padj_thresh: float,
                  fc_thresh: float, top_n: int) -> dict:
    status = r_backend_status()
    if not status["available"]:
        raise RuntimeError("R backend unavailable: install R/Rscript and keep rna_seq_backend.R in the app directory.")

    tmpdir = Path(tempfile.mkdtemp(prefix="transcriptomicviz_r_", dir="/tmp"))
    counts_path = tmpdir / "counts.csv"
    metadata_path = tmpdir / "metadata.csv"
    out_dir = tmpdir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_df = count_df.copy()
    export_df.index.name = "gene"
    export_df.to_csv(counts_path)

    metadata = pd.DataFrame({
        "sample": list(count_df.columns),
        "group": [group_map.get(sample, "Unassigned") for sample in count_df.columns],
    })
    metadata.to_csv(metadata_path, index=False)

    cmd = [
        status["rscript"],
        status["script"],
        str(counts_path),
        str(metadata_path),
        str(out_dir),
        label_a,
        label_b,
        str(min_cpm),
        str(min_samples),
        str(padj_thresh),
        str(fc_thresh),
        str(top_n),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "R backend failed.")

    manifest_path = out_dir / "manifest.json"
    results_path = out_dir / "de_results.csv"
    filtered_counts_path = out_dir / "filtered_counts.csv"
    if not manifest_path.exists() or not results_path.exists():
        raise RuntimeError("R backend did not produce the expected output files.")

    manifest = json.loads(manifest_path.read_text())
    de_results = pd.read_csv(results_path)
    filt_counts = pd.read_csv(filtered_counts_path, index_col=0) if filtered_counts_path.exists() else None
    return {
        "tmpdir": str(tmpdir),
        "manifest": manifest,
        "de_results": de_results,
        "filtered_counts": filt_counts,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
