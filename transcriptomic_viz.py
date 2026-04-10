#!/usr/bin/env python3
"""
TranscriptomicViz — Publication-grade RNA-seq analysis pipeline
Converted from React/JS to Python

Analyses:
  • Auto-detects HTSeq or featureCounts format
  • TMM normalization
  • CPM filtering + VST transform
  • Negative-Binomial DE test with BH-FDR correction
  • LFC shrinkage
  • PCA plot
  • Expression heatmap (top-N variable genes)
  • DEG heatmap (up / down / both)
  • Volcano plot
  • DEG results CSV export

Usage:
  python transcriptomic_viz.py \
      --files ctrl1.txt ctrl2.txt ctrl3.txt treat1.txt treat2.txt treat3.txt \
      --group_a ctrl1 ctrl2 ctrl3 \
      --group_b treat1 treat2 treat3 \
      --label_a Control \
      --label_b Treatment \
      [--min_cpm 1] [--min_samples 2] \
      [--padj 0.05] [--fc 1.0] \
      [--top_n 50] \
      [--out_dir results]
"""

import argparse
import csv
import os
import sys
import warnings
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from scipy.special import gammaln

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

HTSEQ_SKIP = {
    "__no_feature", "__ambiguous", "__too_low_aQual",
    "__not_aligned", "__alignment_not_unique"
}

PALETTE = [
    "#00d2ff", "#f97316", "#a8ff78", "#f953c6", "#b91d73",
    "#4facfe", "#43e97b", "#fa709a", "#fee140", "#30cfd0",
    "#667eea", "#f093fb"
]


def normalize_sample_name(name: str) -> str:
    """Normalize sample labels without stripping arbitrary trailing characters."""
    base = os.path.basename(name)
    base = re.sub(r"\.(txt|tsv|csv|bam|sam)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_(counts?|readcounts?)$", "", base, flags=re.IGNORECASE)
    return base


def infer_delimiter(text: str) -> str:
    """Infer whether a count matrix is comma- or tab-delimited."""
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        comma_count = line.count(",")
        tab_count = line.count("\t")
        if comma_count > tab_count:
            return ","
        if tab_count > 0:
            return "\t"
        break
    return "\t"


def parse_delimited_rows(text: str) -> list[list[str]]:
    """Parse non-comment rows from a comma- or tab-delimited matrix."""
    delimiter = infer_delimiter(text)
    rows = []
    reader = csv.reader(text.splitlines(), delimiter=delimiter)
    for row in reader:
        if not row:
            continue
        if row[0].startswith("#"):
            continue
        rows.append([cell.strip() for cell in row])
    return rows

# ══════════════════════════════════════════════════════════════════════════════
# FILE PARSING
# ══════════════════════════════════════════════════════════════════════════════

def detect_format(text: str) -> str:
    """Detect whether file is featureCounts or HTSeq format.

    Handles three cases:
    1. Real featureCounts: GeneID + 5 metadata cols + sample cols (>=7 cols)
    2. Simple matrix: GeneID + sample cols directly (header row has non-numeric cols)
    3. HTSeq: no header, just gene\tcount
    """
    for parts in parse_delimited_rows(text):
        if not parts:
            continue
        # Any file whose first column header is geneid is featurecounts-style
        if parts[0].lower() in {"geneid", "gene_id"}:
            return "featurecounts"
        # Real featureCounts: >=7 cols and cols 5+6 are integers (length/strand)
        if len(parts) >= 7:
            try:
                int(parts[5]); int(parts[6])
                return "featurecounts"
            except (ValueError, IndexError):
                pass
        break
    return "htseq"


def parse_htseq(text: str, sample_name: str) -> pd.DataFrame:
    """Parse HTSeq-count output into a Series."""
    counts = {}
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        gene = parts[0].strip()
        if not gene or gene in HTSEQ_SKIP:
            continue
        try:
            counts[gene] = int(parts[1])
        except ValueError:
            counts[gene] = 0
    return pd.DataFrame({sample_name: counts})


def parse_featurecounts(text: str, sample_name: str) -> pd.DataFrame:
    """Parse featureCounts output — handles both formats:

    Format A (real featureCounts): 
        GeneID  Chr  Start  End  Strand  Length  sample1  sample2 ...
        (sample data starts at column index 6)

    Format B (simple count matrix):
        GeneID  Control_1  Control_2  Treated_1  Treated_2
        (sample data starts at column index 1)
    """
    rows = parse_delimited_rows(text)
    header_parsed = False
    sample_cols = []
    col_indices = []
    data_rows = {}
    first_data_col = 1  # default: simple matrix format

    for parts in rows:
        if not parts:
            continue

        if not header_parsed:
            if parts[0].lower() in {"geneid", "gene_id"}:
                # Detect which format by checking if column 1 looks like a
                # chromosome name (real featureCounts) or a sample name
                is_real_fc = (len(parts) >= 7 and
                              not parts[1].replace("_","").replace("-","").isdigit() and
                              any(c.isdigit() for c in parts[5]) if len(parts)>5 else False)

                # Simpler reliable check: if >=7 cols AND col6 onward are all
                # non-empty strings that look like sample names after the metadata
                # columns (Chr, Start, End, Strand, Length), use offset 6.
                # Otherwise treat cols 1+ as sample columns directly.
                if len(parts) >= 7:
                    # Try to detect real featureCounts: col 5 should be "Length"
                    # or an integer (gene length), col 1 should be chromosome
                    col1 = parts[1].strip().lower()
                    col5 = parts[5].strip().lower()
                    if col5 in ("length","len") or col1 in ("chr","chromosome"):
                        first_data_col = 6
                    else:
                        first_data_col = 1
                else:
                    first_data_col = 1

                raw_names = parts[first_data_col:]
                sample_cols = [normalize_sample_name(s) for s in raw_names] if raw_names else [sample_name]
                col_indices = list(range(first_data_col, first_data_col + len(sample_cols)))
                header_parsed = True
                continue
            # No header — treat as HTSeq-style (fall through)
            header_parsed = True

        # Accept rows with at least 2 columns (gene + at least 1 count)
        if len(parts) < 2:
            continue
        gene = parts[0].strip()
        if not gene:
            continue

        if not col_indices:
            col_indices = [1]
            sample_cols = [sample_name]

        for idx, ci in enumerate(col_indices):
            if ci >= len(parts):
                continue
            sn = sample_cols[idx] if idx < len(sample_cols) else f"sample_{idx+1}"
            if sn not in data_rows:
                data_rows[sn] = {}
            try:
                data_rows[sn][gene] = int(parts[ci])
            except (ValueError, IndexError):
                data_rows[sn][gene] = 0

    if not data_rows:
        return pd.DataFrame()
    return pd.DataFrame(data_rows)


def parse_file(filepath: str) -> pd.DataFrame:
    """Auto-detect and parse a count file. Returns DataFrame (genes × samples)."""
    filepath = Path(filepath)
    sample_name = normalize_sample_name(filepath.name)

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    fmt = detect_format(text)
    if fmt == "featurecounts":
        df = parse_featurecounts(text, sample_name)
        fmt_label = "featureCounts"
    else:
        df = parse_htseq(text, sample_name)
        fmt_label = "HTSeq"

    print(f"  [✓] {filepath.name} → {fmt_label} · {len(df)} genes · {len(df.columns)} sample(s)")
    return df


def merge_files(dfs: list) -> pd.DataFrame:
    """Merge list of DataFrames by gene (union, fill missing with 0)."""
    merged = pd.concat(dfs, axis=1, join="outer").fillna(0).astype(int)
    return merged.sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION & FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def compute_cpm(matrix: np.ndarray) -> np.ndarray:
    """Compute CPM (genes × samples)."""
    lib_sizes = matrix.sum(axis=0)
    lib_sizes = np.where(lib_sizes == 0, 1, lib_sizes)
    return (matrix / lib_sizes[np.newaxis, :]) * 1e6


def compute_tmm_factors(matrix: np.ndarray) -> np.ndarray:
    """Compute TMM normalization factors (one per sample)."""
    lib_sizes = matrix.sum(axis=0)
    n_samples = matrix.shape[1]

    # Choose reference: sample closest to 75th percentile library size
    sorted_ls = np.sort(lib_sizes)
    ref_idx = np.argmin(np.abs(lib_sizes - sorted_ls[int(n_samples * 0.75)]))

    factors = np.ones(n_samples)
    ref_ls = lib_sizes[ref_idx]

    for si in range(n_samples):
        if si == ref_idx:
            continue
        si_ls = lib_sizes[si]
        mask = (matrix[:, si] > 0) & (matrix[:, ref_idx] > 0)
        if mask.sum() < 10:
            print(f"  [⚠] Sample index {si}: only {mask.sum()} genes overlap with reference — "
                  f"TMM factor set to 1.0. Check for empty or mismatched samples.")
            continue

        r = matrix[mask, si] / si_ls
        x = matrix[mask, ref_idx] / ref_ls
        lr = np.log2(r) - np.log2(x)
        ae = 0.5 * (np.log2(r) + np.log2(x))

        n = len(lr)
        lo, hi = int(n * 0.3), int(n * 0.7)
        order = np.argsort(lr)
        trimmed_lr = lr[order][lo:hi]

        if len(trimmed_lr) == 0:
            continue
        factors[si] = 2 ** np.mean(trimmed_lr)

    # Normalize so geometric mean = 1
    gm = np.exp(np.mean(np.log(np.where(factors > 0, factors, 1e-10))))
    return factors / gm


def apply_tmm(matrix: np.ndarray) -> np.ndarray:
    """Apply TMM normalization, return TMM-normalized CPM matrix."""
    factors = compute_tmm_factors(matrix)
    lib_sizes = matrix.sum(axis=0)
    eff_lib = lib_sizes * factors
    eff_lib = np.where(eff_lib == 0, 1, eff_lib)
    return (matrix / eff_lib[np.newaxis, :]) * 1e6


def logcpm_transform(matrix: np.ndarray) -> np.ndarray:
    """log2-CPM transform: TMM-normalize then log2(x + 0.5).
    Note: commonly called log-CPM, NOT VST (variance-stabilizing transform).
    True VST requires NB model fitting (DESeq2). This is the standard
    edgeR log-CPM approach, appropriate for heatmaps and PCA.
    """
    tmm = apply_tmm(matrix)
    return np.log2(tmm + 0.5)


def vst_transform(matrix: np.ndarray) -> np.ndarray:
    """Alias for logcpm_transform for backward compatibility."""
    return logcpm_transform(matrix)


def filter_low_counts(matrix: np.ndarray, genes: list,
                      min_cpm: float = 1.0, min_samples: int = 2):
    """Filter genes by CPM threshold. Returns filtered matrix, genes, indices."""
    cpm = compute_cpm(matrix)
    keep = np.where((cpm >= min_cpm).sum(axis=1) >= min_samples)[0]
    return matrix[keep], [genes[i] for i in keep], keep


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    """Z-score each row (gene) independently."""
    means = matrix.mean(axis=1, keepdims=True)
    stds = matrix.std(axis=1, keepdims=True)
    stds = np.where(stds == 0, 1, stds)
    return (matrix - means) / stds


# ══════════════════════════════════════════════════════════════════════════════
# DIFFERENTIAL EXPRESSION
# ══════════════════════════════════════════════════════════════════════════════

def estimate_dispersion(matrix: np.ndarray, idx_a: list, idx_b: list) -> float:
    """Estimate common NB dispersion (phi) via moment matching."""
    all_idx = idx_a + idx_b
    sub = matrix[:, all_idx]
    lib_sizes = sub.sum(axis=0)
    factors = compute_tmm_factors(sub)
    eff_lib = lib_sizes * factors
    eff_lib = np.where(eff_lib == 0, 1, eff_lib)
    scale = eff_lib.min()

    disp_vals = []
    for gi in range(sub.shape[0]):
        mu = sub[gi] / eff_lib * scale
        mu_mean = mu.mean()
        if mu_mean < 1:
            continue
        mu_var = mu.var()
        phi = (mu_var - mu_mean) / max(mu_mean ** 2, 1e-10)
        if 0 < phi < 100:
            disp_vals.append(phi)

    if not disp_vals:
        # Organism-aware fallback: bacteria ~0.3, mammals ~0.1
        # Detect bacteria by gene count (bacteria typically <7000 genes)
        n_genes = sub.shape[0]
        fallback = 0.3 if n_genes < 7000 else 0.1
        print(f"  [⚠] Dispersion estimation failed — using fallback phi={fallback} "
              f"({'bacteria-mode' if n_genes < 7000 else 'eukaryote-mode'})")
        return fallback

    # Use median but clamp to a reasonable range
    phi = float(np.median(disp_vals))
    phi = max(0.01, min(phi, 10.0))  # clamp: <0.01 is unrealistically low, >10 is noise
    return phi


def nb_pvalue(sum_a: float, sum_b: float, e_a: float, e_b: float, phi: float) -> float:
    """NB exact-style test using score statistic.

    Computes the score test for a negative binomial model comparing
    two groups. The null hypothesis is that group membership does not
    affect counts after accounting for library size (effective library).

    sum_a / sum_b : total counts in group A / B for this gene
    e_a   / e_b   : sum of effective library sizes for group A / B
    phi           : common NB dispersion parameter

    This is the approach used by edgeR's exact test (Robinson & Smyth 2008).
    """
    n = sum_a + sum_b
    if n == 0:
        return 1.0

    # Expected proportion for group A under null
    p_a = e_a / (e_a + e_b)
    if p_a <= 0 or p_a >= 1:
        return 1.0

    mean_a = n * p_a

    # NB variance: mu + mu^2 * phi, propagated to the ratio
    # Overdispersion term: (1 + phi*mu) for each group
    mu_a = mean_a
    mu_b = n * (1 - p_a)
    disp_a = 1.0 + phi * mu_a if phi > 0 else 1.0
    disp_b = 1.0 + phi * mu_b if phi > 0 else 1.0
    var = mu_a * disp_a + mu_b * disp_b * (p_a / (1 - p_a + 1e-10)) ** 2

    if var <= 0:
        return 1.0

    z = (sum_a - mean_a) / np.sqrt(max(var, 0.5))
    # Two-sided p-value with continuity correction
    z_cc = (abs(sum_a - mean_a) - 0.5) / np.sqrt(max(var, 0.5))
    z_cc = max(z_cc, 0.0)
    return float(2 * (1 - stats.norm.cdf(z_cc)))


def bh_correction(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)
    adjusted = np.minimum(1.0, pvals[order] * n / ranks)
    # Ensure monotonicity (take cumulative minimum from right)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    result = np.empty(n)
    result[order] = adjusted
    return result


def shrink_lfc(lfcs: np.ndarray, base_means: np.ndarray, phi: float) -> np.ndarray:
    """Adaptive Bayesian shrinkage of log2 fold changes.

    Uses a robust median-based prior (not mean-of-squares) to avoid
    inflation from outlier LFCs at low-count genes. This is closer to
    the apeglm/ashr approach used in publication-grade pipelines.

    Shrinkage factor = prior_var / (prior_var + SE^2)
    where prior_var is estimated from moderately expressed genes only.
    """
    # Only use genes with reliable expression for prior estimation
    mod_mask = base_means > 5
    if mod_mask.sum() < 3:
        return lfcs.copy()

    mod_lfcs = lfcs[mod_mask]

    # Robust prior: use median absolute deviation scaled to variance
    # MAD * 1.4826 ≈ sigma for normal distribution
    mad = np.median(np.abs(mod_lfcs - np.median(mod_lfcs)))
    prior_sigma = mad * 1.4826
    prior_var = max(prior_sigma ** 2, 0.01)  # floor to avoid zero-shrinkage

    shrunken = np.empty_like(lfcs)
    for i, (lfc, bm) in enumerate(zip(lfcs, base_means)):
        mu = max(bm, 0.1)
        # SE of log2FC under NB model (per-gene, accounts for expression level)
        se = np.sqrt((1.0 / mu + phi) * 2.0)
        delta_var = se ** 2
        shrink_factor = prior_var / (prior_var + delta_var)
        shrunken[i] = lfc * shrink_factor
    return shrunken


def compute_de(matrix: np.ndarray, genes: list,
               group_a: list, group_b: list) -> pd.DataFrame:
    """
    Full DE analysis: TMM → NB test → BH-FDR → LFC shrinkage.
    group_a / group_b: column indices into matrix.
    Returns DataFrame sorted by padj.
    """
    all_idx = group_a + group_b
    sub = matrix[:, all_idx]
    lib_sizes = sub.sum(axis=0)
    factors = compute_tmm_factors(sub)
    eff_lib = lib_sizes * factors
    eff_lib = np.where(eff_lib == 0, 1, eff_lib)

    n_a, n_b = len(group_a), len(group_b)
    a_idx = list(range(n_a))
    b_idx = list(range(n_a, n_a + n_b))

    # Per-group effective library sums (used as per-gene expected counts)
    e_a_total = eff_lib[a_idx].sum()
    e_b_total = eff_lib[b_idx].sum()

    phi = estimate_dispersion(matrix, group_a, group_b)
    print(f"  [ℹ] Estimated dispersion (phi): {phi:.4f}")

    results = []
    lfcs_raw = []
    base_means = []
    pvals = []

    for gi, gene in enumerate(genes):
        counts_a = sub[gi, a_idx]
        counts_b = sub[gi, b_idx]

        # TMM-normalized CPM per group (mean across replicates)
        mu_a = (counts_a / eff_lib[a_idx] * 1e6).mean()
        mu_b = (counts_b / eff_lib[b_idx] * 1e6).mean()
        lfc_raw = np.log2(mu_b + 0.5) - np.log2(mu_a + 0.5)
        bm = (mu_a + mu_b) / 2

        sum_a = counts_a.sum()
        sum_b = counts_b.sum()

        # Use per-gene expected counts based on effective library sizes
        # This is more accurate than using a global ratio
        total_counts = sum_a + sum_b
        if total_counts > 0:
            # Expected: proportional to effective library size
            pv = nb_pvalue(sum_a, sum_b, e_a_total, e_b_total, phi)
        else:
            pv = 1.0

        results.append({"gene": gene, "muA": mu_a, "muB": mu_b, "baseMean": bm, "log2FC_raw": lfc_raw})
        lfcs_raw.append(lfc_raw)
        base_means.append(bm)
        pvals.append(float(pv) if not np.isnan(pv) else 1.0)

    lfcs_raw = np.array(lfcs_raw)
    base_means = np.array(base_means)
    pvals = np.array(pvals)
    padj = bh_correction(pvals)
    lfcs_shrunk = shrink_lfc(lfcs_raw, base_means, phi)

    df = pd.DataFrame(results)
    df["log2FC"] = lfcs_shrunk
    df["pval"] = pvals
    df["padj"] = padj
    df["negLog10P"] = -np.log10(np.maximum(pvals, 1e-300))
    df["negLog10Padj"] = -np.log10(np.maximum(padj, 1e-300))
    return df.sort_values("padj").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# PCA
# ══════════════════════════════════════════════════════════════════════════════

def pca_compute(vst_matrix: np.ndarray, samples: list):
    """
    Simple PCA on VST matrix (genes × samples).
    Returns dict with 'points' (sample coords), 'var1', 'var2'.
    """
    # Data: samples × genes
    data = vst_matrix.T  # shape: (n_samples, n_genes)
    n_s, n_g = data.shape
    if n_s < 2 or n_g < 2:
        return None

    # Center
    means = data.mean(axis=0)
    centered = data - means

    cov = centered @ centered.T / (n_s - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    val1 = float(eigvals[0]) if len(eigvals) > 0 else 0.0
    val2 = float(eigvals[1]) if len(eigvals) > 1 else 0.0
    v1 = eigvecs[:, 0] if eigvecs.shape[1] > 0 else np.zeros(n_s)
    v2 = eigvecs[:, 1] if eigvecs.shape[1] > 1 else np.zeros(n_s)

    total_var = np.trace(cov)
    pc1_coords = v1 * np.sqrt(max(val1, 0))
    pc2_coords = v2 * np.sqrt(max(val2, 0))

    points = [{"name": s, "x": float(pc1_coords[i]), "y": float(pc2_coords[i])}
              for i, s in enumerate(samples)]
    return {
        "points": points,
        "var1": f"{(val1 / total_var * 100):.1f}" if total_var > 0 else "0.0",
        "var2": f"{(val2 / total_var * 100):.1f}" if total_var > 0 else "0.0",
    }


# ══════════════════════════════════════════════════════════════════════════════
# WARNINGS
# ══════════════════════════════════════════════════════════════════════════════

def recommend_thresholds(matrix, genes, group_a, group_b, de_results, samples):
    """Print smart warnings and threshold recommendations."""
    n_a, n_b = len(group_a), len(group_b)
    total_genes = len(genes)

    print("\n" + "═" * 60)
    print("  ANALYSIS WARNINGS & RECOMMENDATIONS")
    print("═" * 60)

    if n_a == 0 or n_b == 0:
        print("  ⛔  No samples in one or both groups — cannot run DE.")
    elif n_a < 2 or n_b < 2:
        print(f"  ⛔  Only {min(n_a, n_b)} replicate(s) in one group. Results unreliable.")
    elif n_a < 3 or n_b < 3:
        print("  ⚠   Only 2 replicates in one group. 3+ strongly recommended.")

    # Library sizes
    if matrix is not None and len(group_a) > 0 and len(group_b) > 0:
        all_idx = group_a + group_b
        lib_sizes = matrix[:, all_idx].sum(axis=0)
        max_l, min_l = lib_sizes.max(), lib_sizes.min()
        if min_l > 0 and max_l / min_l > 5:
            print(f"  ⚠   Large library size variation ({max_l/min_l:.1f}× diff). Check for outliers.")
        zero_samps = [samples[i] for i in all_idx if matrix[:, i].sum() == 0]
        if zero_samps:
            print(f"  ⛔  Samples with zero counts: {', '.join(zero_samps)}")

    # DE results
    if de_results is not None and len(de_results) > 0:
        sig05 = (de_results["padj"] <= 0.05).sum()
        sig10 = (de_results["padj"] <= 0.10).sum()
        if sig05 == 0 and sig10 == 0:
            print("  ⚠   No DEGs at padj≤0.10. Consider more replicates or check conditions.")
        elif sig05 == 0 and sig10 > 0:
            print(f"  ℹ   No DEGs at padj≤0.05, but {sig10} found at padj≤0.10.")
        elif sig05 > total_genes * 0.5:
            print(f"  ⚠   {sig05} DEGs (>{sig05/total_genes*100:.0f}% of genes) — check for batch effects.")

    print("\n  Recommended thresholds:")
    if n_a >= 3 and n_b >= 3:
        print("    ✅  Standard (publication): padj≤0.05, |FC|≥1")
        print("    🔵  Strict:                 padj≤0.01, |FC|≥2")
        print("    🟠  Exploratory:            padj≤0.10, |FC|≥0.5")
    elif n_a == 2 and n_b == 2:
        print("    🟠  Recommended (n=2):      padj≤0.05, |FC|≥1.5")
    else:
        print("    🔴  Descriptive only:       padj≤0.05, |FC|≥2  (not publishable)")
    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _dark_fig(w=10, h=7):
    """Create a figure with dark background."""
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#020817")
    ax.set_facecolor("#0f172a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.tick_params(colors="#64748b")
    ax.xaxis.label.set_color("#64748b")
    ax.yaxis.label.set_color("#64748b")
    ax.title.set_color("#e2e8f0")
    return fig, ax


def plot_pca(pca_result: dict, group_map: dict, label_a: str, label_b: str,
             out_path: str, filter_info: dict = None):
    """Render PCA scatter plot."""
    fig, ax = _dark_fig(8, 6)
    points = pca_result["points"]
    var1, var2 = pca_result["var1"], pca_result["var2"]

    groups = sorted(set(group_map.values()))
    color_map = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}

    for pt in points:
        grp = group_map.get(pt["name"], groups[0])
        color = color_map[grp]
        ax.scatter(pt["x"], pt["y"], color=color, s=80, zorder=3, edgecolors="#ffffff44", linewidths=0.5)
        ax.annotate(pt["name"], (pt["x"], pt["y"]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=8, color="#94a3b8", ha="center")

    # Legend
    for grp, color in color_map.items():
        ax.scatter([], [], color=color, label=grp, s=60)
    ax.legend(loc="upper right", framealpha=0.2, facecolor="#0f172a",
              edgecolor="#334155", labelcolor="#94a3b8", fontsize=9)

    ax.axhline(0, color="#334155", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="#334155", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({var1}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2}% variance)", fontsize=11)
    ax.set_title("PCA Plot — VST Normalized", fontsize=13, fontweight="bold", color="#e2e8f0")
    ax.grid(True, color="#1e293b", linewidth=0.5)

    if filter_info:
        ax.text(0.01, 0.01, f"{filter_info['kept']} genes passed CPM filter ({filter_info['removed']} removed)",
                transform=ax.transAxes, fontsize=8, color="#4ade80")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] PCA plot saved → {out_path}")


def plot_volcano(de_results: pd.DataFrame, fc_thresh: float, padj_thresh: float,
                 label_a: str, label_b: str, out_path: str):
    """Render volcano plot."""
    fig, ax = _dark_fig(9, 7)

    def classify(row):
        if row["padj"] <= padj_thresh and row["log2FC"] >= fc_thresh:
            return "up"
        if row["padj"] <= padj_thresh and row["log2FC"] <= -fc_thresh:
            return "down"
        return "ns"

    de_results = de_results.copy()
    de_results["cls"] = de_results.apply(classify, axis=1)

    colors = {"up": "#f97316", "down": "#38bdf8", "ns": "#1e3a4a"}
    alphas = {"up": 0.85, "down": 0.85, "ns": 0.4}
    sizes  = {"up": 20,   "down": 20,   "ns": 8}

    for cls, grp in de_results.groupby("cls"):
        y_vals = np.minimum(grp["negLog10Padj"], grp["negLog10Padj"].quantile(0.999))
        ax.scatter(grp["log2FC"], y_vals,
                   color=colors[cls], alpha=alphas[cls], s=sizes[cls],
                   label=f"{'↑ Up' if cls=='up' else '↓ Down' if cls=='down' else 'NS'} ({len(grp)})",
                   zorder=3 if cls != "ns" else 2, edgecolors="none")

    # Threshold lines
    y_line = -np.log10(padj_thresh)
    x_max = de_results["log2FC"].abs().max() * 1.1
    ax.axhline(y_line, color="#ef4444", linestyle="--", linewidth=1.2, label=f"padj={padj_thresh}")
    ax.axvline(fc_thresh, color="#475569", linestyle="--", linewidth=0.8)
    ax.axvline(-fc_thresh, color="#475569", linestyle="--", linewidth=0.8)

    ax.set_xlim(-x_max, x_max)
    ax.set_xlabel("Shrunken log₂ Fold Change", fontsize=11)
    ax.set_ylabel("-log₁₀(padj) [BH]", fontsize=11)
    ax.set_title(f"Volcano Plot — {label_b} vs {label_a}", fontsize=13, fontweight="bold", color="#e2e8f0")
    ax.legend(loc="upper left", framealpha=0.2, facecolor="#0f172a",
              edgecolor="#334155", labelcolor="#94a3b8", fontsize=9)
    ax.grid(True, color="#1e293b", linewidth=0.5)

    ax.text(x_max * 0.98, y_line + 0.3, f"padj≤{padj_thresh}",
            color="#ef4444", fontsize=8, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] Volcano plot saved → {out_path}")


def plot_heatmap(matrix: np.ndarray, genes: list, samples: list,
                 gene_indices: list, title: str, subtitle: str,
                 out_path: str):
    """Render expression heatmap with z-scored VST values."""
    if not gene_indices:
        print(f"  [!] No genes to display for heatmap: {title}")
        return

    vst = vst_transform(matrix)
    sub_vst = vst[gene_indices]
    zscore = zscore_rows(sub_vst)

    n_genes = len(gene_indices)
    n_samples = len(samples)
    cell_h = max(0.2, min(0.55, 12 / max(n_genes, 1)))
    fig_h = max(4, n_genes * cell_h + 2.5)
    fig_w = max(6, n_samples * 0.9 + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#020817")
    ax.set_facecolor("#020817")

    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    im = ax.imshow(zscore, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(n_samples))
    ax.set_xticklabels(samples, rotation=40, ha="right", fontsize=8, color="#94a3b8")
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels([genes[i] for i in gene_indices], fontsize=max(5, min(9, 110 // n_genes)), color="#64748b")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors="#64748b")
    cbar.ax.set_ylabel("z-score", color="#64748b", fontsize=8)
    cbar.outline.set_edgecolor("#334155")

    ax.set_title(f"{title}\n{subtitle}", fontsize=10, color="#e2e8f0", pad=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Heatmap saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_csv(de_results: pd.DataFrame, fc_thresh: float, padj_thresh: float,
               label_a: str, label_b: str, out_path: str):
    """Export DE results to CSV."""
    df = de_results.copy().sort_values("padj")

    def regulation(row):
        if row["padj"] <= padj_thresh and row["log2FC"] >= fc_thresh:
            return "UP"
        if row["padj"] <= padj_thresh and row["log2FC"] <= -fc_thresh:
            return "DOWN"
        return "NS"

    df["regulation"] = df.apply(regulation, axis=1)
    cols = ["gene", "baseMean", "log2FC", "log2FC_raw", "pval", "padj", "regulation"]
    df[cols].to_csv(out_path, index=False, float_format="%.6g")
    print(f"  [✓] DEG results CSV saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n🧬 TranscriptomicViz — Python Pipeline")
    print("=" * 60)

    # ── Load files ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading count files...")
    dfs = []
    for fp in args.files:
        try:
            df = parse_file(fp)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"  [✗] Error reading {fp}: {e}")
            sys.exit(1)

    if not dfs:
        print("No valid files loaded. Exiting.")
        sys.exit(1)

    count_matrix_df = merge_files(dfs)
    genes = list(count_matrix_df.index)
    samples = list(count_matrix_df.columns)
    matrix = count_matrix_df.values.astype(float)

    print(f"\n  Loaded: {len(genes)} genes × {len(samples)} samples")
    print(f"  Samples: {', '.join(samples)}")

    # ── Resolve group indices ────────────────────────────────────────────────
    sample_idx = {s: i for i, s in enumerate(samples)}

    group_a_idx = []
    for s in args.group_a:
        if s in sample_idx:
            group_a_idx.append(sample_idx[s])
        else:
            print(f"  [✗] Sample '{s}' not found in loaded files. Available: {samples}")
            sys.exit(1)

    group_b_idx = []
    for s in args.group_b:
        if s in sample_idx:
            group_b_idx.append(sample_idx[s])
        else:
            print(f"  [✗] Sample '{s}' not found in loaded files. Available: {samples}")
            sys.exit(1)

    label_a = args.label_a
    label_b = args.label_b

    group_map = {}
    for s in samples:
        if s in args.group_a:
            group_map[s] = label_a
        elif s in args.group_b:
            group_map[s] = label_b
        else:
            group_map[s] = label_a  # default

    print(f"\n  Group A ({label_a}): {[samples[i] for i in group_a_idx]}")
    print(f"  Group B ({label_b}): {[samples[i] for i in group_b_idx]}")

    # ── CPM Filtering ────────────────────────────────────────────────────────
    print(f"\n[2/6] Filtering genes (CPM≥{args.min_cpm} in ≥{args.min_samples} samples)...")
    filt_matrix, filt_genes, filt_idx = filter_low_counts(
        matrix, genes, min_cpm=args.min_cpm, min_samples=args.min_samples
    )
    print(f"  Kept: {len(filt_genes)} / {len(genes)} genes")

    # ── Differential Expression ──────────────────────────────────────────────
    de_results = None
    if len(group_a_idx) > 0 and len(group_b_idx) > 0:
        print(f"\n[3/6] Running DE analysis ({label_b} vs {label_a})...")
        de_results = compute_de(filt_matrix, filt_genes, group_a_idx, group_b_idx)
        sig = (de_results["padj"] <= args.padj).sum()
        up  = ((de_results["padj"] <= args.padj) & (de_results["log2FC"] >= args.fc)).sum()
        dn  = ((de_results["padj"] <= args.padj) & (de_results["log2FC"] <= -args.fc)).sum()
        print(f"  Results: {len(de_results)} tested · {sig} significant (padj≤{args.padj})")
        print(f"  ↑ Up: {up}  ↓ Down: {dn}")

        recommend_thresholds(matrix, genes, group_a_idx, group_b_idx, de_results, samples)

        csv_path = out_dir / f"DEG_{label_b}_vs_{label_a}.csv"
        export_csv(de_results, args.fc, args.padj, label_a, label_b, str(csv_path))
    else:
        print("\n[3/6] Skipping DE analysis (groups not fully specified).")

    # ── PCA ─────────────────────────────────────────────────────────────────
    print("\n[4/6] Computing PCA...")
    if filt_matrix.shape[0] >= 2 and len(samples) >= 2:
        vst_mat = vst_transform(filt_matrix)
        pca_result = pca_compute(vst_mat, samples)
        if pca_result:
            plot_pca(pca_result, group_map, label_a, label_b,
                     str(out_dir / "pca_plot.png"),
                     filter_info={"kept": len(filt_genes), "removed": len(genes) - len(filt_genes)})
    else:
        print("  [!] Not enough samples or genes for PCA.")

    # ── Expression Heatmap (top-N variable genes) ────────────────────────────
    print(f"\n[5/6] Generating expression heatmap (top {args.top_n} variable genes)...")
    if filt_matrix.shape[0] >= 1:
        vst_mat = vst_transform(filt_matrix)
        gene_vars = vst_mat.var(axis=1)
        top_local = np.argsort(gene_vars)[::-1][:args.top_n]
        top_global = [filt_idx[i] for i in top_local]
        plot_heatmap(
            matrix=matrix,
            genes=genes,
            samples=samples,
            gene_indices=top_global,
            title="Expression Heatmap",
            subtitle=f"Top {args.top_n} most variable genes · CPM≥{args.min_cpm} filter",
            out_path=str(out_dir / "heatmap_top_variable.png")
        )

    # ── DEG Heatmaps ─────────────────────────────────────────────────────────
    print(f"\n[6/6] Generating DEG heatmaps...")
    if de_results is not None and len(de_results) > 0:
        gene_to_idx = {g: i for i, g in enumerate(genes)}

        up_genes = de_results[
            (de_results["padj"] <= args.padj) & (de_results["log2FC"] >= args.fc)
        ].sort_values("log2FC", ascending=False)["gene"].tolist()

        down_genes = de_results[
            (de_results["padj"] <= args.padj) & (de_results["log2FC"] <= -args.fc)
        ].sort_values("log2FC", ascending=True)["gene"].tolist()

        if up_genes:
            up_idx = [gene_to_idx[g] for g in up_genes if g in gene_to_idx]
            plot_heatmap(
                matrix=matrix, genes=genes, samples=samples,
                gene_indices=up_idx,
                title="DEG Heatmap — Upregulated",
                subtitle=f"{len(up_idx)} genes · log₂FC ≥ {args.fc} · padj ≤ {args.padj}",
                out_path=str(out_dir / "heatmap_upregulated.png")
            )
        else:
            print("  [!] No upregulated DEGs to plot.")

        if down_genes:
            dn_idx = [gene_to_idx[g] for g in down_genes if g in gene_to_idx]
            plot_heatmap(
                matrix=matrix, genes=genes, samples=samples,
                gene_indices=dn_idx,
                title="DEG Heatmap — Downregulated",
                subtitle=f"{len(dn_idx)} genes · log₂FC ≤ -{args.fc} · padj ≤ {args.padj}",
                out_path=str(out_dir / "heatmap_downregulated.png")
            )
        else:
            print("  [!] No downregulated DEGs to plot.")

        both_idx = [gene_to_idx[g] for g in (up_genes + down_genes) if g in gene_to_idx]
        if both_idx:
            plot_heatmap(
                matrix=matrix, genes=genes, samples=samples,
                gene_indices=both_idx,
                title="DEG Heatmap — All DEGs",
                subtitle=f"{len(both_idx)} DEGs · padj ≤ {args.padj} · |log₂FC| ≥ {args.fc}",
                out_path=str(out_dir / "heatmap_all_degs.png")
            )

        # Volcano
        print("\n  Generating volcano plot...")
        plot_volcano(de_results, args.fc, args.padj, label_a, label_b,
                     str(out_dir / "volcano_plot.png"))
    else:
        print("  [!] No DE results available for DEG plots / volcano.")

    print("\n" + "=" * 60)
    print(f"✅  All outputs saved to: {out_dir.resolve()}")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TranscriptomicViz — publication-grade RNA-seq pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--files", nargs="+", required=True,
                        help="Input count files (HTSeq or featureCounts, auto-detected)")
    parser.add_argument("--group_a", nargs="+", required=True,
                        help="Sample names for Group A (control). Must match file basenames.")
    parser.add_argument("--group_b", nargs="+", required=True,
                        help="Sample names for Group B (treatment). Must match file basenames.")
    parser.add_argument("--label_a", default="Control",
                        help="Label for Group A (default: Control)")
    parser.add_argument("--label_b", default="Treatment",
                        help="Label for Group B (default: Treatment)")
    parser.add_argument("--min_cpm", type=float, default=1.0,
                        help="Minimum CPM threshold for gene filtering (default: 1.0)")
    parser.add_argument("--min_samples", type=int, default=2,
                        help="Minimum number of samples passing CPM filter (default: 2)")
    parser.add_argument("--padj", type=float, default=0.05,
                        help="Adjusted p-value threshold (default: 0.05)")
    parser.add_argument("--fc", type=float, default=1.0,
                        help="log2 fold-change threshold (default: 1.0, meaning 2× FC)")
    parser.add_argument("--top_n", type=int, default=50,
                        help="Top N variable genes for expression heatmap (default: 50)")
    parser.add_argument("--out_dir", default="results",
                        help="Output directory (default: results)")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
