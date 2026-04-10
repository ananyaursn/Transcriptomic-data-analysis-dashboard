"""
adaptive_intelligence.py
━━━━━━━━━━━━━━━━━━━━━━━━
Level 1 Intelligence — Data-Adaptive Pipeline

This module inspects your uploaded count data and automatically:
  • Detects organism from gene ID patterns
  • Estimates optimal CPM cutoff from count distribution
  • Detects outlier samples statistically (not just a ratio rule)
  • Recommends group assignments if sample names follow conventions
  • Adapts dispersion strategy based on replicate count
  • Produces a plain-English data summary before any analysis

Usage (standalone):
    from adaptive_intelligence import DataAdvisor
    advisor = DataAdvisor(matrix, genes, samples)
    report  = advisor.full_report()

Usage (in Streamlit):
    Run adaptive_streamlit_app.py
"""

import re
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster


# ══════════════════════════════════════════════════════════════════════════════
# GENE ID PATTERN DATABASE
# ══════════════════════════════════════════════════════════════════════════════

GENE_ID_PATTERNS = [
    # Bacteria — Enterobacteriaceae / E. coli style
    ("Enterobacter / Klebsiella",  r"^ECL_\d{5}$",           {"min_cpm": 3.0, "fc": 1.5, "padj": 0.05, "top_n": 30}),
    ("Enterobacter cloacae",       r"^ECNIH\d+$",             {"min_cpm": 3.0, "fc": 1.5, "padj": 0.05, "top_n": 30}),
    ("E. coli (K-12)",             r"^b\d{4}$",               {"min_cpm": 5.0, "fc": 1.5, "padj": 0.05, "top_n": 25}),
    ("E. coli (generic locus)",    r"^[A-Z]{2,6}_RS\d{5}$",  {"min_cpm": 5.0, "fc": 1.5, "padj": 0.05, "top_n": 25}),
    ("E. coli (gene name)",        r"^[a-z]{3}[A-Z]$",       {"min_cpm": 5.0, "fc": 1.5, "padj": 0.05, "top_n": 25}),
    ("Pseudomonas aeruginosa",     r"^PA\d{4}$",              {"min_cpm": 5.0, "fc": 1.5, "padj": 0.05, "top_n": 25}),
    ("Bacillus subtilis",          r"^BSU_\d{5}$",            {"min_cpm": 5.0, "fc": 1.5, "padj": 0.05, "top_n": 25}),
    ("Mycobacterium tuberculosis", r"^Rv\d{4}[A-Za-z]?$",    {"min_cpm": 5.0, "fc": 1.5, "padj": 0.05, "top_n": 20}),
    ("Salmonella",                 r"^STM\d{4}$",             {"min_cpm": 5.0, "fc": 1.5, "padj": 0.05, "top_n": 25}),
    ("Staphylococcus aureus",      r"^SAUSA\d{6}$",           {"min_cpm": 5.0, "fc": 1.5, "padj": 0.05, "top_n": 25}),
    # Yeast
    ("S. cerevisiae",              r"^YAL\d{3}[A-Z]$|^Y[A-P][LR]\d{3}[A-Z]$", {"min_cpm": 2.0, "fc": 1.0, "padj": 0.05, "top_n": 30}),
    ("S. pombe",                   r"^SPAC\d+\.\d+[a-z]?$",  {"min_cpm": 2.0, "fc": 1.0, "padj": 0.05, "top_n": 30}),
    # Fungi
    ("Aspergillus",                r"^AFUA_\d+G\d+$|^An\d+g\d+$", {"min_cpm": 1.0, "fc": 1.0, "padj": 0.05, "top_n": 35}),
    # Plants
    ("Arabidopsis thaliana",       r"^AT[1-5MC]G\d{5}$",     {"min_cpm": 0.5, "fc": 1.0, "padj": 0.05, "top_n": 50}),
    ("Rice (Oryza sativa)",        r"^Os\d{2}g\d+$",          {"min_cpm": 0.5, "fc": 1.0, "padj": 0.05, "top_n": 50}),
    # Metazoans
    ("Human (Ensembl)",            r"^ENSG\d{11}$",           {"min_cpm": 1.0, "fc": 1.0, "padj": 0.05, "top_n": 50}),
    ("Mouse (Ensembl)",            r"^ENSMUSG\d{11}$",        {"min_cpm": 1.0, "fc": 1.0, "padj": 0.05, "top_n": 50}),
    ("Zebrafish (Ensembl)",        r"^ENSDARG\d{11}$",        {"min_cpm": 1.0, "fc": 1.0, "padj": 0.05, "top_n": 50}),
    ("Drosophila",                 r"^FBgn\d{7}$",            {"min_cpm": 1.0, "fc": 1.0, "padj": 0.05, "top_n": 40}),
    ("C. elegans",                 r"^WBGene\d{8}$",          {"min_cpm": 1.0, "fc": 1.0, "padj": 0.05, "top_n": 40}),
    ("Rat (Ensembl)",              r"^ENSRNOG\d{11}$",        {"min_cpm": 1.0, "fc": 1.0, "padj": 0.05, "top_n": 50}),
    ("Human/Mouse (HGNC symbol)",  r"^[A-Z][A-Z0-9]{1,8}$",  {"min_cpm": 1.0, "fc": 1.0, "padj": 0.05, "top_n": 50}),
]

SAMPLE_NAME_PATTERNS = [
    # SRA accessions
    (r"^SRR\d+$",   "SRA run accession — group by biological condition, not by accession number"),
    (r"^ERR\d+$",   "ENA run accession — group by biological condition"),
    (r"^DRR\d+$",   "DDBJ run accession — group by biological condition"),
    # Common conventions
    (r"(?i)ctrl|control|untreated|wt|wildtype|wild.type|mock|vehicle|baseline", "likely Control / reference group"),
    (r"(?i)treat|drug|mut|mutant|stress|infected|exposed|stimulated|ko|knockout", "likely Treatment group"),
    (r"(?i)_r(\d)|rep(\d)|replicate(\d)|_(\d)$", "replicate numbering detected"),
]


# ══════════════════════════════════════════════════════════════════════════════
# CORE ADVISOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class DataAdvisor:
    """
    Inspects count data and returns adaptive recommendations.
    All methods are pure — no side effects.
    """

    def __init__(self, matrix: np.ndarray, genes: list, samples: list):
        self.matrix  = matrix.astype(float)
        self.genes   = genes
        self.samples = samples
        self.n_genes   = len(genes)
        self.n_samples = len(samples)
        self._cpm = None  # lazy

    # ── CPM (cached) ──────────────────────────────────────────────────────────
    @property
    def cpm(self) -> np.ndarray:
        if self._cpm is None:
            lib = self.matrix.sum(axis=0)
            lib = np.where(lib == 0, 1, lib)
            self._cpm = (self.matrix / lib[np.newaxis, :]) * 1e6
        return self._cpm

    @staticmethod
    def _prettify_group_label(label: str) -> str:
        parts = [p for p in re.split(r"[_\-\s]+", label) if p]
        pretty = []
        for part in parts:
            if part.isupper() and len(part) <= 5:
                pretty.append(part)
            elif re.search(r"\d", part):
                pretty.append(part[:1].upper() + part[1:])
            else:
                pretty.append(part.capitalize())
        return " ".join(pretty) if pretty else label

    @classmethod
    def _infer_name_group(cls, sample_name: str) -> tuple[str | None, str]:
        normalized = re.sub(r"[\s.\-]+", "_", sample_name.strip()).strip("_")
        normalized_lower = normalized.lower()

        control_pattern = r"(^|_)(ctrl|control|untreated|vehicle|mock|baseline|wt|wildtype|wild_type|sensitive)($|_)"
        if re.search(control_pattern, normalized_lower):
            return "Control", "HIGH"

        stem = re.sub(r"(?i)(?:_|-)?(?:rep(?:licate)?|sample|r)\d+$", "", normalized).strip("_- ")
        stem_lower = stem.lower()

        if re.search(control_pattern, stem_lower):
            return "Control", "HIGH"

        if not stem:
            return None, "LOW"

        numbered_treatment = re.fullmatch(
            r"(?i)(?:treat(?:ment)?|treated|drug|mut(?:ant)?|stress|infected|exposed|stimulated|ko|knockout|res(?:istant)?)[_ -]?(\d+)",
            stem,
        )
        if numbered_treatment:
            return f"Treatment{numbered_treatment.group(1)}", "HIGH"

        numbered_control = re.fullmatch(
            r"(?i)(?:ctrl|control|untreated|vehicle|mock|baseline|wt|wildtype|wild_type|sensitive)[_ -]?(\d+)",
            stem,
        )
        if numbered_control:
            return f"Control{numbered_control.group(1)}", "HIGH"

        generic_treatment = re.fullmatch(
            r"(?i)(treat(?:ment)?|treated|drug|mut(?:ant)?|stress|infected|exposed|stimulated|ko|knockout|res(?:istant)?)",
            stem,
        )
        if generic_treatment:
            return "Treatment", "MEDIUM"

        informative_stem = re.sub(
            r"(?i)(?:^|_)(rna|seq|rnaseq|counts?|sample)(?:_|$)",
            "_",
            stem,
        ).strip("_")
        informative_stem = re.sub(r"_+", "_", informative_stem)
        if not informative_stem:
            return None, "LOW"

        if re.search(r"\d", informative_stem) or "_" in informative_stem or len(informative_stem) > 3:
            return cls._prettify_group_label(informative_stem), "HIGH"

        return None, "LOW"

    # ══════════════════════════════════════════════════════════════════════════
    # 1. ORGANISM DETECTION
    # ══════════════════════════════════════════════════════════════════════════

    def detect_organism(self) -> dict:
        """
        Test a random sample of gene IDs against known patterns.
        Returns best match with confidence score and recommended thresholds.
        """
        test_genes = self.genes[:500]  # test first 500 for speed
        best = {"organism": "Unknown", "confidence": 0.0,
                "thresholds": {"min_cpm": 1.0, "fc": 1.0, "padj": 0.05, "top_n": 50},
                "gene_id_format": "unrecognized",
                "reasoning": "No known gene ID pattern matched."}

        scores = []
        for name, pattern, thresholds in GENE_ID_PATTERNS:
            matches = sum(1 for g in test_genes if re.match(pattern, str(g)))
            confidence = matches / max(len(test_genes), 1)
            scores.append((confidence, name, pattern, thresholds))

        scores.sort(reverse=True)
        if scores and scores[0][0] > 0.05:  # at least 5% of genes match
            conf, name, pattern, thresholds = scores[0]
            best = {
                "organism":      name,
                "confidence":    round(conf * 100, 1),
                "thresholds":    thresholds,
                "gene_id_format": pattern,
                "reasoning":     (
                    f"{conf*100:.0f}% of sampled gene IDs match the '{name}' pattern "
                    f"({pattern}). Confidence: {'HIGH' if conf>0.6 else 'MEDIUM' if conf>0.2 else 'LOW'}."
                ),
                "alternatives":  [(s[1], round(s[0]*100,1)) for s in scores[1:4] if s[0]>0.01]
            }
        return best

    # ══════════════════════════════════════════════════════════════════════════
    # 2. ADAPTIVE CPM THRESHOLD
    # ══════════════════════════════════════════════════════════════════════════

    def suggest_cpm_threshold(self) -> dict:
        """
        Fit the CPM distribution and find the natural noise floor.
        Low-count genes form a spike near 0 — the threshold should sit
        just above this spike (the anti-mode of the distribution).
        """
        # Use median CPM per gene across samples
        median_cpm = np.median(self.cpm, axis=1)
        median_cpm = median_cpm[median_cpm > 0]

        if len(median_cpm) < 10:
            return {"recommended_cpm": 1.0, "reasoning": "Too few genes to estimate distribution."}

        log_cpm = np.log2(median_cpm + 0.01)

        # Find the valley between low-count spike and expressed genes
        # Use kernel density estimation
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(log_cpm, bw_method=0.3)
        x = np.linspace(log_cpm.min(), log_cpm.max(), 500)
        density = kde(x)

        # Find local minima in density (valleys)
        valleys = []
        for i in range(1, len(density) - 1):
            if density[i] < density[i-1] and density[i] < density[i+1]:
                valleys.append((x[i], density[i]))

        # First valley after the initial spike = noise/signal boundary
        recommended_log2 = 0.0  # default = 1 CPM
        reasoning = "No clear bimodal distribution detected — using standard CPM=1."

        if valleys:
            # First valley in range [-1, 4] (log2 CPM 0.5 to 16)
            valid = [(v, d) for v, d in valleys if -1 < v < 4]
            if valid:
                recommended_log2 = valid[0][0]
                recommended_cpm = round(float(2 ** recommended_log2), 2)
                pct_removed = float((median_cpm < recommended_cpm).mean() * 100)
                reasoning = (
                    f"Distribution valley detected at log2-CPM = {recommended_log2:.2f} "
                    f"(≈ {recommended_cpm} CPM). This removes {pct_removed:.0f}% of genes "
                    f"(noise floor below signal peak)."
                )
                return {"recommended_cpm": recommended_cpm, "reasoning": reasoning,
                        "pct_removed": round(pct_removed, 1), "log2_valley": round(recommended_log2, 2)}

        # Fallback: use 10th percentile of expressed genes
        p10 = float(np.percentile(median_cpm[median_cpm > 0.1], 10))
        recommended_cpm = round(max(0.5, min(p10, 5.0)), 1)
        reasoning = f"Using 10th percentile of expressed genes: {recommended_cpm:.1f} CPM."
        return {"recommended_cpm": recommended_cpm, "reasoning": reasoning}

    # ══════════════════════════════════════════════════════════════════════════
    # 3. OUTLIER SAMPLE DETECTION
    # ══════════════════════════════════════════════════════════════════════════

    def detect_outlier_samples(self) -> dict:
        """
        Multi-method outlier detection:
        1. Library size Z-score
        2. Spearman correlation to median profile
        3. PCA distance from centroid
        Returns per-sample risk scores and flags.
        """
        results = {}
        lib_sizes = self.matrix.sum(axis=0)

        # Method 1: Library size Z-score
        if lib_sizes.std() > 0:
            lib_z = np.abs(stats.zscore(lib_sizes))
        else:
            lib_z = np.zeros(self.n_samples)

        # Method 2: Spearman correlation to median expression profile
        log_mat = np.log2(self.cpm + 0.5)
        median_profile = np.median(log_mat, axis=1)
        corr_to_median = []
        for si in range(self.n_samples):
            r, _ = stats.spearmanr(log_mat[:, si], median_profile)
            corr_to_median.append(float(r) if not np.isnan(r) else 0.0)
        corr_arr = np.array(corr_to_median)

        # Method 3: PCA distance from centroid
        centered = log_mat.T - log_mat.T.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            scores = U[:, :2] * S[:2]
            centroid = scores.mean(axis=0)
            pca_dist = np.linalg.norm(scores - centroid, axis=1)
            pca_dist_z = np.abs(stats.zscore(pca_dist)) if pca_dist.std() > 0 else np.zeros(self.n_samples)
        except Exception:
            pca_dist_z = np.zeros(self.n_samples)

        # Composite risk score (0-100)
        for si, s in enumerate(self.samples):
            risk_factors = []
            flags = []

            lz = float(lib_z[si])
            cr = float(corr_arr[si])
            pz = float(pca_dist_z[si])

            if lz > 2.5:
                risk_factors.append(40)
                flags.append(f"library size Z={lz:.1f} (extreme)")
            elif lz > 2.0:
                risk_factors.append(20)
                flags.append(f"library size Z={lz:.1f} (elevated)")

            if cr < 0.85:
                risk_factors.append(40)
                flags.append(f"correlation to median r={cr:.3f} (low)")
            elif cr < 0.92:
                risk_factors.append(15)
                flags.append(f"correlation to median r={cr:.3f} (borderline)")

            if pz > 2.5:
                risk_factors.append(30)
                flags.append(f"PCA outlier Z={pz:.1f}")
            elif pz > 2.0:
                risk_factors.append(15)
                flags.append(f"PCA distance Z={pz:.1f} (elevated)")

            risk_score = min(100, sum(risk_factors))
            risk_level = "HIGH" if risk_score >= 60 else "MEDIUM" if risk_score >= 30 else "LOW"

            results[s] = {
                "risk_score":      risk_score,
                "risk_level":      risk_level,
                "lib_size":        int(lib_sizes[si]),
                "lib_z":           round(lz, 2),
                "corr_to_median":  round(cr, 3),
                "pca_dist_z":      round(pz, 2),
                "flags":           flags,
                "recommendation":  (
                    "⛔ Consider removing before analysis." if risk_level == "HIGH"
                    else "⚠ Monitor — re-check after seeing PCA." if risk_level == "MEDIUM"
                    else "✓ Looks fine."
                ),
            }

        return results

    # ══════════════════════════════════════════════════════════════════════════
    # 4. SAMPLE GROUP INFERENCE
    # ══════════════════════════════════════════════════════════════════════════

    def infer_sample_groups(self) -> dict:
        """
        Try to infer groups two ways:
        A) Pattern-match sample names (ctrl/treat/rep etc.)
        B) Unsupervised clustering on expression profiles
        Returns suggested group assignments + confidence.
        """
        name_groups = {}
        name_confidence = {}

        for s in self.samples:
            inferred_group, confidence = self._infer_name_group(s)
            name_groups[s] = inferred_group
            name_confidence[s] = confidence

        # Expression clustering fallback
        log_mat = np.log2(self.cpm + 0.5).T  # samples × genes
        cluster_groups = {}

        if self.n_samples >= 4:
            try:
                # Hierarchical clustering on top 500 most variable genes
                gene_vars = log_mat.var(axis=0)
                top_idx = np.argsort(gene_vars)[::-1][:500]
                sub = log_mat[:, top_idx]
                # Normalize each sample
                sub = (sub - sub.mean(axis=1, keepdims=True)) / (sub.std(axis=1, keepdims=True) + 1e-10)
                Z = linkage(sub, method="ward", metric="euclidean")
                labels = fcluster(Z, t=2, criterion="maxclust")
                # Map cluster labels to readable names
                unique_labels = sorted(set(labels))
                label_names = ["Group A", "Group B"] + [f"Group {chr(67+i)}" for i in range(len(unique_labels)-2)]
                for si, s in enumerate(self.samples):
                    cluster_groups[s] = label_names[labels[si] - 1]
            except Exception:
                cluster_groups = {s: "Group A" for s in self.samples}
        else:
            cluster_groups = {s: "?" for s in self.samples}

        # Consensus: prefer name-based if confident, else expression-based
        final_groups = {}
        for s in self.samples:
            if name_confidence.get(s) in ("HIGH", "MEDIUM") and name_groups.get(s):
                final_groups[s] = {"group": name_groups[s], "method": "name pattern",
                                   "confidence": name_confidence[s]}
            else:
                final_groups[s] = {"group": cluster_groups.get(s, "Group A"),
                                   "method": "expression clustering", "confidence": "MEDIUM"}

        return final_groups

    # ══════════════════════════════════════════════════════════════════════════
    # 5. ADAPTIVE DISPERSION STRATEGY
    # ══════════════════════════════════════════════════════════════════════════

    def suggest_dispersion_strategy(self, n_replicates_min: int) -> dict:
        """
        Recommend dispersion estimation approach based on:
        - Number of replicates
        - Organism type (gene count proxy)
        - Count distribution characteristics
        """
        is_bacteria = self.n_genes < 7000
        is_low_rep  = n_replicates_min < 3

        # Check if counts are overdispersed
        # Sample some expressed genes and check variance > mean (NB territory)
        expressed = self.matrix[self.matrix.mean(axis=1) > 10]
        if len(expressed) > 20:
            mean_counts = expressed.mean(axis=1)
            var_counts  = expressed.var(axis=1)
            phi_est = np.median((var_counts - mean_counts) / np.maximum(mean_counts**2, 1))
            phi_est = max(0.0, float(phi_est))
        else:
            phi_est = 0.1

        if is_bacteria and is_low_rep:
            strategy = "common_dispersion"
            reason = ("Bacteria + low replicates: use common dispersion. "
                      "Per-gene dispersion is unreliable with <3 replicates.")
        elif is_bacteria:
            strategy = "trended_dispersion"
            reason = ("Bacteria with adequate replicates: use trended dispersion "
                      "(dispersion vs mean expression trend).")
        elif is_low_rep:
            strategy = "common_dispersion"
            reason = ("Low replicates (<3): common dispersion is more stable "
                      "than per-gene estimates.")
        else:
            strategy = "trended_dispersion"
            reason = "Adequate replicates: trended dispersion recommended."

        return {
            "strategy":          strategy,
            "estimated_phi":     round(phi_est, 4),
            "is_bacteria":       is_bacteria,
            "is_low_rep":        is_low_rep,
            "reasoning":         reason,
            "overdispersion_ok": phi_est > 0.01,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 6. FULL REPORT
    # ══════════════════════════════════════════════════════════════════════════

    def full_report(self, n_replicates_min: int = 2) -> dict:
        """Run all analyses and return a unified report dict."""
        organism    = self.detect_organism()
        cpm_suggest = self.suggest_cpm_threshold()
        outliers    = self.detect_outlier_samples()
        groups      = self.infer_sample_groups()
        dispersion  = self.suggest_dispersion_strategy(n_replicates_min)

        # Merge threshold recommendations
        thresholds = dict(organism["thresholds"])
        # Override CPM with data-driven estimate if organism is unknown
        if organism["confidence"] < 20:
            thresholds["min_cpm"] = cpm_suggest["recommended_cpm"]

        # Count high-risk samples
        high_risk = [s for s, r in outliers.items() if r["risk_level"] == "HIGH"]
        medium_risk = [s for s, r in outliers.items() if r["risk_level"] == "MEDIUM"]

        # Plain-English summary
        summary_lines = [
            f"Loaded {self.n_genes:,} genes × {self.n_samples} samples.",
            f"Organism detected: {organism['organism']} (confidence {organism['confidence']}%).",
            f"Recommended CPM threshold: {thresholds['min_cpm']} — {cpm_suggest['reasoning']}",
            f"Recommended |log₂FC| ≥ {thresholds['fc']}, padj ≤ {thresholds['padj']}.",
        ]
        if high_risk:
            summary_lines.append(f"⛔ HIGH-RISK samples detected: {', '.join(high_risk)}. Consider removing before DE analysis.")
        if medium_risk:
            summary_lines.append(f"⚠ MEDIUM-RISK samples to monitor: {', '.join(medium_risk)}.")
        if not high_risk and not medium_risk:
            summary_lines.append("✓ No outlier samples detected.")

        group_conf = {s: v["confidence"] for s, v in groups.items()}
        low_conf_samples = [s for s, c in group_conf.items() if c == "LOW"]
        if low_conf_samples:
            summary_lines.append(f"⚠ Could not infer group for: {', '.join(low_conf_samples)}. Please assign manually.")

        return {
            "organism":     organism,
            "cpm":          cpm_suggest,
            "outliers":     outliers,
            "groups":       groups,
            "dispersion":   dispersion,
            "thresholds":   thresholds,
            "summary":      summary_lines,
            "n_high_risk":  len(high_risk),
            "n_medium_risk":len(medium_risk),
        }
