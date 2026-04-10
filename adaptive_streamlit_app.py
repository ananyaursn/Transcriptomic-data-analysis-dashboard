"""
adaptive_streamlit_app.py  —  Level 1: Data-Adaptive Intelligence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run with: streamlit run adaptive_streamlit_app.py

What this adds over the base app:
  • Infers sample groups from names + expression clustering
  • Keeps the adaptive RNA-seq workflow in a single Streamlit app
"""

import io
import re
import datetime
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.colors import TwoSlopeNorm
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster

from transcriptomic_viz import (
    detect_format, parse_htseq, parse_featurecounts, merge_files,
    compute_cpm, vst_transform, filter_low_counts,
    zscore_rows, compute_de, pca_compute, bh_correction, PALETTE,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TranscriptomicViz — Adaptive",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background-color: #020817; color: #e2e8f0; }
  section[data-testid="stSidebar"] { background-color: #0a1628; }
  section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
  h1,h2,h3 {
    color: #38bdf8 !important;
    overflow-wrap:anywhere;
    word-break:break-word;
    line-height:1.12;
    margin-bottom:0.35rem;
  }
  h1 { font-size: clamp(1.7rem, 3vw, 2.45rem) !important; }
  h2 { font-size: clamp(1.25rem, 2.2vw, 1.75rem) !important; }
  h3 { font-size: clamp(1.05rem, 1.8vw, 1.35rem) !important; }
  [data-testid="metric-container"] { background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:10px; min-width:0; }
  [data-testid="metric-container"] label,
  [data-testid="metric-container"] [data-testid="stMetricLabel"],
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    overflow-wrap:anywhere;
    word-break:break-word;
  }
  .stTabs [data-baseweb="tab-list"] {
    background:#0a1628;
    border-bottom:1px solid #1e293b;
    flex-wrap:wrap;
    gap:0.35rem;
    padding-bottom:0.35rem;
  }
  .stTabs [data-baseweb="tab"] {
    color:#475569;
    white-space:normal;
    height:auto;
    min-height:44px;
    padding:0.45rem 0.8rem;
    text-align:center;
    border-radius:8px 8px 0 0;
  }
  .stTabs [aria-selected="true"] { color:#38bdf8 !important; border-bottom:2px solid #38bdf8; }
  .stMarkdown, .stCaption, div[data-testid="stAlertContent"] {
    overflow-wrap:anywhere;
    word-break:break-word;
  }
  .stCaption { font-size:0.92rem; line-height:1.35; }
  .group-chip-row { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 14px; }
  .group-chip {
    display:inline-flex; align-items:center; gap:8px;
    background:#0f172a; border:1px solid #1e293b; border-radius:999px;
    padding:6px 10px; color:#cbd5e1; font-size:13px; line-height:1.2;
  }
  .group-chip b { color:#ffffff; }
  .stButton > button { background:#0f172a; border:1px solid #334155; color:#94a3b8; border-radius:6px; }
  .stButton > button:hover { border-color:#38bdf8; color:#38bdf8; }
  .warn-box  { background:#2a1500;border:1px solid #f9731655;border-radius:8px;padding:8px 12px;margin-bottom:6px;color:#f97316;font-size:13px; }
  .error-box { background:#2a0a0a;border:1px solid #ef444455;border-radius:8px;padding:8px 12px;margin-bottom:6px;color:#ef4444;font-size:13px; }
  .info-box  { background:#0c2231;border:1px solid #38bdf855;border-radius:8px;padding:8px 12px;margin-bottom:6px;color:#38bdf8;font-size:13px; }
  .ok-box    { background:#0a2010;border:1px solid #4ade8055;border-radius:8px;padding:8px 12px;margin-bottom:6px;color:#4ade80;font-size:13px; }
  .ai-box    { background:#130a2a;border:1px solid #a78bfa55;border-radius:8px;padding:10px 14px;margin-bottom:8px;color:#c4b5fd;font-size:13px; }
  @media (max-width: 900px) {
    .stTabs [data-baseweb="tab"] {
      min-height: 38px;
      padding: 0.35rem 0.55rem;
      font-size: 0.92rem;
    }
    .group-chip-row { gap:6px; }
    .group-chip { font-size:12px; padding:5px 8px; }
  }
  @media (max-width: 640px) {
    h1 { font-size: 1.45rem !important; }
    h2 { font-size: 1.12rem !important; }
    h3 { font-size: 1rem !important; }
    .stCaption { font-size:0.84rem; }
    .stTabs [data-baseweb="tab-list"] { gap:0.25rem; }
    .stTabs [data-baseweb="tab"] {
      min-height: 34px;
      padding: 0.25rem 0.45rem;
      font-size: 0.84rem;
    }
  }
</style>
""", unsafe_allow_html=True)

EMBEDDED_R_BACKEND_SCRIPT = r"""args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 10) {
  stop("Usage: Rscript rna_seq_backend.R counts.csv metadata.csv out_dir label_a label_b min_cpm min_samples padj fc top_n")
}

counts_path <- args[[1]]
metadata_path <- args[[2]]
out_dir <- args[[3]]
label_a <- args[[4]]
label_b <- args[[5]]
min_cpm <- as.numeric(args[[6]])
min_samples <- as.integer(args[[7]])
padj_thresh <- as.numeric(args[[8]])
fc_thresh <- as.numeric(args[[9]])
top_n <- as.integer(args[[10]])

suppressPackageStartupMessages({
  library(DESeq2)
  library(ggplot2)
  library(pheatmap)
  library(jsonlite)
  library(RColorBrewer)
})

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

counts_df <- read.csv(counts_path, row.names = 1, check.names = FALSE)
meta_df <- read.csv(metadata_path, stringsAsFactors = FALSE)
meta_df <- meta_df[meta_df$group != "Unassigned", , drop = FALSE]
if (!(label_a %in% meta_df$group)) {
  stop(sprintf("Reference group '%s' is not present in the selected samples.", label_a))
}

treatment_groups <- unique(meta_df$group[meta_df$group != label_a])
if (length(treatment_groups) < 1) {
  stop("Need at least one treatment group in addition to the reference group.")
}
if (!(label_b %in% treatment_groups)) {
  stop(sprintf("Selected treatment group '%s' is not present in the selected samples.", label_b))
}

ordered_groups <- c(label_a, label_b, setdiff(sort(treatment_groups), label_b))
meta_df$group <- factor(meta_df$group, levels = ordered_groups)
meta_df <- meta_df[order(meta_df$group), , drop = FALSE]
counts_df <- counts_df[, meta_df$sample, drop = FALSE]
count_matrix <- as.matrix(round(counts_df))

lib_sizes <- colSums(count_matrix)
cpm <- sweep(count_matrix, 2, pmax(lib_sizes, 1), "/") * 1e6
keep <- rowSums(cpm >= min_cpm) >= min_samples
filtered_counts <- count_matrix[keep, , drop = FALSE]
if (nrow(filtered_counts) < 2) {
  stop("Too few genes pass filtering for analysis.")
}
write.csv(filtered_counts, file.path(out_dir, "filtered_counts.csv"))

dds <- DESeqDataSetFromMatrix(
  countData = round(filtered_counts),
  colData = meta_df,
  design = ~ group
)
dds$group <- relevel(dds$group, ref = label_a)

run_deseq <- function(dds_obj) {
  tryCatch(
    {
      dds_res <- DESeq(dds_obj, quiet = TRUE)
      attr(dds_res, "fit_mode") <- "standard"
      dds_res
    },
    error = function(err) {
      err_msg <- conditionMessage(err)
      if (!grepl("standard curve fitting techniques will not work", err_msg, fixed = TRUE)) {
        stop(err)
      }

      message("DESeq2 standard dispersion fit failed; using gene-wise dispersion fallback.")
      dds_fallback <- estimateSizeFactors(dds_obj, quiet = TRUE)
      dds_fallback <- estimateDispersionsGeneEst(dds_fallback, quiet = TRUE)
      dispersions(dds_fallback) <- mcols(dds_fallback)$dispGeneEst
      dds_fallback <- nbinomWaldTest(dds_fallback, quiet = TRUE)
      attr(dds_fallback, "fit_mode") <- "gene-wise"
      dds_fallback
    }
  )
}

dds <- run_deseq(dds)

norm_counts <- counts(dds, normalized = TRUE)
transformed_obj <- tryCatch(
  rlog(dds, blind = FALSE),
  error = function(e) {
    tryCatch(
      vst(dds, blind = FALSE),
      error = function(vst_err) normTransform(dds)
    )
  }
)
transformed_mat <- assay(transformed_obj)
contrast_groups <- levels(dds$group)[levels(dds$group) != label_a]

shrink_type <- if (requireNamespace("ashr", quietly = TRUE)) "ashr" else "normal"

build_contrast_df <- function(target_group) {
  res_raw <- results(dds, contrast = c("group", target_group, label_a), alpha = padj_thresh)
  res_shrunk <- lfcShrink(
    dds,
    contrast = c("group", target_group, label_a),
    res = res_raw,
    type = shrink_type,
    quiet = TRUE
  )

  res_df <- as.data.frame(res_raw)
  res_df$gene <- rownames(res_df)
  res_df$log2FC_raw <- res_df$log2FoldChange
  res_df$log2FC <- as.data.frame(res_shrunk)$log2FoldChange
  res_df$baseMean <- ifelse(is.na(res_df$baseMean), 0, res_df$baseMean)
  res_df$pvalue <- ifelse(is.na(res_df$pvalue), 1, res_df$pvalue)
  res_df$padj <- ifelse(is.na(res_df$padj), 1, res_df$padj)
  res_df$negLog10P <- -log10(pmax(res_df$pvalue, 1e-300))
  res_df$negLog10Padj <- -log10(pmax(res_df$padj, 1e-300))
  res_df$muA <- rowMeans(norm_counts[, colData(dds)$group == label_a, drop = FALSE])
  res_df$muB <- rowMeans(norm_counts[, colData(dds)$group == target_group, drop = FALSE])
  res_df$contrast <- sprintf("%s vs %s", target_group, label_a)
  res_df$target_group <- target_group
  res_df <- res_df[, c(
    "gene", "muA", "muB", "baseMean", "log2FC_raw", "log2FC",
    "pvalue", "padj", "negLog10P", "negLog10Padj", "contrast", "target_group"
  )]
  names(res_df)[names(res_df) == "pvalue"] <- "pval"
  res_df[order(res_df$padj, res_df$pval), ]
}

contrast_tables <- lapply(contrast_groups, build_contrast_df)
names(contrast_tables) <- contrast_groups
res_df <- contrast_tables[[label_b]]
write.csv(res_df[, setdiff(names(res_df), c("contrast", "target_group"))], file.path(out_dir, "de_results.csv"), row.names = FALSE)
for (target_group in names(contrast_tables)) {
  out_name <- sprintf("DEG_%s_vs_%s.csv", gsub("[^A-Za-z0-9]+", "_", target_group), gsub("[^A-Za-z0-9]+", "_", label_a))
  write.csv(
    contrast_tables[[target_group]][, setdiff(names(contrast_tables[[target_group]]), c("contrast", "target_group"))],
    file.path(out_dir, out_name),
    row.names = FALSE
  )
}

palette_base <- c("#2166ac", "#b2182b", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02")
group_colors <- setNames(palette_base[seq_along(levels(dds$group))], levels(dds$group))
anno_col <- data.frame(Group = colData(dds)$group)
rownames(anno_col) <- colnames(dds)
anno_colors <- list(Group = group_colors)

plot_device <- function(name, width, height, code) {
  png(file.path(out_dir, paste0(name, ".png")), width = width, height = height, res = 300)
  code()
  dev.off()
  svg(file.path(out_dir, paste0(name, ".svg")), width = width / 100, height = height / 100)
  code()
  dev.off()
}

plot_device("qc_library_sizes", 2100, max(1200, 240 + 140 * ncol(count_matrix)), function() {
  lib_df <- data.frame(sample = colnames(count_matrix), reads = lib_sizes / 1e6, group = colData(dds)$group)
  gg <- ggplot(lib_df, aes(x = reorder(sample, reads), y = reads, fill = group)) +
    geom_col(width = 0.72, color = NA) +
    geom_hline(yintercept = median(lib_df$reads), linetype = "dashed", color = "#4b5563") +
    coord_flip() +
    scale_fill_manual(values = group_colors) +
    labs(title = "Library size by sample", x = NULL, y = "Library size (millions of reads)") +
    theme_bw(base_size = 13) +
    theme(panel.grid.major.y = element_blank(), legend.position = "top")
  print(gg)
})

plot_device("qc_dispersion", 1800, 1500, function() {
  plotDispEsts(dds, main = "Dispersion plot")
})

plot_device("qc_correlation", 2000, 1800, function() {
  corr_mat <- cor(transformed_mat, method = "spearman")
  pheatmap(
    corr_mat,
    color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(200),
    breaks = seq(0.8, 1.0, length.out = 201),
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    display_numbers = TRUE,
    fontsize_number = 9,
    annotation_col = anno_col,
    annotation_row = anno_col,
    annotation_colors = anno_colors,
    main = "Sample-to-sample Spearman correlation"
  )
})

plot_device("qc_sample_distance", 2000, 1800, function() {
  sample_dists <- as.matrix(dist(t(transformed_mat)))
  pheatmap(
    sample_dists,
    color = colorRampPalette(c("black", "white"))(200),
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    annotation_col = anno_col,
    annotation_row = anno_col,
    annotation_colors = anno_colors,
    main = sprintf("Sample distance matrix - %s vs %s", label_a, paste(contrast_groups, collapse = " vs "))
  )
})

plot_device("qc_pvalue_distribution", 1800, 1200, function() {
  pv_df <- data.frame(pval = res_df$pval)
  gg <- ggplot(pv_df, aes(x = pval)) +
    geom_histogram(bins = 20, fill = "#4c78a8", color = "white") +
    geom_hline(yintercept = nrow(pv_df) / 20, linetype = "dashed", color = "#4b5563") +
    labs(title = sprintf("P-value distribution (%s vs %s)", label_b, label_a), x = "P-value", y = "Gene count") +
    theme_bw(base_size = 13)
  print(gg)
})

pca <- prcomp(t(transformed_mat), scale. = FALSE)
pca_df <- data.frame(sample = rownames(pca$x), PC1 = pca$x[, 1], PC2 = pca$x[, 2], group = colData(dds)$group)
var_explained <- (pca$sdev^2) / sum(pca$sdev^2)
plot_device("pca", 1800, 1500, function() {
  gg <- ggplot(pca_df, aes(PC1, PC2, color = group, label = sample)) +
    geom_point(size = 3.5) +
    geom_text(vjust = -0.8, size = 3.6) +
    scale_color_manual(values = group_colors) +
    labs(
      title = "Principal component analysis",
      x = sprintf("PC1 (%.1f%% variance)", var_explained[1] * 100),
      y = sprintf("PC2 (%.1f%% variance)", var_explained[2] * 100)
    ) +
    theme_bw(base_size = 13) +
    theme(legend.position = "top")
  print(gg)
})

combined_df <- do.call(rbind, lapply(contrast_tables, function(df) {
  transform(
    df,
    regulation = ifelse(padj <= padj_thresh & log2FC_raw >= fc_thresh, "Up",
                 ifelse(padj <= padj_thresh & log2FC_raw <= -fc_thresh, "Down", "NS")),
    A = log2(baseMean + 0.5)
  )
}))
combined_df$contrast <- factor(combined_df$contrast, levels = sprintf("%s vs %s", contrast_groups, label_a))
combined_df$target_group <- factor(combined_df$target_group, levels = contrast_groups)
combined_df$regulation <- factor(combined_df$regulation, levels = c("NS", "Down", "Up"))
panel_count <- length(contrast_groups)
volcano_width <- max(2000, 1100 * panel_count)
contrast_colors <- setNames(palette_base[seq_along(contrast_groups)], contrast_groups)

plot_device("volcano", volcano_width, 1500, function() {
  gg <- ggplot(combined_df, aes(log2FC_raw, negLog10Padj)) +
    geom_point(
      data = subset(combined_df, regulation == "NS"),
      color = "#cbd5e1",
      alpha = 0.14,
      size = 0.9
    ) +
    geom_point(
      data = subset(combined_df, regulation != "NS"),
      aes(color = target_group, shape = regulation),
      alpha = 0.85,
      size = 1.9
    ) +
    geom_vline(xintercept = c(-fc_thresh, fc_thresh), linetype = "dotted", color = "#4b5563") +
    geom_hline(yintercept = -log10(padj_thresh), linetype = "dashed", color = "#4b5563") +
    scale_color_manual(values = contrast_colors, name = sprintf("Treatment vs %s", label_a)) +
    scale_shape_manual(values = c("Down" = 17, "Up" = 16), name = "Regulation") +
    labs(
      title = if (panel_count > 1) "Combined volcano plot" else "Volcano plot",
      subtitle = if (panel_count > 1) sprintf("Overlay of %s vs %s", paste(contrast_groups, collapse = ", "), label_a) else sprintf("%s vs %s", label_b, label_a),
      x = "log2 fold change (raw)",
      y = "-log10 adjusted p-value"
    ) +
    theme_bw(base_size = 13) +
    theme(legend.position = "bottom")
  print(gg)
})

plot_device("ma", volcano_width, 1500, function() {
  gg <- ggplot(combined_df, aes(A, log2FC_raw)) +
    geom_point(
      data = subset(combined_df, regulation == "NS"),
      color = "#cbd5e1",
      alpha = 0.14,
      size = 0.9
    ) +
    geom_point(
      data = subset(combined_df, regulation != "NS"),
      aes(color = target_group, shape = regulation),
      alpha = 0.82,
      size = 1.7
    ) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#4b5563") +
    geom_smooth(se = FALSE, color = "#111111", linewidth = 0.7, method = "loess", formula = y ~ x) +
    scale_color_manual(values = contrast_colors, name = sprintf("Treatment vs %s", label_a)) +
    scale_shape_manual(values = c("Down" = 17, "Up" = 16), name = "Regulation") +
    labs(
      title = if (panel_count > 1) "Combined MA plot" else "MA plot",
      subtitle = if (panel_count > 1) sprintf("Overlay of %s vs %s", paste(contrast_groups, collapse = ", "), label_a) else sprintf("%s vs %s", label_b, label_a),
      x = "A = log2(mean normalized count)",
      y = "M = log2 fold change"
    ) +
    theme_bw(base_size = 13) +
    theme(legend.position = "bottom")
  print(gg)
})

var_order <- order(apply(transformed_mat, 1, var), decreasing = TRUE)[seq_len(min(top_n, nrow(transformed_mat)))]
plot_device("expression_heatmap", 1800, 2200, function() {
  pheatmap(
    transformed_mat[var_order, , drop = FALSE],
    scale = "row",
    annotation_col = anno_col,
    annotation_colors = anno_colors,
    show_rownames = TRUE,
    main = sprintf("Top %d variable genes", length(var_order)),
    color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(200)
  )
})

all_genes <- rownames(transformed_mat)
contrast_gene_frames <- lapply(names(contrast_tables), function(target_group) {
  df <- contrast_tables[[target_group]]
  aligned <- df[match(all_genes, df$gene), c("log2FC", "padj")]
  colnames(aligned) <- c(
    sprintf("lfc_%s", target_group),
    sprintf("padj_%s", target_group)
  )
  aligned
})
combined_contrasts <- do.call(cbind, contrast_gene_frames)
rownames(combined_contrasts) <- all_genes
combined_contrasts <- combined_contrasts[complete.cases(combined_contrasts), , drop = FALSE]

core_fc_thresh <- max(1.0, fc_thresh)
strong_fc_thresh <- max(1.5, fc_thresh)
lfc_cols <- grep("^lfc_", colnames(combined_contrasts), value = TRUE)
padj_cols <- grep("^padj_", colnames(combined_contrasts), value = TRUE)

core_mask <- apply(combined_contrasts[, padj_cols, drop = FALSE] < padj_thresh, 1, all) &
  apply(abs(combined_contrasts[, lfc_cols, drop = FALSE]) >= core_fc_thresh, 1, all)
core <- combined_contrasts[core_mask, , drop = FALSE]
if (nrow(core) > 0) {
  core <- core[order(rowMeans(abs(core[, lfc_cols, drop = FALSE]), na.rm = TRUE), decreasing = TRUE), , drop = FALSE]
}

strong_mask <- apply(combined_contrasts[, padj_cols, drop = FALSE] < padj_thresh, 1, any) &
  apply(abs(combined_contrasts[, lfc_cols, drop = FALSE]) >= strong_fc_thresh, 1, any)
strong <- combined_contrasts[strong_mask, , drop = FALSE]
if (nrow(strong) > 0) {
  strong <- strong[order(apply(abs(strong[, lfc_cols, drop = FALSE]), 1, max, na.rm = TRUE), decreasing = TRUE), , drop = FALSE]
}

hybrid_genes <- unique(c(rownames(core), rownames(strong)))
if (length(hybrid_genes) < top_n) {
  fallback <- combined_contrasts[order(apply(abs(combined_contrasts[, lfc_cols, drop = FALSE]), 1, max, na.rm = TRUE), decreasing = TRUE), , drop = FALSE]
  for (gene_id in rownames(fallback)) {
    if (!(gene_id %in% hybrid_genes)) {
      hybrid_genes <- c(hybrid_genes, gene_id)
    }
    if (length(hybrid_genes) >= top_n) {
      break
    }
  }
}
hybrid_genes <- hybrid_genes[seq_len(min(length(hybrid_genes), top_n))]

if (length(hybrid_genes) >= 2) {
  heatmap_title <- sprintf(
    "Top %d Hybrid DEGs (%s vs %s)",
    length(hybrid_genes),
    paste(contrast_groups, collapse = " + "),
    label_a
  )
  plot_device("deg_heatmap", 1800, 2200, function() {
    pheatmap(
      transformed_mat[hybrid_genes, , drop = FALSE],
      scale = "row",
      annotation_col = anno_col,
      annotation_colors = anno_colors,
      show_rownames = TRUE,
      cluster_rows = TRUE,
      cluster_cols = FALSE,
      fontsize_row = 9,
      main = heatmap_title,
      color = colorRampPalette(c("steelblue", "white", "firebrick"))(100)
    )
  })
  write.csv(
    data.frame(gene = hybrid_genes),
    file.path(out_dir, "hybrid_deg_genes.csv"),
    row.names = FALSE
  )
} else {
  file.create(file.path(out_dir, "deg_heatmap.png"))
  file.create(file.path(out_dir, "deg_heatmap.svg"))
}

deg_summary <- lapply(names(contrast_tables), function(target_group) {
  df <- contrast_tables[[target_group]]
  sig_df <- df[df$padj <= padj_thresh & abs(df$log2FC) >= fc_thresh, , drop = FALSE]
  list(
    contrast = sprintf("%s vs %s", target_group, label_a),
    total_deg = nrow(sig_df),
    up_regulated = sum(sig_df$log2FC > 0, na.rm = TRUE),
    down_regulated = sum(sig_df$log2FC < 0, na.rm = TRUE)
  )
})
names(deg_summary) <- names(contrast_tables)

manifest <- list(
  backend = "R",
  backend_label = "DESeq2 + ggplot2/pheatmap",
  reference_group = label_a,
  selected_contrast = label_b,
  contrast_groups = contrast_groups,
  multi_contrast = length(contrast_groups) > 1,
  deg_summary = unname(deg_summary),
  plots = list(
    qc_library_sizes = list(png = file.path(out_dir, "qc_library_sizes.png"), svg = file.path(out_dir, "qc_library_sizes.svg")),
    qc_dispersion = list(png = file.path(out_dir, "qc_dispersion.png"), svg = file.path(out_dir, "qc_dispersion.svg")),
    qc_correlation = list(png = file.path(out_dir, "qc_correlation.png"), svg = file.path(out_dir, "qc_correlation.svg")),
    qc_sample_distance = list(png = file.path(out_dir, "qc_sample_distance.png"), svg = file.path(out_dir, "qc_sample_distance.svg")),
    qc_pvalue_distribution = list(png = file.path(out_dir, "qc_pvalue_distribution.png"), svg = file.path(out_dir, "qc_pvalue_distribution.svg")),
    pca = list(png = file.path(out_dir, "pca.png"), svg = file.path(out_dir, "pca.svg")),
    volcano = list(png = file.path(out_dir, "volcano.png"), svg = file.path(out_dir, "volcano.svg")),
    ma = list(png = file.path(out_dir, "ma.png"), svg = file.path(out_dir, "ma.svg")),
    expression_heatmap = list(png = file.path(out_dir, "expression_heatmap.png"), svg = file.path(out_dir, "expression_heatmap.svg")),
    deg_heatmap = list(png = file.path(out_dir, "deg_heatmap.png"), svg = file.path(out_dir, "deg_heatmap.svg"))
  ),
  result_files = lapply(names(contrast_tables), function(target_group) {
    out_name <- sprintf("DEG_%s_vs_%s.csv", gsub("[^A-Za-z0-9]+", "_", target_group), gsub("[^A-Za-z0-9]+", "_", label_a))
    list(
      contrast = sprintf("%s vs %s", target_group, label_a),
      csv = file.path(out_dir, out_name)
    )
  })
)
write(toJSON(manifest, auto_unbox = TRUE, pretty = TRUE), file.path(out_dir, "manifest.json"))
"""


DEFAULT_RSCRIPT_CANDIDATES = (
    Path("/home/ananya/miniconda3/envs/rnaseq/bin/Rscript"),
    Path("/home/ananya/miniconda3/pkgs/r-base-4.5.3-h1a7235e_0/bin/Rscript"),
)


def find_rscript():
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


def r_backend_status():
    rscript = find_rscript()
    return {
        "available": bool(rscript),
        "rscript": rscript,
        "script": "embedded in adaptive_streamlit_app.py",
    }


def write_embedded_r_backend(tmpdir):
    script_path = Path(tmpdir) / "rna_seq_backend_embedded.R"
    script_path.write_text(EMBEDDED_R_BACKEND_SCRIPT, encoding="utf-8")
    return script_path


def run_r_backend(count_df, group_map, label_a, label_b, min_cpm, min_samples, padj_thresh, fc_thresh, top_n):
    status = r_backend_status()
    if not status["available"]:
        raise RuntimeError("R backend unavailable: install R/Rscript with DESeq2, ggplot2, pheatmap, jsonlite, and RColorBrewer.")

    tmpdir = Path(tempfile.mkdtemp(prefix="transcriptomicviz_r_", dir="/tmp"))
    counts_path = tmpdir / "counts.csv"
    metadata_path = tmpdir / "metadata.csv"
    out_dir = tmpdir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = write_embedded_r_backend(tmpdir)

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
        str(script_path),
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

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_uploaded(f):
    text = f.read().decode("utf-8", errors="replace")
    sn = re.sub(r"\.(txt|tsv|csv)$","",f.name,flags=re.IGNORECASE)
    sn = re.sub(r"_counts?$","",sn,flags=re.IGNORECASE)
    fmt = detect_format(text)
    return (parse_featurecounts(text,sn),"featureCounts") if fmt=="featurecounts" else (parse_htseq(text,sn),"HTSeq")

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=450, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0); return buf.read()

def dark_fig(w=9,h=6):
    fig,ax=plt.subplots(figsize=(w,h))
    fig.patch.set_facecolor("#020817"); ax.set_facecolor("#0f172a")
    for sp in ax.spines.values(): sp.set_edgecolor("#334155")
    ax.tick_params(colors="#64748b")
    ax.xaxis.label.set_color("#64748b"); ax.yaxis.label.set_color("#64748b")
    ax.title.set_color("#e2e8f0")
    return fig,ax


PUB_COLORS = {"up": "#b2182b", "down": "#2166ac", "ns": "#b0b7c3"}


def publication_fig(w=6.8, h=5.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.tick_params(colors="#222222", labelsize=10)
    ax.xaxis.label.set_color("#111111")
    ax.yaxis.label.set_color("#111111")
    return fig, ax


def publication_bytes(fig, fmt="png"):
    buf = io.BytesIO()
    kwargs = {"format": fmt, "bbox_inches": "tight", "facecolor": fig.get_facecolor()}
    if fmt == "png":
        kwargs["dpi"] = 450
    fig.savefig(buf, **kwargs)
    buf.seek(0)
    return buf.read()


def figure_downloads(fig, stem, key):
    c1, c2 = st.columns(2)
    c1.download_button("Download PNG", publication_bytes(fig, "png"), f"{stem}.png", "image/png", key=f"{key}_png")
    c2.download_button("Download SVG", publication_bytes(fig, "svg"), f"{stem}.svg", "image/svg+xml", key=f"{key}_svg")


def backend_plot_downloads(plot_info, key):
    p1, p2 = st.columns(2)
    png_path = plot_info.get("png")
    svg_path = plot_info.get("svg")
    if png_path and Path(png_path).exists():
        p1.download_button("Download PNG", Path(png_path).read_bytes(), Path(png_path).name, "image/png", key=f"{key}_png_file")
    if svg_path and Path(svg_path).exists():
        p2.download_button("Download SVG", Path(svg_path).read_bytes(), Path(svg_path).name, "image/svg+xml", key=f"{key}_svg_file")


def render_backend_plot(manifest, plot_key, image_caption=None, dl_key=None):
    if not manifest:
        return False
    plot_info = manifest.get("plots", {}).get(plot_key)
    if not plot_info:
        return False
    png_path = plot_info.get("png")
    if not png_path or not Path(png_path).exists() or Path(png_path).stat().st_size == 0:
        return False
    st.image(png_path, caption=image_caption, use_container_width=True)
    backend_plot_downloads(plot_info, dl_key or plot_key)
    return True


def normalize_backend_summary(summary_source):
    if isinstance(summary_source, dict):
        items = []
        for key, value in summary_source.items():
            if isinstance(value, dict):
                row = value.copy()
                row.setdefault("contrast", key)
                items.append(row)
            else:
                items.append({"contrast": str(value if value is not None else key)})
        return items
    if isinstance(summary_source, list):
        items = []
        for item in summary_source:
            if isinstance(item, dict):
                items.append(item)
            else:
                items.append({"contrast": str(item)})
        return items
    return []


CONTROL_GROUP_PATTERN = re.compile(r"(?i)^(control|ctrl|untreated|vehicle|mock|baseline|wt|wildtype|wild_type|sensitive)$")


def order_group_labels(labels, group_map):
    labels = [lbl for lbl in labels if lbl]
    if not labels:
        return ["Control", "Treatment"]

    control_like = [lbl for lbl in labels if CONTROL_GROUP_PATTERN.search(lbl)]
    if control_like:
        reference = control_like[0]
    else:
        counts = {lbl: sum(1 for grp in group_map.values() if grp == lbl) for lbl in labels}
        reference = sorted(labels, key=lambda lbl: (-counts.get(lbl, 0), lbl.lower()))[0]

    remainder = [lbl for lbl in labels if lbl != reference]
    return [reference] + sorted(remainder, key=str.lower)


def prettify_group_label(label):
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


def infer_name_group(sample_name):
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
        return prettify_group_label(informative_stem), "HIGH"
    return None, "LOW"


def cluster_sample_groups(matrix, samples):
    if len(samples) < 4:
        return {sample: "?" for sample in samples}

    lib = matrix.sum(axis=0)
    lib = np.where(lib == 0, 1, lib)
    cpm = (matrix / lib[np.newaxis, :]) * 1e6
    log_mat = np.log2(cpm + 0.5).T

    try:
        gene_vars = log_mat.var(axis=0)
        top_idx = np.argsort(gene_vars)[::-1][:500]
        sub = log_mat[:, top_idx]
        sub = (sub - sub.mean(axis=1, keepdims=True)) / (sub.std(axis=1, keepdims=True) + 1e-10)
        labels = fcluster(linkage(sub, method="ward", metric="euclidean"), t=2, criterion="maxclust")
        unique_labels = sorted(set(labels))
        label_names = ["Group A", "Group B"] + [f"Group {chr(67+i)}" for i in range(len(unique_labels) - 2)]
        return {sample: label_names[labels[idx] - 1] for idx, sample in enumerate(samples)}
    except Exception:
        return {sample: "Group A" for sample in samples}


def infer_groups_from_upload(matrix, genes, samples):
    name_groups = {}
    name_confidence = {}
    for sample in samples:
        inferred_group, confidence = infer_name_group(sample)
        name_groups[sample] = inferred_group
        name_confidence[sample] = confidence

    cluster_groups = cluster_sample_groups(matrix, samples)
    inferred = {}
    for sample in samples:
        if name_confidence.get(sample) in ("HIGH", "MEDIUM") and name_groups.get(sample):
            inferred[sample] = {"group": name_groups[sample], "method": "name pattern", "confidence": name_confidence[sample]}
        else:
            inferred[sample] = {"group": cluster_groups.get(sample, "Group A"), "method": "expression clustering", "confidence": "MEDIUM"}

    group_map = {sample: info["group"] for sample, info in inferred.items()}
    labels = order_group_labels(sorted(set(group_map.values())), group_map)

    if len(labels) == 1:
        fallback = "Treatment" if labels[0] != "Treatment" else "Group 2"
        labels.append(fallback)

    reference = labels[0]
    treatment = next((label for label in labels if label != reference), labels[-1])
    return labels, group_map, reference, treatment


def remap_group_assignments(group_map, rename_map, valid_labels):
    valid_labels = [label for label in valid_labels if label]
    fallback_label = valid_labels[0] if valid_labels else "Group 1"
    remapped = {}
    for sample, group in group_map.items():
        updated_group = rename_map.get(group, group)
        if updated_group not in valid_labels:
            updated_group = fallback_label
        remapped[sample] = updated_group
    return remapped


def add_group_ellipse(ax, x_vals, y_vals, color):
    if len(x_vals) < 3:
        return
    cov = np.cov(x_vals, y_vals)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if np.any(eigvals <= 0):
        return
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2.0 * 1.8 * np.sqrt(eigvals)
    ax.add_patch(Ellipse(
        xy=(np.mean(x_vals), np.mean(y_vals)),
        width=width, height=height, angle=angle,
        facecolor=color, edgecolor=color, alpha=0.12, lw=1.2, zorder=1,
    ))


def smooth_xy(x_vals, y_vals, bins=120):
    if len(x_vals) < 20:
        return None, None
    order = np.argsort(x_vals)
    x_sorted = np.asarray(x_vals)[order]
    y_sorted = np.asarray(y_vals)[order]
    window = max(9, len(x_sorted) // 20)
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y_sorted, kernel, mode="same")
    stride = max(1, len(x_sorted) // bins)
    return x_sorted[::stride], y_smooth[::stride]


def publication_heatmap(data, row_labels, col_labels, group_labels, title):
    n_rows, n_cols = data.shape
    fig_h = max(4.6, n_rows * max(0.16, min(0.38, 10 / max(n_rows, 1))) + 1.8)
    fig_w = max(6.5, n_cols * 0.55 + 2.2)
    fig, axes = plt.subplots(
        2, 1, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [0.22, 1], "hspace": 0.02},
    )
    fig.patch.set_facecolor("white")
    ax_top, ax = axes
    group_names = list(dict.fromkeys(group_labels))
    group_colors = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(group_names)}
    top_strip = np.array([[plt.matplotlib.colors.to_rgb(group_colors[g]) for g in group_labels]])
    ax_top.imshow(top_strip, aspect="auto")
    ax_top.set_xticks(range(n_cols))
    ax_top.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=9, color="#111111")
    ax_top.set_yticks([])
    ax_top.tick_params(length=0)
    for sp in ax_top.spines.values():
        sp.set_visible(False)

    im = ax.imshow(
        data, aspect="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3), interpolation="nearest",
    )
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=max(6, min(9, 150 // max(n_rows, 1))), color="#111111")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.ax.tick_params(labelsize=8, colors="#111111")
    cbar.ax.set_ylabel("Row z-score", rotation=270, labelpad=14, color="#111111", fontsize=9)
    handles = [Line2D([0], [0], color=group_colors[g], lw=6, label=g) for g in group_names]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    ax.set_title(title, fontsize=13, color="#111111", pad=10, loc="left")
    fig.tight_layout()
    return fig

# ── Session state ─────────────────────────────────────────────────────────────
for key,val in [
    ("matrix",None),("genes",[]),("samples",[]),("count_df",None),
    ("group_map",{}),("group_labels",["Control","Treatment"]),
    ("de_results",None),("filt_mat",None),("filt_genes",[]),
    ("filt_idx",[]),("run_meta",None),
    ("compare_a","Control"),("compare_b","Treatment"),
    ("compare_all_treatments", False),
    ("compare_targets", []),
    ("selected_samples", None), ("backend_manifest", None),
    ("upload_signature", None),
]:
    if key not in st.session_state: st.session_state[key]=val

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 TranscriptomicViz")
    st.markdown("**Adaptive Analysis**")
    st.caption("Infer sample groups and run transcriptomic analysis.")
    st.divider()

    st.markdown("### 📂 Upload Count Files")
    uploaded = st.file_uploader(
        "HTSeq or featureCounts (auto-detected)",
        type=["txt","tsv","csv"], accept_multiple_files=True
    )

    if uploaded:
        upload_signature = tuple(sorted((f.name, getattr(f, "size", None)) for f in uploaded))
        dfs,fmt_log = [],[]
        for f in uploaded:
            try:
                df,fmt = parse_uploaded(f)
                if not df.empty:
                    dfs.append(df); fmt_log.append(f"✓ {f.name} [{fmt}]")
            except Exception as e:
                st.error(f"Error: {f.name}: {e}")
        if dfs:
            merged = merge_files(dfs)
            # Deduplicate column names if same sample appears twice
            merged = merged.loc[:, ~merged.columns.duplicated()]
            if st.session_state.upload_signature != upload_signature:
                inferred_labels, inferred_map, inferred_ref, inferred_treatment = infer_groups_from_upload(
                    merged.values.astype(float),
                    list(merged.index),
                    list(merged.columns),
                )
                st.session_state.group_labels = inferred_labels
                st.session_state.group_map = inferred_map
                st.session_state.compare_a = inferred_ref
                st.session_state.compare_b = inferred_treatment
                st.session_state.compare_all_treatments = False
                st.session_state.compare_targets = [inferred_treatment]
                st.session_state.de_results = None
                st.session_state.selected_samples = None  # reset on new upload
                st.session_state.backend_manifest = None
                st.session_state.run_meta = None
                st.session_state.upload_signature = upload_signature
            st.session_state.count_df = merged
            st.session_state.samples  = list(merged.columns)
            st.session_state.genes    = list(merged.index)
            st.session_state.matrix   = merged.values.astype(float)
            with st.expander(f"Loaded {len(dfs)} file(s)"):
                for l in fmt_log: st.caption(l)

    if st.session_state.samples:
        st.divider()

        # ── Group management ─────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🏷️ Group Labels")
        labels = list(st.session_state.group_labels)
        original_labels = list(labels)
        new_labels = []
        for i, lbl in enumerate(labels):
            c1,c2 = st.columns([4,1])
            edited = c1.text_input(f"Group {i+1}", value=lbl, key=f"lbl_edit_{i}")
            new_labels.append(edited)
            if c2.button("✕", key=f"del_grp_{i}") and len(labels)>2:
                labels.pop(i); st.session_state.group_labels=labels; st.rerun()
        cleaned_labels = []
        seen_labels = set()
        for i, lbl in enumerate(new_labels):
            candidate = (lbl or "").strip() or f"Group {i+1}"
            if candidate in seen_labels:
                suffix = 2
                deduped = f"{candidate} {suffix}"
                while deduped in seen_labels:
                    suffix += 1
                    deduped = f"{candidate} {suffix}"
                candidate = deduped
            seen_labels.add(candidate)
            cleaned_labels.append(candidate)

        rename_map = {
            old: cleaned_labels[i]
            for i, old in enumerate(original_labels[:len(cleaned_labels)])
        }
        st.session_state.group_map = remap_group_assignments(
            st.session_state.group_map,
            rename_map,
            cleaned_labels,
        )
        st.session_state.group_labels = cleaned_labels
        labels = cleaned_labels
        st.session_state.compare_a = rename_map.get(st.session_state.compare_a, st.session_state.compare_a)
        st.session_state.compare_b = rename_map.get(st.session_state.compare_b, st.session_state.compare_b)
        st.session_state.compare_targets = [
            rename_map.get(target, target)
            for target in st.session_state.compare_targets
            if rename_map.get(target, target) in cleaned_labels
        ]
        if st.session_state.compare_b == st.session_state.compare_a and len(labels) > 1:
            st.session_state.compare_b = next((lbl for lbl in labels if lbl != st.session_state.compare_a), labels[0])

        if st.button("＋ Add Group", use_container_width=True):
            labels.append(f"Group {len(labels)+1}")
            st.session_state.group_labels=labels; st.rerun()

        # ── Sample Selection ─────────────────────────────────────────────
        st.markdown("### ✅ Select Samples")
        st.caption("Uncheck samples to exclude them from analysis.")

        all_samples_full = st.session_state.samples
        if st.session_state.selected_samples is None:
            st.session_state.selected_samples = list(all_samples_full)

        # Keep/remove buttons
        sc1, sc2 = st.columns(2)
        if sc1.button("Select All", use_container_width=True, key="sel_all"):
            st.session_state.selected_samples = list(all_samples_full)
            st.rerun()
        if sc2.button("Clear All", use_container_width=True, key="sel_none"):
            st.session_state.selected_samples = []
            st.rerun()

        new_selected = []
        for si, s in enumerate(all_samples_full):
            checked = s in st.session_state.selected_samples
            if st.checkbox(s, value=checked, key=f"sel_{si}"):
                new_selected.append(s)
        st.session_state.selected_samples = new_selected

        # Use only selected samples from here on
        active_samples = st.session_state.selected_samples
        if not active_samples:
            st.warning("No samples selected. Select at least 2.")

        st.divider()
        st.markdown("### 🔀 Assign Groups")
        st.caption(f"Assigning {len(active_samples)} selected samples")
        group_map = {}
        for si, s in enumerate(active_samples):
            prev = st.session_state.group_map.get(s, labels[0])
            if prev not in labels: prev = labels[0]
            choice = st.selectbox(s, labels, index=labels.index(prev), key=f"sample_select_{si}")
            group_map[s] = choice
        st.session_state.group_map = group_map

        st.divider()
        st.markdown("### ⚖️ Compare Groups")
        labels = order_group_labels(labels, st.session_state.group_map)
        st.session_state.group_labels = labels
        if st.session_state.compare_a not in labels:
            st.session_state.compare_a = labels[0]
        c1,c2 = st.columns(2)
        compare_a = c1.selectbox("Reference (A)", labels,
            index=labels.index(st.session_state.compare_a) if st.session_state.compare_a in labels else 0,
            key="ca")
        rem = [l for l in labels if l != compare_a]
        if not rem:
            c2.caption("Treatment (B)")
            c2.info("Assign samples to a second group to enable comparison.")
            compare_b = compare_a
            st.session_state.compare_all_treatments = False
        else:
            current_targets = [target for target in st.session_state.compare_targets if target in rem]
            if not current_targets:
                current_targets = [st.session_state.compare_b] if st.session_state.compare_b in rem else [rem[0]]
            compare_targets = c2.multiselect(
                "Treatments (B)",
                rem,
                default=current_targets,
                help="Select one or more treatment groups from the uploaded file set.",
                key="cb_multi",
            )
            compare_all = c2.checkbox(
                "Select all treatments",
                value=st.session_state.compare_all_treatments or len(compare_targets) == len(rem),
                key="compare_all_treatments_toggle",
                help="Use every non-reference group in the uploaded file set against the selected reference.",
            )
            if compare_all:
                compare_targets = list(rem)
            elif not compare_targets:
                compare_targets = [rem[0]]
            st.session_state.compare_all_treatments = compare_all
            st.session_state.compare_targets = compare_targets
            compare_b = compare_targets[0]
        st.session_state.compare_a = compare_a
        st.session_state.compare_b = compare_b
        label_a, label_b = compare_a, compare_b

        files_a = [sample for sample in active_samples if st.session_state.group_map.get(sample) == label_a]
        files_b = [sample for sample in active_samples if st.session_state.group_map.get(sample) == label_b]
        c1.caption("Files: " + (", ".join(files_a) if files_a else "None"))
        if rem and st.session_state.compare_targets:
            all_treatment_files = [
                sample for sample in active_samples
                if st.session_state.group_map.get(sample) in st.session_state.compare_targets
            ]
            c2.caption("Files: " + (", ".join(all_treatment_files) if all_treatment_files else "None"))

        # ── Organism Preset ──────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🧫 Organism Preset")
        st.caption("Auto-fills all thresholds. You can still adjust manually.")

        PRESETS = {
            "— select —": None,
            "🧑 Human (Homo sapiens)":           {"min_cpm":1.0,"padj":0.05,"fc":1.0,"top_n":50,"note":"~20,000 genes. Standard thresholds."},
            "🐭 Mouse (Mus musculus)":            {"min_cpm":1.0,"padj":0.05,"fc":1.0,"top_n":50,"note":"Same as human — similar transcriptome size."},
            "🐟 Zebrafish (Danio rerio)":         {"min_cpm":1.0,"padj":0.05,"fc":1.0,"top_n":50,"note":"~26,000 genes. Standard thresholds work well."},
            "🪰 Drosophila melanogaster":         {"min_cpm":1.0,"padj":0.05,"fc":1.0,"top_n":40,"note":"~14,000 genes. Slightly fewer top-N needed."},
            "🪱 C. elegans":                      {"min_cpm":1.0,"padj":0.05,"fc":1.0,"top_n":40,"note":"~20,000 genes. Standard metazoan thresholds."},
            "🌿 Arabidopsis thaliana":            {"min_cpm":0.5,"padj":0.05,"fc":1.0,"top_n":50,"note":"~27,000 genes. Lower CPM — many lowly expressed genes."},
            "🌾 Rice (Oryza sativa)":             {"min_cpm":0.5,"padj":0.05,"fc":1.0,"top_n":50,"note":"~40,000 genes. Large genome — lower CPM avoids over-filtering."},
            "🍺 Yeast (S. cerevisiae)":           {"min_cpm":2.0,"padj":0.05,"fc":1.0,"top_n":30,"note":"~6,000 genes. Higher CPM — compact genome, high expression."},
            "🍞 S. pombe (fission yeast)":        {"min_cpm":2.0,"padj":0.05,"fc":1.0,"top_n":30,"note":"~5,100 genes. Similar to S. cerevisiae."},
            "🦠 E. coli (bacterium)":             {"min_cpm":5.0,"padj":0.05,"fc":1.5,"top_n":25,"note":"~4,200 genes. Higher CPM & FC for bacterial RNA-seq."},
            "🦠 Acinetobacter baumannii":         {"min_cpm":5.0,"padj":0.05,"fc":1.5,"top_n":25,"note":"~3,900 genes. Nosocomial Gram-negative bacterium; stricter bacterial thresholds reduce low-count noise."},
            "🦠 Enterobacter cloacae":            {"min_cpm":3.0,"padj":0.05,"fc":1.5,"top_n":30,"note":"~4,500 genes. Gram-negative Enterobacteriaceae. CPM≥3, FC≥1.5."},
            "🦠 Enterobacter hormaechei":         {"min_cpm":3.0,"padj":0.05,"fc":1.5,"top_n":30,"note":"~5,000 genes. Clinical isolate — expect hypothetical protein IDs."},
            "🦠 Enterobacter aerogenes":          {"min_cpm":3.0,"padj":0.05,"fc":1.5,"top_n":30,"note":"~5,200 genes. Reclassified to Klebsiella in some databases."},
            "🧫 Bacillus subtilis":               {"min_cpm":5.0,"padj":0.05,"fc":1.5,"top_n":25,"note":"~4,100 genes. Same logic as E. coli."},
            "🦠 Pseudomonas aeruginosa":          {"min_cpm":5.0,"padj":0.05,"fc":1.5,"top_n":25,"note":"~5,500 genes. Larger bacterial genome, same high-expression logic."},
            "🧫 Mycobacterium tuberculosis":      {"min_cpm":5.0,"padj":0.05,"fc":1.5,"top_n":20,"note":"~4,000 genes. Slow-growing — FC≥1.5 reduces false hits."},
            "🍄 Aspergillus / Neurospora":        {"min_cpm":1.0,"padj":0.05,"fc":1.0,"top_n":35,"note":"~10,000 genes. Between yeast and metazoans."},
            "🐀 Rat (Rattus norvegicus)":         {"min_cpm":1.0,"padj":0.05,"fc":1.0,"top_n":50,"note":"~21,000 genes. Identical to mouse settings."},
            "🐄 Cow / Pig / Sheep (livestock)":  {"min_cpm":1.0,"padj":0.05,"fc":1.0,"top_n":50,"note":"~20–25,000 genes. Standard thresholds are safe."},
            "🐝 Honey bee (Apis mellifera)":      {"min_cpm":1.0,"padj":0.05,"fc":1.0,"top_n":35,"note":"~10,000 genes. Insect-size genome."},
            "🦟 Mosquito (Anopheles / Aedes)":   {"min_cpm":1.0,"padj":0.05,"fc":1.0,"top_n":35,"note":"~13,000 genes. Similar to Drosophila."},
            "🌊 Chlamydomonas (green alga)":     {"min_cpm":0.5,"padj":0.05,"fc":1.0,"top_n":30,"note":"~17,000 genes. Lower CPM preserves more genes."},
            "⚙️ Custom / Other":                 {"min_cpm":1.0,"padj":0.05,"fc":1.0,"top_n":50,"note":"General safe defaults. Adjust based on your organism."},
        }

        preset_choice = st.selectbox(
            "Select organism",
            list(PRESETS.keys()),
            index=0,
            key="organism_preset_adaptive"
        )
        preset = PRESETS[preset_choice]

        if preset:
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:7px;' +
                f'padding:8px 10px;margin-bottom:6px;font-size:11px;color:#64748b;">' +
                f'💡 {preset["note"]}</div>',
                unsafe_allow_html=True
            )
            # Apply preset to session state so sliders pick it up
            st.session_state["auto_min_cpm"] = preset["min_cpm"]
            st.session_state["auto_padj"]    = preset["padj"]
            st.session_state["auto_fc"]      = preset["fc"]
            st.session_state["auto_top_n"]   = preset["top_n"]

        # ── Thresholds (auto-filled or manual) ───────────────────────────────
        st.divider()
        st.markdown("### ⚙️ Parameters")
        auto_cpm   = st.session_state.get("auto_min_cpm", 1.0)
        auto_padj  = st.session_state.get("auto_padj", 0.05)
        auto_fc    = st.session_state.get("auto_fc", 1.0)
        auto_top_n = st.session_state.get("auto_top_n", 50)

        min_cpm     = st.slider("Min CPM", 0.1, 10.0, float(auto_cpm), 0.1)
        min_samples = st.slider("In ≥ N samples", 1, max(len(st.session_state.samples),2), 2, 1)
        padj_thresh = st.slider("padj ≤", 0.001, 0.2, float(auto_padj), 0.001, format="%.3f")
        fc_thresh   = st.slider("|log₂FC| ≥", 0.0, 4.0, float(auto_fc), 0.1)
        top_n       = st.slider("Top N genes (heatmap)", 10, 200, int(auto_top_n), 5)

        st.divider()
        run = st.button("▶ Run DE Analysis", use_container_width=True, type="primary")
    else:
        label_a, label_b = "Control", "Treatment"
        padj_thresh, fc_thresh, top_n = 0.05, 1.0, 50
        min_cpm, min_samples = 1.0, 2
        run = False

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 TranscriptomicViz Adaptive")
st.caption("Infer sample groups and run RNA-seq analysis.")

if st.session_state.count_df is None:
    st.markdown("""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;
                padding:28px 32px;max-width:620px;margin:40px auto;">
      <div style="color:#a78bfa;font-weight:700;font-size:15px;margin-bottom:14px;">
        🧠 TranscriptomicViz Adaptive
      </div>
      <div style="color:#64748b;font-size:13px;line-height:2.2;">
        ✓ Upload count files in the sidebar<br>
        ✓ Review or edit inferred sample groups<br>
        ✓ Set analysis parameters<br>
        ✓ Run differential expression and explore the plots
      </div>
      <div style="margin-top:14px;color:#475569;font-size:12px;">
        👈 Upload your count files in the sidebar to begin.
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

matrix_full = st.session_state.matrix
genes       = st.session_state.genes
samples_all = st.session_state.samples
gmap        = st.session_state.group_map
all_labels  = st.session_state.group_labels

# Filter to only selected samples
selected = st.session_state.get("selected_samples", None)
if selected is None:
    selected = samples_all
active_idx = [samples_all.index(s) for s in selected if s in samples_all]
samples = [samples_all[i] for i in active_idx]
matrix  = matrix_full[:, active_idx] if active_idx else matrix_full[:, :0]
gmap    = {s: gmap[s] for s in samples if s in gmap}

group_a_idx = [samples.index(s) for s,g in gmap.items() if g==label_a and s in samples]
group_b_idx = [samples.index(s) for s,g in gmap.items() if g==label_b and s in samples]

# Metrics row
mc1, mc2 = st.columns(2)
mc1.metric("Genes", f"{len(genes):,}")
mc2.metric("Samples", len(samples))

group_chip_html = "".join(
    f'<span class="group-chip"><span style="color:{PALETTE[i % len(PALETTE)]};font-size:14px;">■</span>'
    f'<span>{lbl}: <b>{sum(1 for g in gmap.values() if g == lbl)}</b></span></span>'
    for i, lbl in enumerate(all_labels)
)
st.markdown(f'<div class="group-chip-row">{group_chip_html}</div>', unsafe_allow_html=True)

de_results = st.session_state.de_results
r_status = r_backend_status()
backend_manifest = st.session_state.get("backend_manifest")
multi_contrast_mode = bool(backend_manifest and backend_manifest.get("multi_contrast"))

# Note if results are from a different comparison (don't invalidate — just warn)
_meta = st.session_state.get("run_meta", {})
_stale = (de_results is not None and _meta and
          (_meta.get("group_a") != label_a or _meta.get("group_b") != label_b))

# ── Status banner — always visible ───────────────────────────────────────
n_assigned_a = sum(1 for g in gmap.values() if g == label_a)
n_assigned_b = sum(1 for g in gmap.values() if g == label_b)

if n_assigned_a == 0 or n_assigned_b == 0:
    st.warning(f"⚠️ Assign samples to **{label_a}** and **{label_b}** in sidebar, then click ▶ Run Analysis.")
elif de_results is None:
    st.info(f"✅ {n_assigned_a} in {label_a} · {n_assigned_b} in {label_b} · Now click **▶ Run Analysis**")
elif _stale:
    st.warning(f"⚠️ Results shown are from {_meta.get('group_a')} vs {_meta.get('group_b')}. Click ▶ Run Analysis to refresh for {label_a} vs {label_b}.")
else:
    sig = int((de_results["padj"] <= padj_thresh).sum())
    st.success(f"✅ {len(de_results):,} genes tested · {sig} significant (padj≤{padj_thresh}) · All tabs ready")

if r_status["available"]:
    st.caption("Backend: R available. New analyses will use DESeq2 + ggplot2/pheatmap.")
else:
    st.caption("Backend: R unavailable on this machine. Analyses currently fall back to the Python backend.")

# Run DE
if run:
    if not group_a_idx or not group_b_idx:
        st.error("Assign samples to both groups first.")
    else:
        with st.spinner("Running DE analysis..."):
            fm, fg, fi = filter_low_counts(matrix, genes, min_cpm=min_cpm, min_samples=min_samples)
            count_df_active = pd.DataFrame(matrix, index=genes, columns=samples)
            group_map_active = {s: gmap.get(s, "Unassigned") for s in samples}
            backend_label = "Python fallback"
            manifest = None
            try:
                backend_res = run_r_backend(
                    count_df_active,
                    group_map_active,
                    label_a,
                    label_b,
                    min_cpm,
                    min_samples,
                    padj_thresh,
                    fc_thresh,
                    top_n,
                )
                de = backend_res["de_results"]
                manifest = backend_res["manifest"]
                backend_label = manifest.get("backend_label", "R backend")
                filt_counts = backend_res.get("filtered_counts")
                if filt_counts is not None:
                    st.session_state.filt_mat = filt_counts.to_numpy(dtype=float)
                    st.session_state.filt_genes = list(filt_counts.index)
                    gene_to_idx = {g: i for i, g in enumerate(genes)}
                    st.session_state.filt_idx = [gene_to_idx[g] for g in filt_counts.index if g in gene_to_idx]
                else:
                    st.session_state.filt_mat = fm
                    st.session_state.filt_genes = fg
                    st.session_state.filt_idx = fi
                st.session_state.backend_manifest = manifest
            except Exception as exc:
                de = compute_de(fm, fg, group_a_idx, group_b_idx)
                st.session_state.filt_mat = fm
                st.session_state.filt_genes = fg
                st.session_state.filt_idx = fi
                st.session_state.backend_manifest = None
                st.warning(f"R backend unavailable for this run; used Python fallback instead. Reason: {exc}")

            st.session_state.de_results     = de
            st.session_state.filt_samples   = list(samples)
            st.session_state.run_meta   = {
                "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "group_a": label_a, "group_b": label_b,
                "n_a": len(group_a_idx), "n_b": len(group_b_idx),
                "genes_tested": len(st.session_state.filt_genes), "genes_total": len(genes),
                "min_cpm": min_cpm, "min_samples": min_samples,
                "padj_thresh": padj_thresh, "fc_thresh": fc_thresh,
                "backend": backend_label,
            }
        sig = int((de["padj"] <= padj_thresh).sum())
        st.success(f"✅ {len(de):,} genes tested · {sig} significant (padj≤{padj_thresh})")


# Re-read after potential run
de_results = st.session_state.de_results
backend_manifest = st.session_state.get("backend_manifest")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🔬 QC", "📊 PCA", "🌋 Volcano", "📈 MA Plot",
                "🔥 Heatmap", "🧪 DEG Heatmap", "📋 DEG Table"])

with tabs[0]:
    st.subheader("🔬 Quality Control")
    st.caption("DESeq2-style QC for library balance, dispersion, sample concordance, sample distances, and p-value calibration.")
    if render_backend_plot(backend_manifest, "qc_library_sizes", "R backend: library sizes", "qc_lib"):
        render_backend_plot(backend_manifest, "qc_dispersion", "R backend: dispersion plot", "qc_disp")
        render_backend_plot(backend_manifest, "qc_correlation", "R backend: sample correlation", "qc_corr")
        render_backend_plot(backend_manifest, "qc_sample_distance", "R backend: sample distance matrix", "qc_dist")
        if de_results is not None:
            render_backend_plot(backend_manifest, "qc_pvalue_distribution", "R backend: p-value distribution", "qc_pv")
    else:
        lib_df = pd.DataFrame([(s, int(matrix[:,i].sum()), gmap.get(s,"?"))
                               for i,s in enumerate(samples)],
                              columns=["Sample","Total Counts","Group"])
        lib_df = lib_df.sort_values("Total Counts", ascending=True)
        bar_cols = [PALETTE[all_labels.index(g)%len(PALETTE)] if g in all_labels else "#9ca3af" for g in lib_df["Group"]]
        fig_lib, ax_lib = publication_fig(7.2, max(3.8, len(samples) * 0.34 + 1.8))
        ax_lib.barh(lib_df["Sample"], lib_df["Total Counts"] / 1e6, color=bar_cols, edgecolor="none", height=0.72)
        median_lib = float(np.median(lib_df["Total Counts"] / 1e6))
        ax_lib.axvline(median_lib, color="#6b7280", linestyle="--", linewidth=1.0)
        ax_lib.grid(True, axis="x", color="#e5e7eb", linewidth=0.8)
        ax_lib.set_xlabel("Library size (millions of reads)", fontsize=11)
        ax_lib.set_ylabel("")
        ax_lib.set_title("Library size by sample", fontsize=14, color="#111111", loc="left")
        ax_lib.text(0.0, 1.02, f"Median library size: {median_lib:.2f} M reads", transform=ax_lib.transAxes, fontsize=9, color="#4b5563")
        handles = [Line2D([0], [0], color=PALETTE[all_labels.index(lbl)%len(PALETTE)], lw=6, label=lbl) for lbl in all_labels if lbl in set(lib_df["Group"])]
        if handles:
            ax_lib.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")
        fig_lib.tight_layout()
        st.pyplot(fig_lib, use_container_width=False)
        figure_downloads(fig_lib, "adaptive_qc_library_sizes", "adaptive_qc_lib")
        plt.close(fig_lib)

        fq,_,_ = filter_low_counts(matrix,genes,min_cpm=min_cpm,min_samples=min_samples)
        if fq.shape[0] >= 2:
            lcpm_qc = vst_transform(fq)
            mean_expr = np.log2(np.maximum(fq.mean(axis=1), 0.5))
            disp_proxy = fq.var(axis=1) / np.maximum(fq.mean(axis=1), 1.0)
            fig_d, ax_d = publication_fig(6.8, 4.8)
            ax_d.scatter(mean_expr, np.log10(np.maximum(disp_proxy, 1e-6)), s=12, color="#4c78a8", alpha=0.45, edgecolors="none")
            sx_d, sy_d = smooth_xy(mean_expr, np.log10(np.maximum(disp_proxy, 1e-6)))
            if sx_d is not None:
                ax_d.plot(sx_d, sy_d, color="#111111", linewidth=1.1)
            ax_d.grid(True, color="#e5e7eb", linewidth=0.8)
            ax_d.set_xlabel("log2 mean count", fontsize=11)
            ax_d.set_ylabel("log10 variance / mean", fontsize=11)
            ax_d.set_title("Dispersion trend (Python fallback)", fontsize=14, color="#111111", loc="left")
            ax_d.text(0.0, 1.02, "Approximation of the DESeq2 dispersion diagnostic", transform=ax_d.transAxes, fontsize=9, color="#4b5563")
            fig_d.tight_layout()
            st.pyplot(fig_d, use_container_width=False)
            figure_downloads(fig_d, "adaptive_qc_dispersion_fallback", "adaptive_qc_disp")
            plt.close(fig_d)

            ns = len(samples)
            corr_mat = np.array([[float(stats.spearmanr(lcpm_qc[:,i],lcpm_qc[:,j])[0]) for j in range(ns)] for i in range(ns)])
            fig_c, ax_c = publication_fig(max(5.0, ns * 0.65 + 1.8), max(4.6, ns * 0.55 + 1.8))
            im_c = ax_c.imshow(corr_mat, cmap="RdBu_r", vmin=0.8, vmax=1.0, aspect="equal")
            ax_c.set_xticks(range(ns))
            ax_c.set_yticks(range(ns))
            ax_c.set_xticklabels(samples, rotation=40, ha="right", fontsize=8.5)
            ax_c.set_yticklabels(samples, fontsize=8.5)
            for i in range(ns):
                for j in range(ns):
                    txt_color = "white" if corr_mat[i, j] < 0.9 else "#111111"
                    ax_c.text(j, i, f"{corr_mat[i, j]:.3f}", ha="center", va="center", fontsize=7.5, color=txt_color)
            ax_c.set_title("Sample-to-sample Spearman correlation", fontsize=14, color="#111111", loc="left")
            ax_c.text(0.0, 1.02, "Computed on filtered log2-CPM values", transform=ax_c.transAxes, fontsize=9, color="#4b5563")
            ax_c.set_xlabel("")
            ax_c.set_ylabel("")
            cbar_c = fig_c.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
            cbar_c.ax.tick_params(labelsize=8, colors="#111111")
            cbar_c.ax.set_ylabel("Spearman r", rotation=270, labelpad=12, color="#111111", fontsize=9)
            fig_c.tight_layout()
            st.pyplot(fig_c, use_container_width=False)
            figure_downloads(fig_c, "adaptive_qc_correlation", "adaptive_qc_corr")
            plt.close(fig_c)

            dist_mat = np.sqrt(np.maximum(0.0, ((lcpm_qc.T[:, None, :] - lcpm_qc.T[None, :, :]) ** 2).sum(axis=2)))
            fig_sd, ax_sd = publication_fig(max(5.0, ns * 0.65 + 1.8), max(4.6, ns * 0.55 + 1.8))
            im_sd = ax_sd.imshow(dist_mat, cmap="Greys_r", aspect="equal")
            ax_sd.set_xticks(range(ns))
            ax_sd.set_yticks(range(ns))
            ax_sd.set_xticklabels(samples, rotation=40, ha="right", fontsize=8.5)
            ax_sd.set_yticklabels(samples, fontsize=8.5)
            for i in range(ns):
                for j in range(ns):
                    txt_color = "white" if dist_mat[i, j] > np.nanmedian(dist_mat) else "#111111"
                    ax_sd.text(j, i, f"{dist_mat[i, j]:.1f}", ha="center", va="center", fontsize=7.2, color=txt_color)
            ax_sd.set_title("Sample distance matrix", fontsize=14, color="#111111", loc="left")
            ax_sd.text(0.0, 1.02, "Euclidean distances on filtered log2-CPM values", transform=ax_sd.transAxes, fontsize=9, color="#4b5563")
            ax_sd.set_xlabel("")
            ax_sd.set_ylabel("")
            cbar_sd = fig_sd.colorbar(im_sd, ax=ax_sd, fraction=0.046, pad=0.04)
            cbar_sd.ax.tick_params(labelsize=8, colors="#111111")
            cbar_sd.ax.set_ylabel("Distance", rotation=270, labelpad=12, color="#111111", fontsize=9)
            fig_sd.tight_layout()
            st.pyplot(fig_sd, use_container_width=False)
            figure_downloads(fig_sd, "adaptive_qc_sample_distance", "adaptive_qc_dist")
            plt.close(fig_sd)

        if de_results is not None:
            pv = de_results["pval"].values
            fig_pv, ax_pv = publication_fig(6.8, 4.4)
            ax_pv.hist(pv, bins=20, color="#4c78a8", edgecolor="white", linewidth=0.8)
            ax_pv.axhline(len(pv) / 20.0, color="#6b7280", linestyle="--", linewidth=1.0)
            ax_pv.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
            ax_pv.set_xlabel("P-value", fontsize=11)
            ax_pv.set_ylabel("Gene count", fontsize=11)
            ax_pv.set_title("P-value distribution", fontsize=14, color="#111111", loc="left")
            ax_pv.text(0.0, 1.02, "Dashed line indicates uniform expectation under the null", transform=ax_pv.transAxes, fontsize=9, color="#4b5563")
            fig_pv.tight_layout()
            st.pyplot(fig_pv, use_container_width=False)
            figure_downloads(fig_pv, "adaptive_qc_pvalue_distribution", "adaptive_qc_pv")
            plt.close(fig_pv)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PCA
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("📊 PCA Plot")
    st.caption("Publication-style PCA of log2-CPM normalized counts.")
    if not render_backend_plot(backend_manifest, "pca", "R backend: PCA", "backend_pca"):
        fm2, _, _ = filter_low_counts(matrix, genes, min_cpm, min_samples)
        if fm2 is None or len(fm2) < 2:
            st.warning("No genes passed CPM filter. Try lowering Min CPM in the sidebar.")
        elif len(samples) < 2:
            st.warning("Need at least 2 samples for PCA.")
        else:
            vst2 = vst_transform(fm2)
            pca = pca_compute(vst2, samples)
            if pca:
                groups_unique = sorted(set(gmap.values()))
                cmap = {g: PALETTE[i%len(PALETTE)] for i,g in enumerate(groups_unique)}
                fig_pca, ax_pca = publication_fig(6.8, 5.4)
                for grp in groups_unique:
                    pts = [p for p in pca["points"] if gmap.get(p["name"])==grp]
                    if not pts:
                        continue
                    x_vals = [p["x"] for p in pts]
                    y_vals = [p["y"] for p in pts]
                    add_group_ellipse(ax_pca, x_vals, y_vals, cmap[grp])
                    ax_pca.scatter(x_vals, y_vals, s=58, color=cmap[grp], edgecolors="white", linewidths=0.8, label=grp, zorder=3)
                    for p in pts:
                        ax_pca.annotate(p["name"], (p["x"], p["y"]), xytext=(4, 4), textcoords="offset points", fontsize=8.5, color="#111111")
                ax_pca.axhline(0, color="#9ca3af", linestyle="--", linewidth=0.8, zorder=0)
                ax_pca.axvline(0, color="#9ca3af", linestyle="--", linewidth=0.8, zorder=0)
                ax_pca.grid(True, color="#e5e7eb", linewidth=0.8)
                ax_pca.set_xlabel(f"PC1 ({pca['var1']}% variance)", fontsize=11)
                ax_pca.set_ylabel(f"PC2 ({pca['var2']}% variance)", fontsize=11)
                ax_pca.set_title("Principal component analysis", fontsize=14, color="#111111", loc="left")
                ax_pca.text(0.0, 1.02, "log2-CPM normalized counts", transform=ax_pca.transAxes, fontsize=9, color="#4b5563")
                ax_pca.legend(frameon=False, fontsize=9, loc="best")
                fig_pca.tight_layout()
                st.pyplot(fig_pca, use_container_width=False)
                figure_downloads(fig_pca, "adaptive_pca_publication", "adaptive_pca")
                plt.close(fig_pca)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — VOLCANO
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🌋 Volcano Plot")
    if multi_contrast_mode:
        st.caption(f"Combined volcano plot overlay for all treatment groups vs {label_a}. Selected table contrast remains {label_b} vs {label_a}.")
    else:
        st.caption(f"Publication-style volcano plot for {label_b} vs {label_a}.")
    if _stale:
        st.warning(f"⚠️ Showing results from last run ({_meta.get('group_a')} vs {_meta.get('group_b')}). Click ▶ Run Analysis to update for {label_a} vs {label_b}.")
    if de_results is None:
        st.markdown('<div style="background:#0c2231;border:1px solid #38bdf855;border-radius:8px;padding:14px;color:#38bdf8;">' +
            f'<b>Step 1:</b> Assign samples to <b>{label_a}</b> and <b>{label_b}</b> groups in the sidebar<br>' +
            '<b>Step 2:</b> Click <b>▶ Run Analysis</b> at the bottom of the sidebar<br>' +
            '<b>Step 3:</b> Come back to this tab — plot will appear here</div>',
            unsafe_allow_html=True)
    elif not render_backend_plot(
        backend_manifest,
        "volcano",
        "R backend: combined volcano plot" if multi_contrast_mode else "R backend: volcano plot",
        "backend_volcano",
    ):
        vc1, vc2, vc3, vc4 = st.columns([2, 2, 2, 2])
        use_raw_v     = vc1.toggle("Raw log₂FC",     value=True, help="Switch to unshrunken fold change. Use if points look over-compressed.")
        show_labels_v = vc2.toggle("Label DEG names", value=False, help="Show gene names on significant points.")
        show_ns_v     = vc3.toggle("Show NS points",  value=True,  help="Toggle non-significant grey points on/off.")
        light_mode_v  = vc4.toggle("Light background", value=False, help="Switch to white background for publication.")

        fc_col_v = "log2FC_raw" if use_raw_v else "log2FC"
        dv = de_results.copy()

        def classify_v(r):
            if r["padj"] <= padj_thresh and r[fc_col_v] >= fc_thresh:  return "up"
            if r["padj"] <= padj_thresh and r[fc_col_v] <= -fc_thresh: return "down"
            return "ns"
        dv["cls"] = dv.apply(classify_v, axis=1)

        y_vals = dv["negLog10Padj"].replace([np.inf,-np.inf], np.nan).dropna()
        y_cap  = max(float(y_vals.quantile(0.95))*1.3, -np.log10(padj_thresh)*2.0, 10.0)
        x_vals = dv[fc_col_v].replace([np.inf,-np.inf], np.nan).dropna()
        x_cap  = max(float(x_vals.abs().quantile(0.99))*1.2, fc_thresh+0.5, 1.5)

        dv["y_plot"] = dv["negLog10Padj"].clip(upper=y_cap)
        dv["capped"] = dv["negLog10Padj"] > y_cap

        n_up   = (dv["cls"]=="up").sum()
        n_down = (dv["cls"]=="down").sum()
        n_ns   = (dv["cls"]=="ns").sum()
        n_cap  = dv["capped"].sum()

        fig_v, ax_v = publication_fig(7.0, 5.6)
        if show_ns_v:
            ns_d = dv[dv["cls"]=="ns"]
            ax_v.scatter(ns_d[fc_col_v], ns_d["y_plot"], s=12, color=PUB_COLORS["ns"], alpha=0.35, edgecolors="none", label=f"Not significant ({n_ns})", zorder=1)
        for cls_name, color in [("down", PUB_COLORS["down"]), ("up", PUB_COLORS["up"])]:
            sub_d = dv[dv["cls"]==cls_name]
            if sub_d.empty:
                continue
            capped = sub_d[sub_d["capped"]]
            uncapped = sub_d[~sub_d["capped"]]
            if not uncapped.empty:
                ax_v.scatter(uncapped[fc_col_v], uncapped["y_plot"], s=28, color=color, alpha=0.9, edgecolors="white", linewidths=0.35, label=f"{cls_name.capitalize()} ({len(sub_d)})", zorder=3)
            if not capped.empty:
                ax_v.scatter(capped[fc_col_v], capped["y_plot"], s=34, marker="^", color=color, alpha=0.95, edgecolors="white", linewidths=0.35, zorder=4)

        if show_labels_v:
            sig_d = dv[dv["cls"]!="ns"].nsmallest(12, "padj")
            for _, row in sig_d.iterrows():
                ax_v.annotate(row["gene"], (row[fc_col_v], row["y_plot"]), xytext=(4, 3), textcoords="offset points", fontsize=8, color="#111111")

        y_sig = -np.log10(padj_thresh)
        ax_v.axhline(y_sig, color="#6b7280", linestyle="--", linewidth=1.0)
        if fc_thresh > 0:
            ax_v.axvline(fc_thresh, color="#9ca3af", linestyle=":", linewidth=1.0)
            ax_v.axvline(-fc_thresh, color="#9ca3af", linestyle=":", linewidth=1.0)
        ax_v.set_xlim(-x_cap, x_cap)
        ax_v.set_ylim(0, y_cap * 1.08)
        ax_v.grid(True, color="#e5e7eb", linewidth=0.8)
        ax_v.set_xlabel("log2 fold change (raw)" if use_raw_v else "log2 fold change (shrunken)", fontsize=11)
        ax_v.set_ylabel("-log10 adjusted p-value", fontsize=11)
        ax_v.set_title("Volcano plot", fontsize=14, color="#111111", loc="left")
        cap_note = f" | capped points: {n_cap}" if n_cap > 0 else ""
        ax_v.text(0.0, 1.02, f"{label_b} vs {label_a} | up: {n_up} | down: {n_down} | padj <= {padj_thresh} | |log2FC| >= {fc_thresh}{cap_note}", transform=ax_v.transAxes, fontsize=8.8, color="#4b5563")
        ax_v.legend(frameon=False, fontsize=9, loc="upper right")
        fig_v.tight_layout()
        st.pyplot(fig_v, use_container_width=False)
        figure_downloads(fig_v, "adaptive_volcano_publication", "adaptive_volcano")
        plt.close(fig_v)
        if n_cap > 0:
            st.caption(f"▲ = {n_cap} gene(s) capped at y={y_cap:.0f} for display. Hover to see true values.")


with tabs[3]:
    st.subheader("MA Plot")
    if de_results is None:
        st.info(f"Click ▶ Run Analysis in the sidebar to compare {label_b} vs {label_a}.")
    elif not render_backend_plot(
        backend_manifest,
        "ma",
        "R backend: combined MA plot" if multi_contrast_mode else "R backend: MA plot",
        "backend_ma",
    ):
        df_m=de_results.copy()
        def cls_m(r):
            if r["padj"]<=padj_thresh and r["log2FC_raw"]>=fc_thresh: return "up"
            if r["padj"]<=padj_thresh and r["log2FC_raw"]<=-fc_thresh: return "down"
            return "ns"
        df_m["cls"]=df_m.apply(cls_m,axis=1)
        df_m["A"]=np.log2(df_m["baseMean"]+0.5)
        fig_m, ax_m = publication_fig(7.0, 5.2)
        for c, grp in df_m.groupby("cls"):
            ax_m.scatter(
                grp["A"], grp["log2FC_raw"],
                s=14 if c == "ns" else 20,
                color=PUB_COLORS[c],
                alpha=0.35 if c == "ns" else 0.82,
                edgecolors="none",
                label=f"{'Up' if c=='up' else 'Down' if c=='down' else 'Not significant'} ({len(grp)})",
            )
        sx, sy = smooth_xy(df_m["A"].to_numpy(), df_m["log2FC_raw"].to_numpy())
        if sx is not None:
            ax_m.plot(sx, sy, color="#111111", linewidth=1.2, label="Trend")
        ax_m.axhline(0, color="#6b7280", linestyle="--", linewidth=1.0)
        ax_m.grid(True, color="#e5e7eb", linewidth=0.8)
        ax_m.set_xlabel("A = log2(mean CPM)", fontsize=11)
        ax_m.set_ylabel("M = log2 fold change", fontsize=11)
        ax_m.set_title("MA plot", fontsize=14, color="#111111", loc="left")
        ax_m.text(0.0, 1.02, f"{label_b} vs {label_a}", transform=ax_m.transAxes, fontsize=9, color="#4b5563")
        ax_m.legend(frameon=False, fontsize=9, loc="upper right")
        fig_m.tight_layout()
        st.pyplot(fig_m, use_container_width=False)
        figure_downloads(fig_m, "adaptive_ma_publication", "adaptive_ma")
        plt.close(fig_m)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPRESSION HEATMAP
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Expression Heatmap")
    st.caption(f"Publication-style heatmap of the top {top_n} most variable genes.")
    if not render_backend_plot(backend_manifest, "expression_heatmap", "R backend: expression heatmap", "backend_expr_heatmap"):
        fm3, _, fi3 = filter_low_counts(matrix, genes, min_cpm, min_samples)
        if fm3 is None or fm3.shape[0] == 0:
            st.warning("No genes passed CPM filter. Try lowering Min CPM.")
        elif True:
            lcpm3 = vst_transform(fm3)
            top_loc = np.argsort(lcpm3.var(axis=1))[::-1][:top_n]
            top_glob = [fi3[i] for i in top_loc]
            sub_z = zscore_rows(lcpm3[top_loc])
            fig_h = publication_heatmap(sub_z, [genes[i] for i in top_glob], samples, [gmap.get(s, "Unassigned") for s in samples], f"Top {top_n} variable genes")
            st.pyplot(fig_h,use_container_width=False)
            figure_downloads(fig_h, "adaptive_expression_heatmap_publication", "adaptive_expr_heatmap")
            plt.close(fig_h)

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — DEG HEATMAP
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("DEG Heatmap")
    if multi_contrast_mode:
        st.caption("Hybrid DEG heatmap built from all treatment-vs-reference contrasts using shared + strong DEG ranking.")
    if de_results is None or len(de_results) == 0:
        st.info(f"Click ▶ Run Analysis in the sidebar to compare {label_b} vs {label_a}.")
    elif not render_backend_plot(
        backend_manifest,
        "deg_heatmap",
        "R backend: hybrid DEG heatmap" if multi_contrast_mode else f"R backend: DEG heatmap for {label_b} vs {label_a}",
        "backend_deg_heatmap",
    ):
        g2i={g:i for i,g in enumerate(genes)}
        up_g=de_results[(de_results["padj"]<=padj_thresh)&(de_results["log2FC"]>=fc_thresh)].sort_values("log2FC",ascending=False)["gene"].tolist()
        dn_g=de_results[(de_results["padj"]<=padj_thresh)&(de_results["log2FC"]<=-fc_thresh)].sort_values("log2FC")["gene"].tolist()
        show=st.radio("Show",["All DEGs","↑ Up only","↓ Down only"],horizontal=True)
        idx_list=([g2i[g] for g in (up_g+dn_g) if g in g2i] if show=="All DEGs"
                  else [g2i[g] for g in up_g if g in g2i] if "Up" in show
                  else [g2i[g] for g in dn_g if g in g2i])
        if not idx_list:
            st.warning("No DEGs at current thresholds.")
        else:
            lcpm_h=vst_transform(matrix)
            sub_h=zscore_rows(lcpm_h[idx_list])
            heat_title = f"DEG heatmap ({len(idx_list)} genes, padj <= {padj_thresh})"
            figh = publication_heatmap(sub_h, [genes[i] for i in idx_list], samples, [gmap.get(s, "Unassigned") for s in samples], heat_title)
            st.pyplot(figh,use_container_width=False)
            figure_downloads(figh, "adaptive_deg_heatmap_publication", "adaptive_deg_heatmap")
            plt.close(figh)

# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — DEG TABLE
# ════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("DEG Results Table")
    if de_results is None:
        st.info(f"Click ▶ Run Analysis in the sidebar to compare {label_b} vs {label_a}.")
    else:
        summary_source = (backend_manifest or {}).get("deg_summary") if backend_manifest else None
        if summary_source:
            summary_rows = []
            for item in normalize_backend_summary(summary_source):
                summary_rows.append({
                    "Contrast": item.get("contrast", "?"),
                    "Total DEGs": item.get("total_deg", 0),
                    "Up-regulated": item.get("up_regulated", 0),
                    "Down-regulated": item.get("down_regulated", 0),
                })
            if summary_rows:
                st.caption("DEG count summary across all treatment-vs-control contrasts.")
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        df_t=de_results.copy()
        def reg(r):
            if r["padj"]<=padj_thresh and r["log2FC"]>=fc_thresh: return "↑ UP"
            if r["padj"]<=padj_thresh and r["log2FC"]<=-fc_thresh: return "↓ DOWN"
            return "NS"
        df_t["Regulation"]=df_t.apply(reg,axis=1)
        c1,c2,c3=st.columns([2,1,1])
        srch=c1.text_input("🔍 Search gene","")
        flt=c2.selectbox("Filter",["All","Significant","↑ Up","↓ Down"])
        n_rows=c3.number_input("Rows",10,500,25,10)
        if srch: df_t=df_t[df_t["gene"].str.contains(srch,case=False)]
        if flt=="Significant": df_t=df_t[df_t["Regulation"]!="NS"]
        elif flt=="↑ Up": df_t=df_t[df_t["Regulation"]=="↑ UP"]
        elif flt=="↓ Down": df_t=df_t[df_t["Regulation"]=="↓ DOWN"]
        disp_t=df_t[["gene","baseMean","log2FC","log2FC_raw","pval","padj","Regulation"]].head(n_rows)
        disp_t.columns=["Gene","Base Mean","log₂FC (shrunk)","log₂FC (raw)","p-value","padj","Regulation"]
        st.dataframe(
            disp_t.style
              .applymap(lambda v:"color:#f97316;font-weight:bold" if v=="↑ UP" else "color:#38bdf8;font-weight:bold" if v=="↓ DOWN" else "color:#475569",subset=["Regulation"])
              .format({"Base Mean":"{:.1f}","log₂FC (shrunk)":"{:.3f}","log₂FC (raw)":"{:.3f}","p-value":"{:.2e}","padj":"{:.2e}"}),
            use_container_width=True, hide_index=True, height=420,
        )
        csv_buf=io.StringIO()
        rm=st.session_state.get("run_meta",{})
        csv_buf.write(f"# TranscriptomicViz Adaptive — DE Results\n")
        csv_buf.write(f"# {rm.get('timestamp','')}\n")
        csv_buf.write(f"# {label_b} vs {label_a}\n")
        csv_buf.write("# Organism: unknown\n")
        csv_buf.write(f"# Normalization: TMM | Test: NB score | FDR: BH | Shrinkage: MAD-based Bayesian\n#\n")
        de_results.to_csv(csv_buf,index=False,float_format="%.6g")
        st.download_button("⬇ Download CSV",csv_buf.getvalue().encode(),
                           f"DEG_{label_b}_vs_{label_a}_adaptive.csv","text/csv",use_container_width=True)
        result_files = (backend_manifest or {}).get("result_files") if backend_manifest else None
        if result_files:
            st.caption("Per-contrast DESeq2 result files from the R backend.")
            dl_cols = st.columns(min(3, len(result_files)))
            for idx, item in enumerate(result_files):
                csv_path = item.get("csv")
                if not csv_path or not Path(csv_path).exists():
                    continue
                with dl_cols[idx % len(dl_cols)]:
                    st.download_button(
                        f"⬇ {item.get('contrast', 'Contrast CSV')}",
                        Path(csv_path).read_bytes(),
                        Path(csv_path).name,
                        "text/csv",
                        key=f"contrast_csv_{idx}",
                        use_container_width=True,
                    )
