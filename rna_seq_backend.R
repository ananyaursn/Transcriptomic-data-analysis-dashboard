args <- commandArgs(trailingOnly = TRUE)
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
