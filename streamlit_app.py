"""
TranscriptomicViz — Streamlit Web App
Run with: streamlit run streamlit_app.py

Requires transcriptomic_viz.py to be in the same folder.
"""

import io
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Import all logic from the pipeline script ──────────────────────────────
from transcriptomic_viz import (
    detect_format, parse_htseq, parse_featurecounts, merge_files,
    compute_cpm, vst_transform, filter_low_counts, zscore_rows,
    compute_de, pca_compute, bh_correction,
    PALETTE,
)
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TranscriptomicViz",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Dark background */
  .stApp { background-color: #020817; color: #e2e8f0; }
  section[data-testid="stSidebar"] { background-color: #0a1628; }
  section[data-testid="stSidebar"] * { color: #94a3b8 !important; }

  /* Headers */
  h1, h2, h3 { color: #38bdf8 !important; font-family: 'Space Grotesk', sans-serif; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 10px;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #0a1628; border-bottom: 1px solid #1e293b; }
  .stTabs [data-baseweb="tab"] { color: #475569; }
  .stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8; }

  /* Buttons */
  .stButton > button {
    background: #0f172a;
    border: 1px solid #334155;
    color: #94a3b8;
    border-radius: 6px;
  }
  .stButton > button:hover { border-color: #38bdf8; color: #38bdf8; }

  /* Dataframe */
  .stDataFrame { border: 1px solid #1e293b; border-radius: 8px; }

  /* Warning / info boxes */
  .warn-box  { background:#2a1500; border:1px solid #f9731655; border-radius:8px; padding:8px 12px; margin-bottom:6px; color:#f97316; font-size:13px; }
  .error-box { background:#2a0a0a; border:1px solid #ef444455; border-radius:8px; padding:8px 12px; margin-bottom:6px; color:#ef4444; font-size:13px; }
  .info-box  { background:#0c2231; border:1px solid #38bdf855; border-radius:8px; padding:8px 12px; margin-bottom:6px; color:#38bdf8; font-size:13px; }
  .ok-box    { background:#0a2010; border:1px solid #4ade8055; border-radius:8px; padding:8px 12px; margin-bottom:6px; color:#4ade80; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_uploaded(uploaded_file) -> pd.DataFrame:
    """Parse a Streamlit UploadedFile into a counts DataFrame."""
    text = uploaded_file.read().decode("utf-8", errors="replace")
    sample_name = re.sub(r"\.(txt|tsv|csv)$", "", uploaded_file.name, flags=re.IGNORECASE)
    sample_name = re.sub(r"_counts?$", "", sample_name, flags=re.IGNORECASE)
    fmt = detect_format(text)
    if fmt == "featurecounts":
        return parse_featurecounts(text, sample_name), "featureCounts"
    return parse_htseq(text, sample_name), "HTSeq"


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def dark_fig(w=9, h=6):
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


# ══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS (return fig for st.pyplot)
# ══════════════════════════════════════════════════════════════════════════════

def make_pca(pca_result, group_map, label_a, label_b, filter_info=None):
    """Interactive Plotly PCA plot with hover tooltips."""
    points = pca_result["points"]
    var1, var2 = pca_result["var1"], pca_result["var2"]
    groups = sorted(set(group_map.values()))
    color_map = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}

    fig = go.Figure()
    for grp in groups:
        pts = [p for p in points if group_map.get(p["name"], groups[0]) == grp]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[p["x"] for p in pts],
            y=[p["y"] for p in pts],
            mode="markers+text",
            name=grp,
            text=[p["name"] for p in pts],
            textposition="top center",
            textfont=dict(color="#94a3b8", size=10),
            marker=dict(
                color=color_map[grp],
                size=14,
                opacity=0.88,
                line=dict(color="rgba(255,255,255,0.2)", width=1),
            ),
            customdata=[[p["name"], f"{p['x']:.3f}", f"{p['y']:.3f}", grp] for p in pts],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "────────────────────<br>"
                "Group: <b>%{customdata[3]}</b><br>"
                "PC1:   <b>%{customdata[1]}</b><br>"
                "PC2:   <b>%{customdata[2]}</b><br>"
                "<extra></extra>"
            ),
            hoverlabel=dict(
                bgcolor="#0f172a",
                bordercolor="#334155",
                font=dict(color="#e2e8f0", size=12, family="monospace"),
            ),
        ))

    subtitle = ""
    if filter_info:
        subtitle = f"<br><sup>✓ {filter_info['kept']} genes passed CPM filter · {filter_info['removed']} removed · log2-CPM normalized</sup>"

    fig.add_hline(y=0, line=dict(color="#334155", dash="dash", width=1))
    fig.add_vline(x=0, line=dict(color="#334155", dash="dash", width=1))

    fig.update_layout(
        title=dict(
            text=f"PCA Plot — log2-CPM Normalized{subtitle}",
            font=dict(color="#e2e8f0", size=16),
        ),
        xaxis=dict(
            title=f"PC1 ({var1}% variance)",
            title_font=dict(color="#94a3b8"),
            tickfont=dict(color="#64748b"),
            gridcolor="#1e293b",
            zerolinecolor="#334155",
        ),
        yaxis=dict(
            title=f"PC2 ({var2}% variance)",
            title_font=dict(color="#94a3b8"),
            tickfont=dict(color="#64748b"),
            gridcolor="#1e293b",
            zerolinecolor="#334155",
        ),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#020817",
        legend=dict(
            font=dict(color="#94a3b8", size=12),
            bgcolor="#0f172a",
            bordercolor="#334155",
            borderwidth=1,
        ),
        height=520,
        hovermode="closest",
        margin=dict(t=80, l=60, r=40, b=60),
    )
    return fig


def make_volcano(de_results, fc_thresh, padj_thresh, label_a, label_b, use_raw_fc=False):
    """Interactive Plotly volcano plot with hover tooltips."""
    df = de_results.copy()

    # Choose FC column
    fc_col = "log2FC_raw" if use_raw_fc else "log2FC"
    fc_label = "Raw log₂FC" if use_raw_fc else "Shrunken log₂FC"

    def cls(row):
        if row["padj"] <= padj_thresh and row[fc_col] >= fc_thresh:
            return "up"
        if row["padj"] <= padj_thresh and row[fc_col] <= -fc_thresh:
            return "down"
        return "ns"

    df["cls"] = df.apply(cls, axis=1)

    # Smart y-axis cap: show 95th percentile + 20% headroom
    # This prevents a few extreme padj values from collapsing all other points
    y_raw   = df["negLog10Padj"].replace([np.inf, -np.inf], np.nan).dropna()
    y_95    = float(y_raw.quantile(0.95))
    y_max   = max(y_95 * 1.25, -np.log10(padj_thresh) * 1.5, 10)
    df["y"] = df["negLog10Padj"].clip(upper=y_max)
    # Mark genes that were capped
    df["capped"] = df["negLog10Padj"] > y_max

    df["size"] = df["cls"].map({"up": 9, "down": 9, "ns": 5})
    df["regulation"] = df["cls"].map({
        "up":   "🟠 Upregulated",
        "down": "🔵 Downregulated",
        "ns":   "⚫ Not significant"
    })

    color_map = {
        "🟠 Upregulated":    "#f97316",
        "🔵 Downregulated":  "#38bdf8",
        "⚫ Not significant": "#1e3a4a",
    }

    fig = go.Figure()

    for label, group in df.groupby("regulation", sort=False):
        # Split capped vs normal for marker symbol
        for capped, subset in group.groupby("capped"):
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset[fc_col],
                y=subset["y"],
                mode="markers",
                name=label,
                showlegend=(not capped),
                marker=dict(
                    color=color_map[label],
                    size=subset["size"],
                    opacity=0.35 if "significant" in label else 0.88,
                    symbol="triangle-up" if capped else "circle",
                    line=dict(width=0),
                ),
                customdata=np.stack([
                    subset["gene"],
                    subset[fc_col].round(3),
                    subset["log2FC_raw"].round(3),
                    subset["log2FC"].round(3),
                    subset["padj"].apply(lambda v: f"{v:.2e}"),
                    subset["pval"].apply(lambda v: f"{v:.2e}"),
                    subset["baseMean"].round(1),
                    ["▲ capped" if c else "" for c in subset["capped"]],
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b>  %{customdata[7]}<br>"
                    "──────────────────────<br>"
                    f"{fc_label}: <b>%{{customdata[1]}}</b><br>"
                    "log₂FC (raw):     %{customdata[2]}<br>"
                    "log₂FC (shrunk):  %{customdata[3]}<br>"
                    "padj (BH):        <b>%{customdata[4]}</b><br>"
                    "p-value:          %{customdata[5]}<br>"
                    "Base mean CPM:    %{customdata[6]}<br>"
                    "<extra></extra>"
                ),
                hoverlabel=dict(
                    bgcolor="#0f172a",
                    bordercolor="#334155",
                    font=dict(color="#e2e8f0", size=12, family="monospace"),
                ),
            ))

    y_line = -np.log10(padj_thresh)
    x_vals = df[fc_col].replace([np.inf, -np.inf], np.nan).dropna()
    x_max  = max(float(x_vals.abs().quantile(0.99)) * 1.25, fc_thresh + 0.5, 1.5)

    n_up   = (df["cls"] == "up").sum()
    n_down = (df["cls"] == "down").sum()
    n_ns   = (df["cls"] == "ns").sum()
    n_cap  = df["capped"].sum()
    cap_note = f"  ·  ▲{n_cap} capped at y={y_max:.0f}" if n_cap > 0 else ""

    # Threshold lines
    fig.add_hline(y=y_line, line=dict(color="#ef4444", dash="dash", width=1.5),
                  annotation_text=f"padj={padj_thresh}",
                  annotation_font=dict(color="#ef4444", size=11))
    fig.add_vline(x= fc_thresh, line=dict(color="#475569", dash="dash", width=1))
    fig.add_vline(x=-fc_thresh, line=dict(color="#475569", dash="dash", width=1))

    fig.update_layout(
        title=dict(
            text=(f"Volcano Plot — <b>{label_b}</b> vs <b>{label_a}</b>"
                  f"<br><sup>🟠 {n_up} up  ·  🔵 {n_down} down  ·  ⚫ {n_ns} NS"
                  f"  ·  padj≤{padj_thresh}  ·  |log₂FC|≥{fc_thresh}{cap_note}</sup>"),
            font=dict(color="#e2e8f0", size=16),
        ),
        xaxis=dict(
            title=fc_label,
            title_font=dict(color="#94a3b8"),
            tickfont=dict(color="#64748b"),
            gridcolor="#1e293b",
            zerolinecolor="#334155",
            range=[-x_max, x_max],
        ),
        yaxis=dict(
            title="-log₁₀(padj) [BH]",
            title_font=dict(color="#94a3b8"),
            tickfont=dict(color="#64748b"),
            gridcolor="#1e293b",
            zerolinecolor="#334155",
            range=[0, y_max * 1.05],
        ),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#020817",
        legend=dict(
            font=dict(color="#94a3b8", size=12),
            bgcolor="#0f172a",
            bordercolor="#334155",
            borderwidth=1,
        ),
        height=560,
        hovermode="closest",
        margin=dict(t=90, l=60, r=40, b=60),
    )
    return fig


def make_heatmap(matrix, genes, samples, gene_indices, title, subtitle):
    if not gene_indices:
        return None
    vst = vst_transform(matrix)
    sub = vst[gene_indices]
    z = zscore_rows(sub)

    n_genes, n_samples = len(gene_indices), len(samples)
    cell_h = max(0.2, min(0.55, 12 / max(n_genes, 1)))
    fig_h = max(4, n_genes * cell_h + 2.5)
    fig_w = max(6, n_samples * 0.9 + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#020817")
    ax.set_facecolor("#020817")

    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    ax.set_xticks(range(n_samples))
    ax.set_xticklabels(samples, rotation=40, ha="right",
                       fontsize=8, color="#94a3b8")
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels([genes[i] for i in gene_indices],
                       fontsize=max(5, min(9, 110 // n_genes)), color="#64748b")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    im = ax.imshow(z, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors="#64748b")
    cbar.ax.set_ylabel("z-score", color="#64748b", fontsize=8)
    cbar.outline.set_edgecolor("#334155")

    ax.set_title(f"{title}\n{subtitle}", fontsize=10, color="#e2e8f0", pad=8)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB FALLBACKS (used only for PNG download buttons)
# ══════════════════════════════════════════════════════════════════════════════

def _volcano_mpl(de_results, fc_thresh, padj_thresh, label_a, label_b):
    """Static matplotlib volcano — used only for PNG export."""
    fig, ax = dark_fig(9, 6)
    df = de_results.copy()
    def cls(r):
        if r["padj"] <= padj_thresh and r["log2FC"] >= fc_thresh: return "up"
        if r["padj"] <= padj_thresh and r["log2FC"] <= -fc_thresh: return "down"
        return "ns"
    df["cls"] = df.apply(cls, axis=1)
    colors = {"up": "#f97316", "down": "#38bdf8", "ns": "#1e3a4a"}
    q999 = df["negLog10Padj"].quantile(0.999)
    for c, grp in df.groupby("cls"):
        y = np.minimum(grp["negLog10Padj"], q999)
        ax.scatter(grp["log2FC"], y, color=colors[c],
                   alpha=0.35 if c=="ns" else 0.85,
                   s=7 if c=="ns" else 20,
                   label=f"{'↑ Up' if c=='up' else '↓ Down' if c=='down' else 'NS'} ({len(grp)})",
                   edgecolors="none")
    y_line = -np.log10(padj_thresh)
    x_max = df["log2FC"].abs().max() * 1.1
    ax.axhline(y_line, color="#ef4444", linestyle="--", linewidth=1.2)
    ax.axvline(fc_thresh, color="#475569", linestyle="--", linewidth=0.8)
    ax.axvline(-fc_thresh, color="#475569", linestyle="--", linewidth=0.8)
    ax.set_xlim(-x_max, x_max)
    ax.set_xlabel("Shrunken log₂ Fold Change", fontsize=11)
    ax.set_ylabel("-log₁₀(padj) [BH]", fontsize=11)
    ax.set_title(f"Volcano Plot — {label_b} vs {label_a}", fontsize=13, fontweight="bold", color="#e2e8f0")
    ax.legend(loc="upper left", framealpha=0.2, facecolor="#0f172a", edgecolor="#334155", labelcolor="#94a3b8", fontsize=9)
    ax.grid(True, color="#1e293b", linewidth=0.5)
    fig.tight_layout()
    return fig


def _pca_mpl(pca_result, group_map, label_a, label_b, filter_info=None):
    """Static matplotlib PCA — used only for PNG export."""
    fig, ax = dark_fig(8, 6)
    points = pca_result["points"]
    var1, var2 = pca_result["var1"], pca_result["var2"]
    groups = sorted(set(group_map.values()))
    color_map = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
    for pt in points:
        grp = group_map.get(pt["name"], groups[0])
        color = color_map[grp]
        ax.scatter(pt["x"], pt["y"], color=color, s=90, zorder=3, edgecolors=(1, 1, 1, 0.27), linewidths=0.5)
        ax.annotate(pt["name"], (pt["x"], pt["y"]), textcoords="offset points", xytext=(0,10), fontsize=8, color="#94a3b8", ha="center")
    for grp, color in color_map.items():
        ax.scatter([], [], color=color, label=grp, s=60)
    ax.legend(loc="upper right", framealpha=0.2, facecolor="#0f172a", edgecolor="#334155", labelcolor="#94a3b8", fontsize=9)
    ax.axhline(0, color="#334155", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="#334155", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({var1}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2}% variance)", fontsize=11)
    ax.set_title("PCA Plot — log2-CPM Normalized", fontsize=13, fontweight="bold", color="#e2e8f0")
    ax.grid(True, color="#1e293b", linewidth=0.5)
    if filter_info:
        ax.text(0.01, 0.01, f"{filter_info['kept']} genes passed CPM filter", transform=ax.transAxes, fontsize=8, color="#4ade80")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

for key, default in [
    ("count_df", None), ("samples", []), ("genes", []),
    ("matrix", None), ("group_map", {}), ("de_results", None),
    ("group_labels", ["Control", "Treatment"]),
    ("compare_a", "Control"), ("compare_b", "Treatment"),
    ("selected_samples", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧬 TranscriptomicViz")
    st.caption("TMM · NB test · BH-FDR · LFC-shrink · VST")
    st.divider()

    # ── File upload ──────────────────────────────────────────────────────────
    st.markdown("### 📂 Upload Count Files")
    uploaded = st.file_uploader(
        "HTSeq or featureCounts (auto-detected)",
        type=["txt", "tsv", "csv"],
        accept_multiple_files=True,
        help="Upload one or more count files. Mix of HTSeq and featureCounts is fine."
    )

    if uploaded:
        dfs = []
        fmt_log = []
        for f in uploaded:
            try:
                df, fmt = parse_uploaded(f)
                if not df.empty:
                    dfs.append(df)
                    fmt_log.append(f"✓ {f.name} [{fmt}]")
            except Exception as e:
                st.error(f"Error reading {f.name}: {e}")

        if dfs:
            merged = merge_files(dfs)
            # Deduplicate column names if same sample appears twice
            merged = merged.loc[:, ~merged.columns.duplicated()]
            merged = merged.loc[:, ~merged.columns.duplicated()]
            st.session_state.count_df = merged
            st.session_state.samples  = list(merged.columns)
            st.session_state.genes    = list(merged.index)
            st.session_state.matrix   = merged.values.astype(float)
            st.session_state.de_results = None
            st.session_state.selected_samples = None  # reset on new upload

            with st.expander(f"Loaded {len(dfs)} file(s)", expanded=False):
                for l in fmt_log:
                    st.caption(l)

    # ── Group assignment ─────────────────────────────────────────────────────
    if st.session_state.samples:
        st.divider()
        st.markdown("### 🏷️ Group Labels")

        # Manage group label list
        labels = st.session_state.group_labels
        new_labels = []
        for i, lbl in enumerate(labels):
            c1, c2 = st.columns([4, 1])
            edited = c1.text_input(f"Group {i+1}", value=lbl, key=f"lbl_edit_{i}")
            new_labels.append(edited)
            if c2.button("✕", key=f"del_lbl_edit_{i}", help="Remove group") and len(labels) > 2:
                labels.pop(i)
                st.session_state.group_labels = labels
                st.rerun()

        st.session_state.group_labels = new_labels
        labels = new_labels

        if st.button("＋ Add Group", use_container_width=True):
            labels.append(f"Group {len(labels)+1}")
            st.session_state.group_labels = labels
            st.rerun()

        # Assign each sample to a group
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
            if prev not in labels:
                prev = labels[0]
            idx = labels.index(prev)
            choice = st.selectbox(s, labels, index=idx, key=f"sample_select_{si}")
            group_map[s] = choice
        st.session_state.group_map = group_map

        # ── Comparison selector (for DE / Volcano) ───────────────────────────
        st.divider()
        st.markdown("### ⚖️ Compare Groups")
        st.caption("Select which two groups to compare for DE analysis.")
        col1, col2 = st.columns(2)
        compare_a = col1.selectbox("Group A (reference)", labels,
                                   index=labels.index(st.session_state.compare_a)
                                   if st.session_state.compare_a in labels else 0,
                                   key="sel_compare_a")
        remaining = [l for l in labels if l != compare_a]
        default_b = st.session_state.compare_b if st.session_state.compare_b in remaining else remaining[0]
        compare_b = col2.selectbox("Group B (treatment)", remaining,
                                   index=remaining.index(default_b),
                                   key="sel_compare_b")
        st.session_state.compare_a = compare_a
        st.session_state.compare_b = compare_b
        label_a, label_b = compare_a, compare_b

        # ── Organism Presets ─────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🧫 Organism Preset")
        st.caption("Auto-fills all thresholds below. You can still adjust manually.")

        PRESETS = {
            "— select —": None,
            "🧑 Human (Homo sapiens)": {
                "min_cpm": 1.0, "padj": 0.05, "fc": 1.0, "top_n": 50,
                "note": "Standard thresholds. ~20,000 protein-coding genes. CPM≥1 removes most noise."
            },
            "🐭 Mouse (Mus musculus)": {
                "min_cpm": 1.0, "padj": 0.05, "fc": 1.0, "top_n": 50,
                "note": "Same as human — transcriptomes are similar in size and expression range."
            },
            "🐟 Zebrafish (Danio rerio)": {
                "min_cpm": 1.0, "padj": 0.05, "fc": 1.0, "top_n": 50,
                "note": "~26,000 genes. Standard thresholds work well."
            },
            "🪰 Fruit fly (Drosophila melanogaster)": {
                "min_cpm": 1.0, "padj": 0.05, "fc": 1.0, "top_n": 40,
                "note": "~14,000 genes — smaller genome. Slightly fewer top-N needed."
            },
            "🪱 C. elegans (nematode)": {
                "min_cpm": 1.0, "padj": 0.05, "fc": 1.0, "top_n": 40,
                "note": "~20,000 genes. Expression ranges similar to other metazoans."
            },
            "🌿 Arabidopsis thaliana": {
                "min_cpm": 0.5, "padj": 0.05, "fc": 1.0, "top_n": 50,
                "note": "~27,000 genes. Lower CPM cutoff — plant transcriptomes have many lowly expressed genes."
            },
            "🌾 Rice (Oryza sativa)": {
                "min_cpm": 0.5, "padj": 0.05, "fc": 1.0, "top_n": 50,
                "note": "~40,000 genes. Large genome — lower CPM avoids over-filtering."
            },
            "🍺 Yeast (S. cerevisiae)": {
                "min_cpm": 2.0, "padj": 0.05, "fc": 1.0, "top_n": 30,
                "note": "~6,000 genes. Raise CPM cutoff — compact genome, high expression, less noise."
            },
            "🍞 S. pombe (fission yeast)": {
                "min_cpm": 2.0, "padj": 0.05, "fc": 1.0, "top_n": 30,
                "note": "~5,100 genes. Similar to S. cerevisiae — higher CPM, fewer top-N."
            },
            "🦠 E. coli (bacterium)": {
                "min_cpm": 5.0, "padj": 0.05, "fc": 1.5, "top_n": 25,
                "note": "~4,200 genes. Bacteria have no introns — reads map densely. Higher CPM & FC recommended."
            },
            "🦠 Enterobacter cloacae": {
                "min_cpm": 3.0, "padj": 0.05, "fc": 1.5, "top_n": 30,
                "note": "~4,500 genes. Gram-negative Enterobacteriaceae. CPM>=3 balances noise vs gene retention. FC>=1.5 standard for bacteria."
            },
            "🦠 Enterobacter hormaechei": {
                "min_cpm": 3.0, "padj": 0.05, "fc": 1.5, "top_n": 30,
                "note": "~5,000 genes. Clinical isolate — expect many hypothetical protein IDs in results. Same thresholds as E. cloacae."
            },
            "🦠 Enterobacter aerogenes (Klebsiella aerogenes)": {
                "min_cpm": 3.0, "padj": 0.05, "fc": 1.5, "top_n": 30,
                "note": "~5,200 genes. Reclassified to Klebsiella in some databases. Slightly larger genome, same bacterial thresholds apply."
            },
            "🧫 Bacillus subtilis": {
                "min_cpm": 5.0, "padj": 0.05, "fc": 1.5, "top_n": 25,
                "note": "~4,100 genes. Same logic as E. coli — raise CPM and FC for bacterial data."
            },
            "🦠 Pseudomonas aeruginosa": {
                "min_cpm": 5.0, "padj": 0.05, "fc": 1.5, "top_n": 25,
                "note": "~5,500 genes. Larger bacterial genome but same high-expression logic applies."
            },
            "🧫 Mycobacterium tuberculosis": {
                "min_cpm": 5.0, "padj": 0.05, "fc": 1.5, "top_n": 20,
                "note": "~4,000 genes. Slow-growing — counts can be lower. FC≥1.5 helps reduce false hits."
            },
            "🍄 Aspergillus / Neurospora (fungi)": {
                "min_cpm": 1.0, "padj": 0.05, "fc": 1.0, "top_n": 35,
                "note": "~10,000 genes. Fungal transcriptomes sit between yeast and metazoans."
            },
            "🐀 Rat (Rattus norvegicus)": {
                "min_cpm": 1.0, "padj": 0.05, "fc": 1.0, "top_n": 50,
                "note": "~21,000 genes. Essentially identical to mouse settings."
            },
            "🐄 Cow / Pig / Sheep (livestock)": {
                "min_cpm": 1.0, "padj": 0.05, "fc": 1.0, "top_n": 50,
                "note": "~20–25,000 genes. Annotation quality varies — standard thresholds are safe."
            },
            "🐝 Honey bee (Apis mellifera)": {
                "min_cpm": 1.0, "padj": 0.05, "fc": 1.0, "top_n": 35,
                "note": "~10,000 genes. Insect-size genome, standard thresholds work."
            },
            "🦟 Mosquito (Anopheles / Aedes)": {
                "min_cpm": 1.0, "padj": 0.05, "fc": 1.0, "top_n": 35,
                "note": "~13,000 genes. Similar to Drosophila."
            },
            "🐠 Medaka / Stickleback (teleost fish)": {
                "min_cpm": 1.0, "padj": 0.05, "fc": 1.0, "top_n": 50,
                "note": "~25,000 genes. Standard thresholds, similar to zebrafish."
            },
            "🌊 Chlamydomonas (green alga)": {
                "min_cpm": 0.5, "padj": 0.05, "fc": 1.0, "top_n": 30,
                "note": "~17,000 genes. Algal RNA-seq can be noisy — lower CPM preserves more genes."
            },
            "⚙️ Custom / Other": {
                "min_cpm": 1.0, "padj": 0.05, "fc": 1.0, "top_n": 50,
                "note": "General safe defaults. Adjust based on your organism's genome size and expression range."
            },
        }

        preset_choice = st.selectbox(
            "Select organism",
            list(PRESETS.keys()),
            index=0,
            key="organism_preset"
        )

        preset = PRESETS[preset_choice]

        # Show note if a preset is selected
        if preset:
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:7px;'
                f'padding:8px 10px;margin-bottom:6px;font-size:11px;color:#64748b;">'
                f'💡 {preset["note"]}</div>',
                unsafe_allow_html=True
            )

        # Defaults: from preset if selected, else previous values
        def _pv(key, fallback):
            if preset:
                return preset[key]
            return st.session_state.get(f"manual_{key}", fallback)

        # ── Filter settings ──────────────────────────────────────────────────
        st.divider()
        st.markdown("### ⚙️ CPM Filter")
        st.caption("Removes genes too lowly expressed to be reliable.")

        min_cpm = st.slider(
            "Min CPM  *(counts per million)*",
            0.1, 10.0, _pv("min_cpm", 1.0), 0.1,
            help=(
                "A gene must have at least this many CPM in enough samples to be kept.\n\n"
                "**Higher** = stricter, fewer genes, less noise.\n"
                "**Lower** = keep more lowly-expressed genes.\n\n"
                "Rule of thumb: use 1 for mammals/plants, 2–5 for bacteria/yeast."
            )
        )

        n_samp_total = max(len(st.session_state.samples), 2)
        default_min_samp = min(_pv("min_cpm", 2), n_samp_total)  # reuse as proxy
        min_samples = st.slider(
            "In ≥ N samples",
            1, n_samp_total, min(2, n_samp_total), 1,
            help=(
                "How many samples must pass the CPM threshold for a gene to be kept.\n\n"
                "Set this to the **size of your smallest group** (e.g. 3 if you have 3 replicates per group). "
                "This ensures a gene is expressed in at least one full condition."
            )
        )

        # ── DE thresholds ────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🎯 DE Thresholds")
        st.caption("Controls which genes are called differentially expressed.")

        padj_thresh = st.slider(
            "padj ≤  *(adjusted p-value)*",
            0.001, 0.2, float(_pv("padj", 0.05)), 0.001,
            format="%.3f",
            help=(
                "False Discovery Rate threshold after Benjamini-Hochberg correction.\n\n"
                "**0.05** = standard for publication (5% expected false positives).\n"
                "**0.01** = strict, high-confidence hits.\n"
                "**0.10** = exploratory, finds more candidates but less reliable.\n\n"
                "Always use padj, not raw p-value, for multiple-testing correction."
            )
        )

        fc_thresh = st.slider(
            "|log₂FC| ≥  *(fold change)*",
            0.0, 4.0, float(_pv("fc", 1.0)), 0.1,
            help=(
                "Minimum fold change between groups (in log₂ scale).\n\n"
                "log₂FC = 1  → 2× change\n"
                "log₂FC = 2  → 4× change\n"
                "log₂FC = 0  → any change (padj only)\n\n"
                "**Mammals/plants:** 1.0 is standard.\n"
                "**Bacteria:** 1.5 is better — bacteria naturally show large swings.\n"
                "**Low replicates (n=2):** use 1.5–2.0 to compensate."
            )
        )

        top_n = st.slider(
            "Top N variable genes  *(heatmap)*",
            10, 200, int(_pv("top_n", 50)), 5,
            help=(
                "How many of the most variable genes to show in the expression heatmap.\n\n"
                "**50** is a good default — enough to see patterns without overcrowding.\n"
                "**Bacteria/yeast:** use 20–30 (smaller genomes).\n"
                "**Mammals:** 50–100 is fine.\n\n"
                "This does NOT affect DE results — only the heatmap visualization."
            )
        )

        # ── Run button ───────────────────────────────────────────────────────
        st.divider()
        run = st.button("▶ Run Analysis", use_container_width=True, type="primary")
    else:
        label_a  = st.session_state.compare_a
        label_b  = st.session_state.compare_b
        padj_thresh, fc_thresh, top_n = 0.05, 1.0, 50
        min_cpm, min_samples = 1.0, 2
        run = False

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

st.title("🧬 TranscriptomicViz")
st.caption("Publication-grade RNA-seq analysis · HTSeq + featureCounts · auto-detected")

# ── Landing ──────────────────────────────────────────────────────────────────
if st.session_state.count_df is None:
    st.markdown("""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:28px 32px;max-width:600px;margin:40px auto;">
      <div style="color:#4ade80;font-weight:600;font-size:14px;margin-bottom:12px;">⚙ Publication-grade pipeline</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 20px;color:#64748b;font-size:13px;line-height:2;">
        <span>✓ TMM normalization</span><span>✓ NB exact test</span>
        <span>✓ BH-FDR correction</span><span>✓ LFC shrinkage</span>
        <span>✓ CPM filter + log2-CPM</span><span>✓ Smart warnings</span>
        <span>✓ PCA plot</span><span>✓ Volcano plot</span>
        <span>✓ Heatmaps</span><span>✓ CSV download</span>
      </div>
      <div style="margin-top:16px;color:#475569;font-size:12px;">
        👈 Upload your count files in the sidebar to get started.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
matrix_full = st.session_state.matrix
genes       = st.session_state.genes
samples_all = st.session_state.samples
gmap        = st.session_state.group_map
all_labels  = st.session_state.group_labels

# Filter to only selected samples
selected = st.session_state.get("selected_samples", None)
if selected is None or len(selected) == 0:
    selected = samples_all
active_idx = [samples_all.index(s) for s in selected if s in samples_all]
samples = [samples_all[i] for i in active_idx]
matrix  = matrix_full[:, active_idx] if active_idx else matrix_full
gmap    = {s: gmap[s] for s in samples if s in gmap}

group_a_idx = [samples.index(s) for s, g in gmap.items() if g == label_a and s in samples]
group_b_idx = [samples.index(s) for s, g in gmap.items() if g == label_b and s in samples]

# Show a metric card per group
metric_cols = st.columns(2 + len(all_labels))
metric_cols[0].metric("Total Genes", f"{len(genes):,}")
metric_cols[1].metric("Samples", len(samples))
for i, lbl in enumerate(all_labels):
    n = sum(1 for g in gmap.values() if g == lbl)
    color_idx = i % len(PALETTE)
    metric_cols[2 + i].metric(lbl, n)

st.markdown(
    " · ".join(
        f'<span style="color:{PALETTE[i%len(PALETTE)]};font-weight:600">■ {lbl} '
        f'({sum(1 for g in gmap.values() if g==lbl)})</span>'
        for i, lbl in enumerate(all_labels)
    ),
    unsafe_allow_html=True
)

# ── Warnings ──────────────────────────────────────────────────────────────────
def show_warnings(matrix, genes, group_a, group_b, samples, de_results=None):
    n_a, n_b = len(group_a), len(group_b)
    if n_a == 0 or n_b == 0:
        st.markdown('<div class="error-box">⛔ No samples assigned to one or both groups.</div>', unsafe_allow_html=True)
    elif n_a < 2 or n_b < 2:
        st.markdown(f'<div class="error-box">⛔ Only {min(n_a,n_b)} replicate(s) in one group — statistics unreliable.</div>', unsafe_allow_html=True)
    elif n_a < 3 or n_b < 3:
        st.markdown('<div class="warn-box">⚠ Only 2 replicates in one group. 3+ strongly recommended.</div>', unsafe_allow_html=True)

    if len(group_a) > 0 and len(group_b) > 0:
        all_idx = group_a + group_b
        lib_sizes = matrix[:, all_idx].sum(axis=0)
        if lib_sizes.min() > 0 and lib_sizes.max() / lib_sizes.min() > 5:
            st.markdown(f'<div class="warn-box">⚠ Large library size variation ({lib_sizes.max()/lib_sizes.min():.1f}× diff). Check for outlier samples.</div>', unsafe_allow_html=True)

    if de_results is not None and len(de_results) > 0:
        sig05 = (de_results["padj"] <= 0.05).sum()
        sig10 = (de_results["padj"] <= 0.10).sum()
        if sig05 == 0 and sig10 == 0:
            st.markdown('<div class="warn-box">⚠ No DEGs at padj≤0.10. Consider more replicates or check conditions.</div>', unsafe_allow_html=True)
        elif sig05 == 0:
            st.markdown(f'<div class="info-box">ℹ No DEGs at padj≤0.05, but {sig10} found at padj≤0.10.</div>', unsafe_allow_html=True)
        elif sig05 > len(genes) * 0.5:
            st.markdown(f'<div class="warn-box">⚠ {sig05} DEGs (>{sig05/len(genes)*100:.0f}% of genes) — check for batch effects.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ok-box">✓ {sig05} significant DEGs found at padj≤0.05</div>', unsafe_allow_html=True)

# ── Run analysis ──────────────────────────────────────────────────────────────
if run:
    if len(group_a_idx) == 0 or len(group_b_idx) == 0:
        st.error("Assign at least one sample to each group before running.")
    else:
        with st.spinner("Running DE analysis (TMM → NB test → BH-FDR → LFC shrinkage)..."):
            filt_mat, filt_genes, filt_idx = filter_low_counts(
                matrix, genes, min_cpm=min_cpm, min_samples=min_samples
            )
            de = compute_de(filt_mat, filt_genes, group_a_idx, group_b_idx)
            import datetime
            st.session_state.de_results  = de
            st.session_state.filt_mat    = filt_mat
            st.session_state.filt_genes  = filt_genes
            st.session_state.filt_idx    = filt_idx
            st.session_state.run_meta    = {
                "timestamp":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "group_a":     label_a,
                "group_b":     label_b,
                "n_a":         len(group_a_idx),
                "n_b":         len(group_b_idx),
                "genes_tested":len(filt_genes),
                "genes_total": len(genes),
                "min_cpm":     min_cpm,
                "min_samples": min_samples,
                "padj_thresh": padj_thresh,
                "fc_thresh":   fc_thresh,
                "phi":         None,  # filled after DE
            }
        sig_n = int((de["padj"] <= padj_thresh).sum())
        st.success(f"✅ Analysis complete — {len(de):,} genes tested · {sig_n} significant (padj≤{padj_thresh})")

de_results = st.session_state.de_results

# ── Warnings ─────────────────────────────────────────────────────────────────
show_warnings(matrix, genes, group_a_idx, group_b_idx, samples, de_results)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🔬 QC", "📊 PCA", "🌋 Volcano Plot", "📈 MA Plot", "🔥 Expression Heatmap", "🧪 DEG Heatmap", "📋 DEG Table"])

# ═════════════ QC TAB ════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("🔬 Quality Control")
    st.caption("Library sizes always shown. Run analysis first for p-value distribution, correlation, and audit trail.")

    # Library sizes
    st.markdown("#### 📚 Library Sizes per Sample")
    lib_sizes_list = [(s, int(matrix[:, i].sum()), gmap.get(s, "?")) for i, s in enumerate(samples)]
    lib_df = pd.DataFrame(lib_sizes_list, columns=["Sample", "Total Counts", "Group"])
    lib_df = lib_df.sort_values("Total Counts", ascending=True)
    all_labels_qc = st.session_state.group_labels
    bar_colors = [PALETTE[all_labels_qc.index(g) % len(PALETTE)] if g in all_labels_qc else "#64748b"
                  for g in lib_df["Group"]]
    fig_lib = go.Figure(go.Bar(
        x=lib_df["Total Counts"], y=lib_df["Sample"], orientation="h",
        marker_color=bar_colors,
        text=[f"{v/1e6:.2f}M" for v in lib_df["Total Counts"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b> · %{x:,} reads<extra></extra>",
    ))
    med_lib = lib_df["Total Counts"].median()
    fig_lib.add_vline(x=med_lib, line=dict(color="#ef4444", dash="dash"),
                      annotation_text=f"median {med_lib/1e6:.1f}M",
                      annotation_font=dict(color="#ef4444", size=10))
    fig_lib.update_layout(
        height=max(280, len(samples)*28+80),
        plot_bgcolor="#0f172a", paper_bgcolor="#020817",
        xaxis=dict(title="Total Read Counts", tickfont=dict(color="#64748b"), gridcolor="#1e293b", title_font=dict(color="#94a3b8")),
        yaxis=dict(tickfont=dict(color="#94a3b8")),
        margin=dict(l=10, r=80, t=20, b=40),
    )
    st.plotly_chart(fig_lib, use_container_width=True)
    max_l = lib_df["Total Counts"].max(); min_l = lib_df["Total Counts"].min()
    if min_l > 0 and max_l / min_l > 5:
        st.markdown(f'<div class="warn-box">⚠ Library size range: {min_l:,} – {max_l:,} ({max_l/min_l:.1f}× difference). TMM compensates but check for outliers.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ok-box">✓ Library sizes balanced: {min_l:,} – {max_l:,}</div>', unsafe_allow_html=True)

    # Sample correlation heatmap
    st.markdown("#### 🔗 Sample-to-Sample Correlation (Spearman, log2-CPM)")
    filt_mat_qc = st.session_state.get("filt_mat", None)
    if filt_mat_qc is None:
        filt_mat_qc, _, _ = filter_low_counts(matrix, genes, min_cpm=min_cpm, min_samples=min_samples)
    if filt_mat_qc.shape[0] >= 2:
        lcpm_qc = vst_transform(filt_mat_qc)
        ns = len(samples)
        corr_mat = np.zeros((ns, ns))
        for i in range(ns):
            for j in range(ns):
                r, _ = stats.spearmanr(lcpm_qc[:, i], lcpm_qc[:, j])
                corr_mat[i, j] = float(r)
        fig_corr = go.Figure(go.Heatmap(
            z=corr_mat, x=samples, y=samples,
            colorscale="RdBu", zmin=0.8, zmax=1.0,
            text=np.round(corr_mat, 3), texttemplate="%{text:.3f}",
            textfont=dict(size=9),
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Spearman r = %{z:.3f}<extra></extra>",
            colorbar=dict(tickfont=dict(color="#64748b"), title=dict(text="r", font=dict(color="#64748b"))),
        ))
        fig_corr.update_layout(
            height=max(350, ns*42+60),
            plot_bgcolor="#0f172a", paper_bgcolor="#020817",
            xaxis=dict(tickfont=dict(color="#94a3b8"), tickangle=40),
            yaxis=dict(tickfont=dict(color="#94a3b8")),
            margin=dict(l=10, r=10, t=20, b=80),
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        off_diag = corr_mat[np.triu_indices(ns, k=1)]
        if len(off_diag) > 0:
            min_r = off_diag.min()
            if min_r < 0.85:
                st.markdown(f'<div class="warn-box">⚠ Lowest pairwise correlation: r={min_r:.3f}. Samples below r=0.85 may be outliers.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ok-box">✓ All pairwise correlations ≥ {min_r:.3f}</div>', unsafe_allow_html=True)

    # P-value distribution
    de_qc = st.session_state.de_results
    if de_qc is not None:
        st.markdown("#### 📊 P-value Distribution")
        st.caption("Expected: flat (uniform) baseline with a spike near p=0 for true DEGs. Spikes elsewhere = model problems.")
        pv = de_qc["pval"].values
        fig_pv = go.Figure(go.Histogram(
            x=pv, nbinsx=20, marker_color="#38bdf8", opacity=0.85,
            hovertemplate="p ∈ %{x:.2f}<br>%{y} genes<extra></extra>",
        ))
        exp_per_bin = len(pv) / 20
        fig_pv.add_hline(y=exp_per_bin, line=dict(color="#f97316", dash="dash", width=1.5),
                         annotation_text="uniform expectation", annotation_font=dict(color="#f97316", size=10))
        fig_pv.update_layout(
            height=260, plot_bgcolor="#0f172a", paper_bgcolor="#020817",
            xaxis=dict(title="p-value", tickfont=dict(color="#64748b"), gridcolor="#1e293b", title_font=dict(color="#94a3b8")),
            yaxis=dict(title="Gene count", tickfont=dict(color="#64748b"), gridcolor="#1e293b", title_font=dict(color="#94a3b8")),
            margin=dict(l=10, r=10, t=20, b=50), bargap=0.05,
        )
        st.plotly_chart(fig_pv, use_container_width=True)
        n_high = (pv > 0.5).sum()
        pi0 = min(1.0, 2 * n_high / max(len(pv), 1))
        if pi0 > 0.95:
            st.markdown('<div class="info-box">ℹ Distribution looks uniform — most genes appear non-DE (expected if few true DEGs).</div>', unsafe_allow_html=True)
        elif pi0 < 0.5:
            st.markdown(f'<div class="warn-box">⚠ Estimated π₀ ≈ {pi0:.2f} — unusually high DE rate. Check group assignments and batch effects.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ok-box">✓ p-value distribution healthy. Estimated null fraction π₀ ≈ {pi0:.2f}</div>', unsafe_allow_html=True)

        # Audit trail
        run_meta = st.session_state.get("run_meta", None)
        if run_meta:
            st.markdown("#### 🗂️ Analysis Audit Trail")
            import datetime
            meta_rows = [
                ("Timestamp",        run_meta.get("timestamp", "—")),
                ("Comparison",       f"{run_meta.get('group_b','?')} vs {run_meta.get('group_a','?')}"),
                ("Group A (ref)",    f"{run_meta.get('group_a','?')} · n={run_meta.get('n_a','?')}"),
                ("Group B",          f"{run_meta.get('group_b','?')} · n={run_meta.get('n_b','?')}"),
                ("Genes tested",     f"{run_meta.get('genes_tested',0):,} / {run_meta.get('genes_total',0):,}"),
                ("CPM filter",       f"≥{run_meta.get('min_cpm','?')} in ≥{run_meta.get('min_samples','?')} samples"),
                ("padj threshold",   f"≤{run_meta.get('padj_thresh','?')} (BH-FDR)"),
                ("|log₂FC| ≥",      f"{run_meta.get('fc_thresh','?')}"),
                ("Normalization",    "TMM (Robinson & Oshlack 2010)"),
                ("Statistical test", "NB score test + continuity correction"),
                ("LFC shrinkage",    "Adaptive Bayesian, MAD-based prior"),
                ("Transform",        "log2-CPM (edgeR-style, NOT DESeq2 VST)"),
            ]
            meta_df = pd.DataFrame(meta_rows, columns=["Parameter", "Value"])
            st.dataframe(meta_df, use_container_width=True, hide_index=True, height=400)
    else:
        st.info("Run analysis to unlock p-value distribution and audit trail.")

# ═════════════ PCA ════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("PCA Plot")
    st.caption("CPM-filtered · log2-CPM normalized · top variable genes")

    filt_mat   = st.session_state.get("filt_mat", None)
    filt_genes = st.session_state.get("filt_genes", genes)
    filt_idx   = st.session_state.get("filt_idx", list(range(len(genes))))

    if filt_mat is None:
        # Run filter even without DE so PCA always works
        filt_mat, filt_genes, filt_idx = filter_low_counts(
            matrix, genes, min_cpm=min_cpm, min_samples=min_samples
        )

    if filt_mat.shape[0] < 2 or len(samples) < 2:
        st.warning("Need ≥2 samples and ≥2 genes passing CPM filter.")
    else:
        vst_mat = vst_transform(filt_mat)
        pca = pca_compute(vst_mat, samples)
        if pca:
            fig = make_pca(pca, gmap, label_a, label_b,
                           filter_info={"kept": len(filt_genes),
                                        "removed": len(genes) - len(filt_genes)})
            st.plotly_chart(fig, use_container_width=True)
            mpl_fig = _pca_mpl(pca, gmap, label_a, label_b,
                               filter_info={"kept": len(filt_genes),
                                            "removed": len(genes) - len(filt_genes)})
            st.download_button("⬇ Download PNG", fig_to_bytes(mpl_fig),
                               "pca_plot.png", "image/png")
            plt.close(mpl_fig)
        else:
            st.warning("PCA failed — check your data.")

# ═════════════ VOLCANO ════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Volcano Plot")
    st.caption(f"{label_b} vs {label_a} · NB test · BH-FDR · shrunken LFC")

    if de_results is None:
        st.info("Click **▶ Run Analysis** in the sidebar to generate results.")
    else:
        vc1, vc2, vc3 = st.columns([2, 2, 3])
        use_raw_fc = vc1.toggle("Use raw log₂FC", value=False,
                                help="Shrunken FC is better for publication. Raw FC shows the unmodified values — useful if points look over-compressed.")
        show_gene_labels = vc2.toggle("Label DEG points", value=False,
                                      help="Show gene names on significant points. Turn off if too crowded.")

        fig = make_volcano(de_results, fc_thresh, padj_thresh, label_a, label_b,
                           use_raw_fc=use_raw_fc)

        # Optionally add gene labels for significant points
        if show_gene_labels:
            df_sig = de_results[de_results["padj"] <= padj_thresh].copy()
            fc_col = "log2FC_raw" if use_raw_fc else "log2FC"
            y_raw = de_results["negLog10Padj"].replace([np.inf,-np.inf], np.nan).dropna()
            y_95  = float(y_raw.quantile(0.95))
            y_max = max(y_95 * 1.25, -np.log10(padj_thresh) * 1.5, 10)
            fig.add_trace(go.Scatter(
                x=df_sig[fc_col].clip(lower=-y_max, upper=y_max),
                y=df_sig["negLog10Padj"].clip(upper=y_max),
                mode="text",
                text=df_sig["gene"],
                textposition="top center",
                textfont=dict(size=9, color="#e2e8f0"),
                showlegend=False,
                hoverinfo="skip",
            ))

        st.plotly_chart(fig, use_container_width=True)
        st.caption("▲ triangle = point capped (extreme padj). Hover any dot for details. Use raw log₂FC toggle if points are over-compressed.")
        mpl_fig = _volcano_mpl(de_results, fc_thresh, padj_thresh, label_a, label_b)
        st.download_button("⬇ Download PNG", fig_to_bytes(mpl_fig),
                           "volcano_plot.png", "image/png")
        plt.close(mpl_fig)

# ═════════════ MA PLOT ════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("MA Plot")
    st.caption("Mean expression (A) vs log₂ fold change (M) — standard RNA-seq QC plot")
    st.markdown("An MA plot shows **whether fold changes depend on expression level**. "
                "Low-count genes often show extreme FC — they should cluster near M=0. "
                "After TMM normalization, the cloud should be centred on M=0.")

    if de_results is None:
        st.info("Click **▶ Run Analysis** in the sidebar to generate results.")
    else:
        ma_fc_col = st.toggle("Use raw log₂FC for MA plot", value=True,
                              help="Raw FC is standard for MA plots. Shrunken FC hides the compression effect this plot is meant to diagnose.")

        df_ma = de_results.copy()
        fc_col_ma = "log2FC_raw" if ma_fc_col else "log2FC"

        def cls_ma(row):
            if row["padj"] <= padj_thresh and row[fc_col_ma] >= fc_thresh: return "up"
            if row["padj"] <= padj_thresh and row[fc_col_ma] <= -fc_thresh: return "down"
            return "ns"

        df_ma["cls"] = df_ma.apply(cls_ma, axis=1)
        df_ma["A"] = np.log2(df_ma["baseMean"] + 0.5)

        color_ma = {"up": "#f97316", "down": "#38bdf8", "ns": "#1e3a4a"}
        labels_ma = {
            "up":   f"🟠 Up ({(df_ma.cls=='up').sum()})",
            "down": f"🔵 Down ({(df_ma.cls=='down').sum()})",
            "ns":   f"⚫ NS ({(df_ma.cls=='ns').sum()})",
        }

        fig_ma = go.Figure()
        for c, grp in df_ma.groupby("cls"):
            fig_ma.add_trace(go.Scatter(
                x=grp["A"], y=grp[fc_col_ma],
                mode="markers",
                name=labels_ma[c],
                marker=dict(color=color_ma[c], size=5 if c=="ns" else 7,
                            opacity=0.3 if c=="ns" else 0.85, line=dict(width=0)),
                customdata=np.stack([grp["gene"],
                                     grp[fc_col_ma].round(3),
                                     grp["padj"].apply(lambda v: f"{v:.2e}"),
                                     grp["baseMean"].round(1)], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "A (mean expr): %{x:.2f}<br>"
                    "M (log₂FC):    <b>%{customdata[1]}</b><br>"
                    "padj:          %{customdata[2]}<br>"
                    "Base mean CPM: %{customdata[3]}<extra></extra>"
                ),
                hoverlabel=dict(bgcolor="#0f172a", bordercolor="#334155",
                                font=dict(color="#e2e8f0", size=11, family="monospace")),
            ))

        # LOWESS smoothing line to show trend
        try:
            from scipy.stats import spearmanr
            from scipy.ndimage import uniform_filter1d
            sort_idx = np.argsort(df_ma["A"].values)
            a_sorted = df_ma["A"].values[sort_idx]
            m_sorted = df_ma[fc_col_ma].values[sort_idx]
            window = max(1, len(m_sorted) // 10)
            m_smooth = uniform_filter1d(m_sorted, size=window)
            fig_ma.add_trace(go.Scatter(
                x=a_sorted, y=m_smooth,
                mode="lines", name="Trend (smoothed)",
                line=dict(color="#a78bfa", width=2),
                hoverinfo="skip",
            ))
        except Exception:
            pass

        fig_ma.add_hline(y=0, line=dict(color="#475569", dash="dash", width=1))
        fig_ma.add_hline(y= fc_thresh, line=dict(color="#334155", dash="dot", width=1))
        fig_ma.add_hline(y=-fc_thresh, line=dict(color="#334155", dash="dot", width=1))

        fig_ma.update_layout(
            title=dict(text=f"MA Plot — <b>{label_b}</b> vs <b>{label_a}</b>",
                       font=dict(color="#e2e8f0", size=15)),
            xaxis=dict(title="A = log₂(mean CPM)", tickfont=dict(color="#64748b"),
                       gridcolor="#1e293b", title_font=dict(color="#94a3b8")),
            yaxis=dict(title="M = log₂ Fold Change", tickfont=dict(color="#64748b"),
                       gridcolor="#1e293b", title_font=dict(color="#94a3b8")),
            plot_bgcolor="#0f172a", paper_bgcolor="#020817",
            legend=dict(font=dict(color="#94a3b8", size=11), bgcolor="#0f172a",
                        bordercolor="#334155", borderwidth=1),
            height=520, hovermode="closest",
            margin=dict(t=60, l=60, r=30, b=60),
        )
        st.plotly_chart(fig_ma, use_container_width=True)
        st.caption("Purple line = smoothed trend. Should be flat at M=0 after TMM normalization. "
                   "If the trend is curved, normalization may be insufficient (consider removing outlier samples).")

# ═════════════ EXPRESSION HEATMAP ════════════════════════════════════════════
with tabs[4]:
    st.subheader("Expression Heatmap")
    st.caption(f"Top {top_n} most variable genes · TMM → log2-CPM → z-score")

    filt_mat = st.session_state.get("filt_mat", None)
    if filt_mat is None:
        filt_mat, filt_genes, filt_idx = filter_low_counts(
            matrix, genes, min_cpm=min_cpm, min_samples=min_samples
        )

    if filt_mat.shape[0] == 0:
        st.warning("No genes passed CPM filter. Try lowering Min CPM.")
    else:
        vst_mat   = vst_transform(filt_mat)
        gene_vars = vst_mat.var(axis=1)
        top_local = np.argsort(gene_vars)[::-1][:top_n]
        top_global = [filt_idx[i] for i in top_local]

        fig = make_heatmap(matrix, genes, samples, list(top_global),
                           "Expression Heatmap",
                           f"Top {top_n} variable genes · CPM≥{min_cpm}")
        if fig:
            st.pyplot(fig, use_container_width=False)
            st.download_button("⬇ Download PNG", fig_to_bytes(fig),
                               "heatmap_top_variable.png", "image/png")
            plt.close(fig)

# ═════════════ DEG HEATMAP ════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("DEG Heatmap")

    if de_results is None:
        st.info("Click **▶ Run Analysis** in the sidebar to generate results.")
    else:
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        up_genes = de_results[
            (de_results["padj"] <= padj_thresh) & (de_results["log2FC"] >= fc_thresh)
        ].sort_values("log2FC", ascending=False)["gene"].tolist()
        dn_genes = de_results[
            (de_results["padj"] <= padj_thresh) & (de_results["log2FC"] <= -fc_thresh)
        ].sort_values("log2FC")["gene"].tolist()

        display = st.radio("Show", ["All DEGs", "↑ Upregulated only", "↓ Downregulated only"],
                           horizontal=True)

        if display == "↑ Upregulated only":
            idx_list = [gene_to_idx[g] for g in up_genes if g in gene_to_idx]
            title, subtitle = "DEG Heatmap — Upregulated", f"{len(idx_list)} genes · log₂FC ≥ {fc_thresh}"
        elif display == "↓ Downregulated only":
            idx_list = [gene_to_idx[g] for g in dn_genes if g in gene_to_idx]
            title, subtitle = "DEG Heatmap — Downregulated", f"{len(idx_list)} genes · log₂FC ≤ -{fc_thresh}"
        else:
            idx_list = [gene_to_idx[g] for g in (up_genes + dn_genes) if g in gene_to_idx]
            title, subtitle = "DEG Heatmap — All DEGs", f"{len(idx_list)} DEGs · padj ≤ {padj_thresh}"

        if not idx_list:
            st.warning("No DEGs found with current thresholds. Try relaxing padj or FC.")
        else:
            fig = make_heatmap(matrix, genes, samples, idx_list, title, subtitle)
            if fig:
                st.pyplot(fig, use_container_width=False)
                st.download_button("⬇ Download PNG", fig_to_bytes(fig),
                                   "heatmap_degs.png", "image/png")
                plt.close(fig)

# ═════════════ DEG TABLE ═════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("DEG Results Table")
    st.caption(f"{label_b} vs {label_a} · sortable · searchable · CSV export")

    if de_results is None:
        st.info("Click **▶ Run Analysis** in the sidebar to generate results.")
    else:
        def regulation(row):
            if row["padj"] <= padj_thresh and row["log2FC"] >= fc_thresh:
                return "↑ UP"
            if row["padj"] <= padj_thresh and row["log2FC"] <= -fc_thresh:
                return "↓ DOWN"
            return "NS"

        display_df = de_results.copy()
        display_df["regulation"] = display_df.apply(regulation, axis=1)
        display_df = display_df[["gene", "baseMean", "log2FC", "log2FC_raw",
                                  "pval", "padj", "regulation"]]
        display_df.columns = ["Gene", "Base Mean", "log₂FC (shrunk)",
                               "log₂FC (raw)", "p-value", "padj (BH)", "Regulation"]

        # Filter controls
        c1, c2, c3 = st.columns([2, 1, 1])
        search  = c1.text_input("🔍 Search gene", "")
        flt_opt = c2.selectbox("Filter", ["All", "Significant", "↑ Up", "↓ Down"])
        show_n  = c3.number_input("Rows per page", 10, 200, 20, 10)

        if search:
            display_df = display_df[display_df["Gene"].str.contains(search, case=False)]
        if flt_opt == "Significant":
            display_df = display_df[display_df["Regulation"] != "NS"]
        elif flt_opt == "↑ Up":
            display_df = display_df[display_df["Regulation"] == "↑ UP"]
        elif flt_opt == "↓ Down":
            display_df = display_df[display_df["Regulation"] == "↓ DOWN"]

        st.caption(f"Showing {min(show_n, len(display_df))} of {len(display_df)} rows")
        st.dataframe(
            display_df.head(show_n).style
                .applymap(lambda v: "color: #f97316; font-weight:600" if v == "↑ UP"
                          else ("color: #38bdf8; font-weight:600" if v == "↓ DOWN"
                                else "color: #475569"), subset=["Regulation"])
                .format({"Base Mean": "{:.1f}", "log₂FC (shrunk)": "{:.3f}",
                         "log₂FC (raw)": "{:.3f}", "p-value": "{:.2e}", "padj (BH)": "{:.2e}"}),
            use_container_width=True, height=420
        )

        # CSV download with metadata header
        import datetime
        run_meta = st.session_state.get("run_meta", {})
        csv_buf = io.StringIO()
        csv_buf.write(f"# TranscriptomicViz — DE Analysis Results\n")
        csv_buf.write(f"# Generated: {run_meta.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n")
        csv_buf.write(f"# Comparison: {label_b} vs {label_a}\n")
        csv_buf.write(f"# Group A (reference): {label_a} · n={len(group_a_idx)} samples\n")
        csv_buf.write(f"# Group B: {label_b} · n={len(group_b_idx)} samples\n")
        csv_buf.write(f"# Genes tested: {len(de_results):,} (after CPM≥{min_cpm} filter in ≥{min_samples} samples)\n")
        csv_buf.write(f"# Normalization: TMM (Robinson & Oshlack 2010, Genome Biology)\n")
        csv_buf.write(f"# Test: Negative Binomial score test with continuity correction\n")
        csv_buf.write(f"# Multiple testing: Benjamini-Hochberg FDR\n")
        csv_buf.write(f"# LFC shrinkage: Adaptive Bayesian (MAD-based prior)\n")
        csv_buf.write(f"# log2FC column: shrunken LFC (use log2FC_raw for MA plots)\n")
        csv_buf.write(f"#\n")
        de_results.to_csv(csv_buf, index=False, float_format="%.6g")
        st.download_button(
            "⬇ Download Full CSV (with methods header)",
            csv_buf.getvalue().encode(),
            f"DEG_{label_b}_vs_{label_a}.csv",
            "text/csv",
            use_container_width=True,
        )
