"""Reproducible figures for the Meridian README and report.

Usage:
    python figures/make_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
DPI = 180


# ── fig1: Architecture ───────────────────────────────────────────────────

def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")

    # Colors
    pipe_bg = "#161b22"
    pipe_box = "#21262d"
    pipe_text = "#e6edf3"
    pipe_border = "#30363d"
    meas_bg = "#0d2137"
    meas_box_l1 = "#122d4f"
    meas_box_l2 = "#1a1b2f"
    meas_border = "#2f5496"
    accent = "#58a6ff"
    arrow_color = "#8b949e"
    observe_color = "#f0883e"

    # Pipeline background
    pipe_rect = FancyBboxPatch((0.3, 3.2), 5.8, 9.4,
                                boxstyle="round,pad=0.15",
                                facecolor=pipe_bg, edgecolor=pipe_border, linewidth=1.2)
    ax.add_patch(pipe_rect)
    ax.text(3.2, 12.35, "RETRIEVAL  PIPELINE", fontsize=9, fontweight="bold",
            color=accent, ha="center", fontfamily="monospace")

    # Pipeline boxes
    phases = [
        ("Ingestion", "Chunking + SAC summary", 11.5),
        ("Indexing", "Dense (voyage-4) + BM25", 10.5),
        ("Routing", "Document top-k filter", 9.5),
        ("Retrieval", "Dense + Sparse channels", 8.5),
        ("CC Fusion", "Score-weighted merge", 7.5),
        ("Selector", "LLM chunk promotion", 6.5),
        ("Synthesis", "LLM answer + citations", 5.5),
        ("Verification", "Deterministic citation check", 4.5),
    ]

    box_w, box_h = 4.8, 0.65
    box_x = 0.8

    for name, desc, y in phases:
        rect = FancyBboxPatch((box_x, y), box_w, box_h,
                               boxstyle="round,pad=0.08",
                               facecolor=pipe_box, edgecolor=pipe_border, linewidth=0.8)
        ax.add_patch(rect)
        ax.text(box_x + 0.2, y + 0.4, name, fontsize=8.5, fontweight="bold",
                color=pipe_text, va="center")
        ax.text(box_x + box_w - 0.15, y + 0.4, desc, fontsize=6.5,
                color="#8b949e", ha="right", va="center")

    # Arrows between pipeline boxes
    for i in range(len(phases) - 1):
        y_from = phases[i][2]
        y_to = phases[i + 1][2] + box_h
        ax.annotate("", xy=(box_x + box_w / 2, y_to),
                     xytext=(box_x + box_w / 2, y_from),
                     arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=1.0))

    # Measurement background
    meas_rect = FancyBboxPatch((0.3, 0.2), 9.4, 2.7,
                                boxstyle="round,pad=0.15",
                                facecolor=meas_bg, edgecolor=meas_border, linewidth=1.5)
    ax.add_patch(meas_rect)
    ax.text(5.0, 2.65, "FORENSIC  MEASUREMENT  LAYER", fontsize=9, fontweight="bold",
            color=observe_color, ha="center", fontfamily="monospace")

    # Layer 1 box
    l1_rect = FancyBboxPatch((0.6, 0.4), 4.2, 2.0,
                              boxstyle="round,pad=0.1",
                              facecolor=meas_box_l1, edgecolor=meas_border, linewidth=0.8)
    ax.add_patch(l1_rect)
    ax.text(2.7, 2.1, "Layer 1 — Deterministic", fontsize=7.5, fontweight="bold",
            color="#e6edf3", ha="center")
    ax.text(2.7, 1.7, "Span taxonomy:", fontsize=6.5, color="#c9d1d9", ha="center")
    ax.text(2.7, 1.35, "DRM / CBF / SGP / ICR / OVR / OK", fontsize=6.5, fontweight="bold",
            color=accent, ha="center", fontfamily="monospace")
    ax.text(2.7, 0.95, "P@k, R@k (character overlap)", fontsize=6.5, color="#c9d1d9", ha="center")
    ax.text(2.7, 0.6, "No LLM judges", fontsize=6.5, fontstyle="italic", color="#8b949e", ha="center")

    # Layer 2 box
    l2_rect = FancyBboxPatch((5.2, 0.4), 4.2, 2.0,
                              boxstyle="round,pad=0.1",
                              facecolor=meas_box_l2, edgecolor=meas_border, linewidth=0.8)
    ax.add_patch(l2_rect)
    ax.text(7.3, 2.1, "Layer 2 — LLM-Judged", fontsize=7.5, fontweight="bold",
            color="#e6edf3", ha="center")
    ax.text(7.3, 1.7, "Correctness (span-informed)", fontsize=6.5, color="#c9d1d9", ha="center")
    ax.text(7.3, 1.35, "Faithfulness (holistic)", fontsize=6.5, color="#c9d1d9", ha="center")
    ax.text(7.3, 0.95, "Pinned model, separate", fontsize=6.5, color="#c9d1d9", ha="center")
    ax.text(7.3, 0.6, "Never contaminates Layer 1", fontsize=6.5, fontstyle="italic",
            color="#8b949e", ha="center")

    # Observe arrows — two separate, from different pipeline stages
    # Arrow 1: Retrieval/Fusion -> Layer 1
    fusion_y = 7.5  # CC Fusion box y
    ax.annotate("",
                xy=(2.7, 2.45),
                xytext=(box_x + box_w + 0.15, fusion_y + box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=observe_color, lw=1.5,
                                connectionstyle="arc3,rad=0.25", linestyle="--"))
    ax.text(6.7, 4.8, "observes\nretrieval", fontsize=5.5, color=observe_color,
            ha="center", fontstyle="italic", rotation=-50)

    # Arrow 2: Synthesis -> Layer 2
    synth_y = 5.5  # Synthesis box y
    ax.annotate("",
                xy=(7.3, 2.45),
                xytext=(box_x + box_w + 0.15, synth_y + box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=observe_color, lw=1.5,
                                connectionstyle="arc3,rad=0.15", linestyle="--"))
    ax.text(7.8, 3.9, "observes\nanswers", fontsize=5.5, color=observe_color,
            ha="center", fontstyle="italic", rotation=-55)

    plt.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig1_architecture.png", dpi=DPI, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close()
    print("  fig1_architecture.png")


# ── fig2: Config-stack delta ─────────────────────────────────────────────

def fig2_config_delta():
    corpora = ["ContractNLI", "PrivacyQA", "CUAD", "MAUD", "Average"]
    arm0_correct = [62.9, 49.0, 61.9, 68.0, 60.5]
    arm1_correct = [71.1, 55.7, 73.7, 72.7, 68.3]
    arm0_faith = [89.1, 94.4, 92.1, 91.3, 91.7]
    arm1_faith = [94.1, 97.9, 96.0, 95.5, 95.9]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#0d1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    x = np.arange(len(corpora))
    w = 0.35

    # Correctness
    bars0 = ax1.bar(x - w/2, arm0_correct, w, label="Arm 0 (RRF baseline)",
                     color="#21262d", edgecolor="#30363d", linewidth=0.5)
    bars1 = ax1.bar(x + w/2, arm1_correct, w, label="Arm 1 (best config)",
                     color="#58a6ff", edgecolor="#388bfd", linewidth=0.5)
    for i, (a0, a1) in enumerate(zip(arm0_correct, arm1_correct)):
        delta = a1 - a0
        ax1.text(i + w/2, a1 + 0.8, f"+{delta:.1f}", ha="center", fontsize=7,
                 color="#58a6ff", fontweight="bold")
    ax1.set_ylabel("Correctness (%)", color="#e6edf3", fontsize=9)
    ax1.set_title("Answer Correctness", color="#e6edf3", fontsize=10, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(corpora, fontsize=7.5, color="#c9d1d9", rotation=15, ha="right")
    ax1.set_ylim(40, 82)
    ax1.legend(fontsize=7, loc="upper left", facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9")
    ax1.yaxis.label.set_color("#e6edf3")

    # Faithfulness
    bars0f = ax2.bar(x - w/2, arm0_faith, w, label="Arm 0 (RRF baseline)",
                      color="#21262d", edgecolor="#30363d", linewidth=0.5)
    bars1f = ax2.bar(x + w/2, arm1_faith, w, label="Arm 1 (best config)",
                      color="#3fb950", edgecolor="#2ea043", linewidth=0.5)
    for i, (a0, a1) in enumerate(zip(arm0_faith, arm1_faith)):
        delta = a1 - a0
        ax2.text(i + w/2, a1 + 0.3, f"+{delta:.1f}", ha="center", fontsize=7,
                 color="#3fb950", fontweight="bold")
    ax2.set_ylabel("Faithfulness (%)", color="#e6edf3", fontsize=9)
    ax2.set_title("Faithfulness (Holistic)", color="#e6edf3", fontsize=10, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(corpora, fontsize=7.5, color="#c9d1d9", rotation=15, ha="right")
    ax2.set_ylim(85, 100)
    ax2.legend(fontsize=7, loc="upper left", facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9")

    fig.suptitle("Config-stack delta: +7.9pp correctness / +4.2pp faithfulness (controlled, same index)",
                 color="#e6edf3", fontsize=9.5, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig2_config_delta.png", dpi=DPI, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close()
    print("  fig2_config_delta.png")


# ── fig3: Routing boundary ──────────────────────────────────────────────

def fig3_routing_boundary():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    corpora = ["CUAD", "MAUD", "PrivacyQA", "ContractNLI", "NFCorpus\n(medical)"]
    recall = [100, 100, 89, 76, 0]  # 0 = routing OFF (hurts)
    colors = ["#3fb950", "#3fb950", "#e3b341", "#f85149", "#8b949e"]
    labels_right = [
        "distinctive docs",
        "distinctive docs",
        "moderate confusion",
        "NDA / contract confusion",
        "routing OFF wins\n(dispersed relevance)",
    ]

    bars = ax.barh(range(len(corpora)), recall, color=colors, edgecolor="#30363d",
                    linewidth=0.5, height=0.6)

    ax.set_yticks(range(len(corpora)))
    ax.set_yticklabels(corpora, fontsize=9, color="#e6edf3", fontweight="bold")
    ax.set_xlabel("Routing recall (%)", color="#e6edf3", fontsize=9)
    ax.set_xlim(0, 115)
    ax.invert_yaxis()

    for i, (r, label) in enumerate(zip(recall, labels_right)):
        x_pos = r + 2 if r > 0 else 2
        ax.text(x_pos, i, f"{r}%" if r > 0 else "OFF", fontsize=8.5,
                color=colors[i], fontweight="bold", va="center")
        ax.text(x_pos + 12 if r > 0 else 12, i, label, fontsize=7,
                color="#8b949e", va="center", fontstyle="italic")

    ax.set_title("Routing benefit tracks document-discrimination difficulty\n"
                 "(Finding 23 → Finding 45 → Finding 47: concentrated vs dispersed relevance)",
                 color="#e6edf3", fontsize=9, fontweight="bold", pad=12)

    # Annotation arrow for the boundary
    ax.annotate("", xy=(50, 3.8), xytext=(50, 1.2),
                arrowprops=dict(arrowstyle="<->", color="#f0883e", lw=1.5))
    ax.text(53, 2.5, "routing\nboundary", fontsize=7, color="#f0883e",
            fontstyle="italic", va="center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.tick_params(colors="#8b949e")

    plt.tight_layout()
    fig.savefig(OUT / "fig3_routing_boundary.png", dpi=DPI, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close()
    print("  fig3_routing_boundary.png")


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...")
    fig1_architecture()
    fig2_config_delta()
    fig3_routing_boundary()
    print("Done.")
