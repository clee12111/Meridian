"""Generate Meridian_Autopsy.pdf — the 5-page condensed record.

Usage:
    python figures/make_autopsy_pdf.py
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from pathlib import Path

OUT = Path(__file__).parent.parent / "Meridian_Autopsy.pdf"
FIGURES = Path(__file__).parent

# Colors
DARK = HexColor("#0d1117")
MED = HexColor("#161b22")
BORDER = HexColor("#30363d")
TEXT = HexColor("#1a1a1a")
ACCENT = HexColor("#0969da")
LIGHT_GRAY = HexColor("#656d76")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    "Title2", parent=styles["Title"], fontSize=18, spaceAfter=6,
    textColor=TEXT,
))
styles.add(ParagraphStyle(
    "Heading2b", parent=styles["Heading2"], fontSize=13, spaceAfter=4,
    spaceBefore=12, textColor=TEXT,
))
styles.add(ParagraphStyle(
    "Heading3b", parent=styles["Heading3"], fontSize=11, spaceAfter=3,
    spaceBefore=8, textColor=TEXT,
))
styles.add(ParagraphStyle(
    "Body2", parent=styles["BodyText"], fontSize=9, leading=12,
    spaceAfter=4, textColor=TEXT,
))
styles.add(ParagraphStyle(
    "Caption", parent=styles["BodyText"], fontSize=8, leading=10,
    textColor=LIGHT_GRAY, spaceAfter=8, alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    "BulletBody", parent=styles["BodyText"], fontSize=9, leading=12,
    leftIndent=18, bulletIndent=6, spaceAfter=3, textColor=TEXT,
))


def build():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.6*inch, bottomMargin=0.6*inch,
    )

    story = []
    W = doc.width

    # ── Page 1: Title + Thesis + Headline ─────────────────────
    story.append(Paragraph("Meridian — Condensed Autopsy", styles["Title2"]))
    story.append(Paragraph(
        "<i>A forensic measurement framework for RAG systems. "
        "The measurement layer is the contribution.</i>",
        styles["Body2"],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("The thesis", styles["Heading2b"]))
    story.append(Paragraph(
        "RAG evaluation needs a deterministic trust anchor — a measurement layer that "
        "uses no LLM judges in its core computation, so that when a number moves, you "
        "know whether the pipeline changed or the measurement changed. Meridian's Layer 1 "
        "classifies every retrieved span against ground-truth character offsets using pure "
        "arithmetic. Layer 2 (LLM-judged correctness and faithfulness) sits on top, "
        "separate and labeled. The layers never contaminate each other.",
        styles["Body2"],
    ))
    story.append(Paragraph(
        "<b>The proof it works:</b> four silent measurement bugs were caught by the "
        "framework's own verify-before-trust discipline before they could ship wrong "
        "numbers. Each was caught by systematic verification — not by the numbers "
        "looking wrong.",
        styles["Body2"],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Headline numbers", styles["Heading2b"]))
    headline_data = [
        ["Metric", "Value", "Type"],
        ["Config-stack delta (correctness)", "+7.9pp (60.5% → 68.3%)", "Controlled"],
        ["Config-stack delta (faithfulness)", "+4.2pp (91.7% → 95.9%)", "Controlled"],
        ["Benchmark regime", "4 corpora × 194 queries, combined index", "—"],
        ["Ruler calibration", "3/4 corpora within ~2-3pp drift", "Validated"],
        ["Measurement bugs caught", "4 (before shipping)", "Methodology"],
        ["Findings + corrections", "47 findings, 7 corrections", "Evidence trail"],
    ]
    t = Table(headline_data, colWidths=[W*0.38, W*0.40, W*0.22])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f6f8fa")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Architecture figure
    story.append(Paragraph("Architecture", styles["Heading2b"]))
    arch_img = str(FIGURES / "fig1_architecture.png")
    story.append(Image(arch_img, width=W*0.55, height=W*0.55*9/7))
    story.append(Paragraph(
        "Pipeline spine (top) observed by the measurement layer (bottom). "
        "Layer 1 observes retrieval output; Layer 2 observes answers. "
        "Arrows are one-way — measurement observes, never interferes.",
        styles["Caption"],
    ))

    # ── Page 2: The Four Caught Bugs ──────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("The four caught bugs", styles["Heading2b"]))
    story.append(Paragraph(
        "Each was caught by verification checks, not by the numbers looking wrong. "
        "A wrong number from a silent bug looks exactly like a correct number from "
        "a real measurement — the only defense is checking the instrument.",
        styles["Body2"],
    ))
    story.append(Spacer(1, 4))

    bugs = [
        ("<b>1. Pro model-ID alias (Finding 34).</b> PRO_MODEL=\"deepseek-chat\" "
         "is a legacy DeepSeek alias that routes to flash, not Pro. Every prior "
         "\"Pro\" test was actually flash-vs-flash. Caught by billing check "
         "($0.00 on Pro line) + API docs. Fixed: model ID corrected to "
         "\"deepseek-v4-pro\", served-model logging added to every synthesis call."),
        ("<b>2. BM25 channel mismatch (Finding 45).</b> Combined-index dense retrieval "
         "searched 4,496 mini-doc chunks; BM25 searched the full 96,256-chunk CUAD "
         "parquet. Two channels measuring different pools, producing a hybrid that was "
         "neither combined nor per-corpus. Caught by crash on CUAD + post-mortem. "
         "Fixed: BM25 filtered to the same mini-doc set."),
        ("<b>3. Zero-span offset bug (Finding 45).</b> Qdrant payloads don't store "
         "character spans (spans live in parquets). The combined-index builder defaulted "
         "to (0,0). All P@k/R@8 computed as near-zero — a plausible result, not an "
         "obvious crash. CUAD showed R@8=0.049 with 100% routing recall. Caught by "
         "sanity-check against calibration. Fixed: spans from parquets, recomputed. "
         "Real CUAD R@8=0.814."),
        ("<b>4. Chunk-size inconsistency (Finding 46).</b> MAUD uses 2048-char chunks; "
         "the other three corpora use 512-char. Previously undocumented. Confounds "
         "MAUD's external P@k comparison with the paper's 500-char RCTS. Caught during "
         "external-baseline verification."),
    ]
    for b in bugs:
        story.append(Paragraph(b, styles["BulletBody"], bulletText="•"))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "This is the project's thesis in practice: measurement rigor requires "
        "systematic verification at every step. The measurement layer's value "
        "is not just the taxonomy — it's the discipline of checking the instrument "
        "before trusting the reading.",
        styles["Body2"],
    ))

    # ── Page 3: Results ───────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Results", styles["Heading2b"]))

    story.append(Paragraph("Config-stack delta (controlled)", styles["Heading3b"]))
    story.append(Image(str(FIGURES / "fig2_config_delta.png"),
                        width=W*0.92, height=W*0.92*4.5/10))
    story.append(Paragraph(
        "Arm 0 (RRF baseline) vs Arm 1 (CC + routing + selector), same combined "
        "index. The delta isolates the method contribution.",
        styles["Caption"],
    ))

    results_data = [
        ["Corpus", "Correct Arm 0→1", "Faith Arm 0→1", "P@1 Arm 1", "R@8 Arm 1"],
        ["ContractNLI", "62.9→71.1 (+8.2)", "89.1→94.1 (+5.0)", "0.422", "0.810"],
        ["PrivacyQA", "49.0→55.7 (+6.7)", "94.4→97.9 (+3.5)", "0.297", "0.579"],
        ["CUAD", "61.9→73.7 (+11.8)", "92.1→96.0 (+3.9)", "0.394", "0.814"],
        ["MAUD", "68.0→72.7 (+4.7)", "91.3→95.5 (+4.2)", "0.270", "0.783"],
        ["Average", "60.5→68.3 (+7.9)", "91.7→95.9 (+4.2)", "—", "—"],
    ]
    t2 = Table(results_data, colWidths=[W*0.18, W*0.24, W*0.22, W*0.16, W*0.16])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f6f8fa")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t2)
    story.append(Spacer(1, 6))

    story.append(Paragraph("External comparison (system-vs-system)", styles["Heading3b"]))
    story.append(Paragraph(
        "vs arXiv 2408.10343 Table 5 (RCTS, text-embedding-3-large, dense-only). "
        "3 un-confounded corpora (512-char ≈ paper's 500-char). Full stack vs bare "
        "baseline — advantage bundles embedder + hybrid + fusion + routing. "
        "MAUD excluded (2048-char confound). ContractNLI caveated (benchmark provenance).",
        styles["Body2"],
    ))

    ext_data = [
        ["Corpus", "Meridian P@1", "RCTS P@1", "Meridian R@8", "RCTS R@8"],
        ["ContractNLI", "0.422", "0.066", "0.810", "0.250"],
        ["CUAD", "0.394", "0.020", "0.814", "0.317"],
        ["PrivacyQA", "0.297", "0.144", "0.579", "0.424"],
    ]
    t3 = Table(ext_data, colWidths=[W*0.22, W*0.18, W*0.18, W*0.20, W*0.18])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f6f8fa")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t3)

    # ── Page 4: Routing boundary + NFCorpus ───────────────────
    story.append(PageBreak())
    story.append(Paragraph("Routing boundary", styles["Heading2b"]))
    story.append(Image(str(FIGURES / "fig3_routing_boundary.png"),
                        width=W*0.88, height=W*0.88*4.5/7))
    story.append(Paragraph(
        "Routing benefit tracks document-discrimination difficulty. "
        "100% on distinctive corpora (CUAD, MAUD), 76% on homogeneous "
        "(ContractNLI), OFF on dispersed-relevance medical text (NFCorpus).",
        styles["Caption"],
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("NFCorpus transfer (retrieval stack only)", styles["Heading3b"]))
    story.append(Paragraph(
        "nDCG@10 = 0.399, above classic BEIR baselines (BM25 0.325, BM25+CE 0.350, "
        "contriever 0.328). CC fusion transfers (+5.6pp over RRF). The span-forensic "
        "framework was NOT exercised (BEIR has no character spans). Routing correctly "
        "self-disabled. The genuine finding: routing is a concentrated-relevance "
        "technique — the sweep auto-detects when it helps vs hurts.",
        styles["Body2"],
    ))

    # ── Page 5: Limitations + What this is not ────────────────
    story.append(PageBreak())
    story.append(Paragraph("Limitations", styles["Heading2b"]))

    limitations = [
        "<b>Routing degrades to 76%</b> on ContractNLI in the combined regime "
        "(NDA/commercial-contract confusion). 47/194 queries get zero "
        "correct-document chunks.",
        "<b>Routing hurts on dispersed relevance.</b> Confirmed on NFCorpus. "
        "Routing is concentrated-relevance only.",
        "<b>Span framework requires character-span ground truth.</b> Does not "
        "apply to document-level benchmarks (most of BEIR). This is a scope boundary.",
        "<b>MAUD external comparison confounded</b> by 2048-char vs 500-char "
        "chunk granularity.",
        "<b>Affirmative-only evaluation.</b> Correctness measures recall, "
        "not false-positive rate.",
        "<b>Synthesis variance band ±2-4pp.</b> Deltas below this are "
        "indistinguishable from nondeterminism.",
        "<b>External multipliers are system-level,</b> not method-level. "
        "The advantage bundles embedder + hybrid + fusion + routing.",
    ]
    for lim in limitations:
        story.append(Paragraph(lim, styles["BulletBody"], bulletText="•"))

    story.append(Spacer(1, 12))
    story.append(Paragraph("What Meridian is not", styles["Heading2b"]))
    story.append(Paragraph(
        "Not an autonomous research agent (abandoned in v1). "
        "Not a SOTA-everywhere claim (beats classic baselines, not modern dense SOTA). "
        "Not a framework-transfer claim (span taxonomy ran on LegalBench-RAG only; "
        "NFCorpus was retrieval-transfer). "
        "Not a tutorial.",
        styles["Body2"],
    ))
    story.append(Spacer(1, 12))

    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<i>47 findings with 7 corrections — the full evidence trail is in "
        "docs/DECISIONS.md. The complete methodology and results are in "
        "docs/REPORT.md.</i>",
        styles["Caption"],
    ))

    doc.build(story)
    print(f"  {OUT}")


if __name__ == "__main__":
    print("Generating Autopsy PDF...")
    build()
    print("Done.")
