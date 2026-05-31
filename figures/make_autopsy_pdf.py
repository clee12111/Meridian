"""Generate Meridian_Autopsy.pdf — image-free, table-driven, engineering-register.

The four caught bugs are the centerpiece, written as a post-mortem.

Usage:
    python figures/make_autopsy_pdf.py
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT
from pathlib import Path

OUT = Path(__file__).parent.parent / "Meridian_Autopsy.pdf"

BORDER = HexColor("#cccccc")
HEADER_BG = HexColor("#f0f0f0")
TEXT = HexColor("#1a1a1a")
LIGHT = HexColor("#555555")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("T", parent=styles["Title"], fontSize=16, spaceAfter=4, textColor=TEXT))
styles.add(ParagraphStyle("Sub", parent=styles["Normal"], fontSize=9, textColor=LIGHT, spaceAfter=10))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceAfter=4, spaceBefore=14, textColor=TEXT))
styles.add(ParagraphStyle("H3", parent=styles["Heading3"], fontSize=10, spaceAfter=3, spaceBefore=10, textColor=TEXT))
styles.add(ParagraphStyle("B", parent=styles["BodyText"], fontSize=9, leading=12, spaceAfter=4, textColor=TEXT))
styles.add(ParagraphStyle("Bul", parent=styles["BodyText"], fontSize=9, leading=12,
                           leftIndent=14, bulletIndent=4, spaceAfter=3, textColor=TEXT))
styles.add(ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8, leading=10, textColor=LIGHT, spaceAfter=6))


def make_table(data, col_widths, header_row=True):
    t = Table(data, colWidths=col_widths)
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header_row:
        style.append(("BACKGROUND", (0, 0), (-1, 0), HEADER_BG))
        style.append(("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


def build():
    doc = SimpleDocTemplate(str(OUT), pagesize=letter,
                            leftMargin=0.7*inch, rightMargin=0.7*inch,
                            topMargin=0.55*inch, bottomMargin=0.55*inch)
    story = []
    W = doc.width

    # ── PAGE 1: Thesis + Headline ─────────────────────────────
    story.append(Paragraph("Meridian — Condensed Autopsy", styles["T"]))
    story.append(Paragraph("A forensic measurement framework for RAG. "
                            "The measurement layer is the contribution.", styles["Sub"]))

    story.append(Paragraph("Thesis", styles["H2"]))
    story.append(Paragraph(
        "RAG evaluation needs a deterministic trust anchor. LLM judges drift with "
        "model updates, temperature, and prompt wording. Meridian's Layer 1 classifies "
        "every retrieved span against ground-truth character offsets using pure set "
        "intersection — no embeddings, no model calls. Layer 2 (LLM-judged correctness "
        "and faithfulness) sits on top, pinned and separate. The layers never "
        "contaminate each other. When a number moves, the source is identifiable.", styles["B"]))
    story.append(Paragraph(
        "The proof: four silent measurement bugs were caught by the framework's "
        "verify-before-trust discipline before they shipped wrong numbers. Each was "
        "caught by instrument verification, not by the output looking wrong.", styles["B"]))

    story.append(Paragraph("Headline numbers", styles["H2"]))
    story.append(make_table([
        ["Metric", "Value", "Claim type"],
        ["Config-stack delta (correctness)", "+7.9pp (60.5% \u2192 68.3%)", "Controlled (same index)"],
        ["Config-stack delta (faithfulness)", "+4.2pp (91.7% \u2192 95.9%)", "Controlled (same index)"],
        ["Benchmark regime", "4 corpora \u00d7 194q, 11,524-chunk combined index", "\u2014"],
        ["Ruler calibration", "3/4 corpora within ~2-3pp embedding-drift floor", "Validated (Finding 44)"],
        ["Measurement bugs caught", "4 (each before shipping)", "Methodology"],
        ["Evidence trail", "47 findings, 7 corrections", "In docs/DECISIONS.md"],
    ], [W*0.36, W*0.40, W*0.24]))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Measurement-layer design", styles["H2"]))
    story.append(Paragraph(
        "<b>Layer 1 (deterministic, no LLM):</b> Six failure types — DRM (wrong document), "
        "CBF (chunk boundary), SGP (span gap), ICR (wrong section), OVR (over-retrieval), "
        "OK (correct). Classification: <font face='Courier' size=8>gt_chars &amp; retrieved_chars</font> "
        "(character-set intersection). Per-span scoring (not merged-character-set) to handle "
        "multi-span evidence (43% of queries). P@k, R@k are character-overlap ratios.", styles["B"]))
    story.append(Paragraph(
        "<b>Layer 2 (LLM-judged, separate):</b> Correctness (span-informed: judge sees answer + "
        "golden evidence) and faithfulness (holistic groundedness: each claim vs full retrieved context, "
        "RAGAS definition). Pinned model (DeepSeek-v4-flash, T=0, thinking disabled). Never "
        "contaminates Layer 1.", styles["B"]))
    story.append(Paragraph(
        "<b>Calibration:</b> Paper's exact stack replicated (RCTS 500-char, text-embedding-3-large, "
        "dense-only, sqlite-vec). 3/4 corpora reproduce within ~2-3pp (embedding drift). ContractNLI "
        "diverges (benchmark-file provenance). Ruler confirmed (Finding 44).", styles["B"]))

    # ── PAGE 2-3: THE FOUR CAUGHT BUGS (centerpiece) ──────────
    story.append(PageBreak())
    story.append(Paragraph("The four caught bugs — post-mortem", styles["H2"]))
    story.append(Paragraph(
        "Each bug produced a plausible-looking output indistinguishable from a correct result. "
        "Detection required checking the instrument, not the reading. This section is the "
        "thesis demonstrated.", styles["B"]))
    story.append(Spacer(1, 4))

    # Bug 1
    story.append(Paragraph("Bug 1: Pro model-ID alias (Finding 34)", styles["H3"]))
    story.append(make_table([
        ["Aspect", "Detail"],
        ["Mechanism", "PRO_MODEL=\"deepseek-chat\" in run_headline.py. DeepSeek's legacy alias "
         "\"deepseek-chat\" routes to deepseek-v4-flash (documented for deprecation 2026-07-24), "
         "not to deepseek-v4-pro. Every synthesis call tagged \"Pro\" actually ran the same model "
         "as the flash arm."],
        ["Why invisible", "The harness logged the REQUESTED model string (the constant), not the "
         "API response's served model field. Output records said model=\"deepseek-chat\" — "
         "indistinguishable from a real Pro call without checking the API response or billing."],
        ["How caught", "Billing dashboard: $0.00 on the deepseek-v4-pro line after a 776-query run. "
         "DeepSeek API docs confirmed the alias routing. Token/latency comparison showed identical "
         "distributions (same model both arms)."],
        ["Fix", "Model ID corrected to \"deepseek-v4-pro\". served_model field added to synthesize() "
         "(getattr(response._raw_response, 'model', None)). Corrected Pro run: 776/776 records "
         "show served_model=\"deepseek-v4-pro\", 2.3-4.2x latency (impossible if same model). "
         "Result: Pro indistinguishable from flash within \u00b12-4pp variance band."],
    ], [W*0.14, W*0.86], header_row=True))
    story.append(Spacer(1, 6))

    # Bug 2
    story.append(Paragraph("Bug 2: BM25 channel mismatch (Finding 45)", styles["H3"]))
    story.append(make_table([
        ["Aspect", "Detail"],
        ["Mechanism", "Combined-index dense retrieval (sqlite-vec) searched 4,496 mini-doc chunks. "
         "BM25 retrieval loaded the FULL corpus parquet (96,256 chunks for CUAD). The two channels "
         "of the hybrid retrieval searched different pools — dense saw 72 mini-split documents, "
         "BM25 saw all 462 CUAD documents."],
        ["Why invisible", "ContractNLI and PrivacyQA completed with 0 errors and saved results. "
         "ContractNLI's mismatch (701 mini vs 3,797 full) was proportionally smaller. PrivacyQA's "
         "mini=full (7 docs), so no mismatch. The output records looked normal — the hybrid just "
         "quietly mixed two different search scopes."],
        ["How caught", "CUAD crashed (96K-chunk BM25 index exceeded memory). Post-mortem traced the "
         "BM25 loading path: pd.read_parquet(params['parquet']) loaded the FULL parquet, not the "
         "mini-doc subset. ContractNLI results (already saved) were then flagged as corrupted."],
        ["Fix", "BM25 filtered to the same mini-doc set as the dense index before building the "
         "BM25Okapi index. Combined BM25: 11,524 chunks (channel-matched). ContractNLI re-run. "
         "Verified: log line \"BM25 combined: 11524 total chunks (channel-matched to dense index)\"."],
    ], [W*0.14, W*0.86], header_row=True))
    story.append(Spacer(1, 6))

    # Bug 3
    story.append(Paragraph("Bug 3: Zero-span offset (Finding 45)", styles["H3"]))
    story.append(make_table([
        ["Aspect", "Detail"],
        ["Mechanism", "The combined sqlite-vec index was built by scrolling Qdrant collections and "
         "copying payloads. Qdrant payloads contain chunk_id, doc_id, content, dataset_name — "
         "but NOT start_idx/end_idx (character spans live in the parquet files). "
         "payload.get('start_idx', 0) defaulted every span to (0, 0)."],
        ["Why invisible", "P@k/R@k are character-overlap ratios. With all spans at (0,0), the "
         "intersection with ground-truth spans was near-zero but not exactly zero (a few GT spans "
         "start near offset 0). CUAD showed R@8=0.049 with 100% routing recall — anomalous but not "
         "obviously impossible. Correctness/faithfulness judges (Layer 2) were unaffected (they score "
         "answers, not spans)."],
        ["How caught", "Sanity-check: CUAD R@8=0.049 with 100% routing recall and 8.0/8 correct-doc "
         "chunks contradicted expectations (perfect routing should yield high R@8). Spot-check of "
         "chunk_meta table: SELECT start_idx, end_idx FROM chunk_meta LIMIT 5 returned all zeros. "
         "Traced to payload.get() defaulting missing fields."],
        ["Fix", "Span lookup from parquet files via chunk_id join. Spot-checked 3 chunks: looked-up "
         "span text matched saved content text exactly. P@k/R@8 recomputed from saved records "
         "(no re-retrieval needed — chunks and answers were correct, only the metric was wrong). "
         "CUAD R@8: 0.049 \u2192 0.814 (real)."],
    ], [W*0.14, W*0.86], header_row=True))
    story.append(Spacer(1, 6))

    # Bug 4
    story.append(Paragraph("Bug 4: Chunk-size inconsistency (Finding 46)", styles["H3"]))
    story.append(make_table([
        ["Aspect", "Detail"],
        ["Mechanism", "MAUD's baseline SAC collection uses 2048-char chunks with ~512 overlap. "
         "ContractNLI, PrivacyQA, and CUAD use 512-char chunks with 128 overlap. The 4x "
         "difference was never documented as a deliberate choice. CLAUDE.md and Finding 45 "
         "stated \"SAC uses ~2048-char chunks\" as a blanket claim — wrong for 3 of 4 corpora."],
        ["Why invisible", "Per-corpus evaluations never compared chunk sizes across corpora. "
         "The combined-index headline mixed both sizes in one pool without flagging the asymmetry. "
         "The blanket claim passed review because MAUD (the largest corpus) dominated chunk counts."],
        ["How caught", "External-baseline verification: comparing P@k to the paper's RCTS 500-char "
         "chunks. Investigation of whether the comparison was confounded led to checking actual "
         "chunk sizes per corpus. Parquet analysis: MAUD median 1,802 chars, others median 407-434."],
        ["Fix", "Documented as Finding 46. CLAUDE.md corrected: \"512-char (CNL/PQA/CUAD), 2048-char "
         "(MAUD)\". External comparison: 3 corpora un-confounded (512 \u2248 500), MAUD excluded "
         "(2048 vs 500 = chunk-granularity confound). No code change — measurement/reporting fix."],
    ], [W*0.14, W*0.86], header_row=True))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Pattern: each bug produced a number that could have been reported as a finding. "
        "The Pro alias would have shipped \"Pro adds nothing\" (true, but for the wrong "
        "reason — it was flash-vs-flash). The span bug would have shipped CUAD R@8=0.049 "
        "as \"retrieval barely works on CUAD\" (false — real R@8=0.814). Verification "
        "caught the instrument error before it became a published conclusion.", styles["B"]))

    # ── PAGE 4: Results ───────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Results", styles["H2"]))

    story.append(Paragraph("Config-stack delta (controlled, method-level)", styles["H3"]))
    story.append(Paragraph(
        "Combined-index regime: 4 LegalBench-RAG corpora, 11,524 SAC chunks from 72 "
        "mini-split documents, 194 queries per corpus. Both arms use the same index, "
        "embedder (voyage-4), and judge.", styles["B"]))
    story.append(make_table([
        ["Corpus", "Correct Arm 0\u21921", "Faith Arm 0\u21921", "P@1 Arm 1", "R@8 Arm 1", "Latency"],
        ["ContractNLI", "62.9\u219271.1 (+8.2)", "89.1\u219294.1 (+5.0)", "0.422", "0.810", "5.6s"],
        ["PrivacyQA", "49.0\u219255.7 (+6.7)", "94.4\u219297.9 (+3.5)", "0.297", "0.579", "7.1s"],
        ["CUAD", "61.9\u219273.7 (+11.8)", "92.1\u219296.0 (+3.9)", "0.394", "0.814", "6.2s"],
        ["MAUD", "68.0\u219272.7 (+4.7)", "91.3\u219295.5 (+4.2)", "0.270", "0.783", "8.4s"],
        ["Average", "60.5\u219268.3 (+7.9)", "91.7\u219295.9 (+4.2)", "\u2014", "\u2014", "\u2014"],
    ], [W*0.16, W*0.20, W*0.20, W*0.12, W*0.12, W*0.10]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("External comparison (system-vs-system)", styles["H3"]))
    story.append(Paragraph(
        "vs arXiv 2408.10343 Table 5: RCTS 500-char, text-embedding-3-large, dense-only, "
        "no reranker. 3 un-confounded corpora (512-char \u2248 500-char). Full Meridian stack "
        "(SAC+CC+hybrid+routing+selector, voyage-4) vs bare baseline. The advantage bundles "
        "embedder + hybrid + fusion + routing — system-level, not method-alone.", styles["B"]))
    story.append(make_table([
        ["Corpus", "Meridian P@1", "RCTS P@1", "Meridian R@8", "RCTS R@8", "Note"],
        ["ContractNLI", "0.422", "0.066", "0.810", "0.250", "Benchmark provenance caveat"],
        ["CUAD", "0.394", "0.020", "0.814", "0.317", "\u2014"],
        ["PrivacyQA", "0.297", "0.144", "0.579", "0.424", "\u2014"],
        ["MAUD", "\u2014", "\u2014", "\u2014", "\u2014", "Excluded: 2048 vs 500-char confound"],
    ], [W*0.16, W*0.14, W*0.12, W*0.14, W*0.12, W*0.30]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("NFCorpus transfer (retrieval stack only)", styles["H3"]))
    story.append(Paragraph(
        "BEIR medical IR (3,633 docs, 323 queries). nDCG@10 = 0.399 (CC \u03b1=0.1, routing OFF). "
        "Above classic baselines: BM25 0.325, BM25+CE 0.350, contriever 0.328. "
        "CC fusion transfers (+5.6pp over RRF). Span-forensic framework NOT exercised "
        "(BEIR has document-level relevance, no character spans). Routing correctly "
        "self-disabled (dispersed relevance). This is a retrieval-transfer result.", styles["B"]))
    story.append(Paragraph(
        "Routing boundary characterized: monotonically hurts on NFCorpus "
        "(OFF 0.397 > k=10 0.333 > k=5 0.273 > k=3 0.227). Routing is a "
        "concentrated-relevance technique — the sweep auto-detects when it "
        "helps vs hurts (Findings 23, 45, 47).", styles["B"]))

    # ── PAGE 5: Limitations + What it's not ───────────────────
    story.append(PageBreak())
    story.append(Paragraph("Limitations", styles["H2"]))
    lims = [
        "<b>Routing degrades to 76%</b> on ContractNLI combined regime (NDA/commercial-contract "
        "confusion). 47/194 queries get 0/8 correct-document chunks.",
        "<b>Routing hurts on dispersed relevance.</b> Confirmed on NFCorpus. "
        "Concentrated-relevance only.",
        "<b>Span framework requires character-span ground truth.</b> Does not apply to "
        "document-level benchmarks (most of BEIR, MS MARCO). Scope boundary.",
        "<b>MAUD external comparison confounded</b> (2048-char vs 500-char chunk granularity).",
        "<b>ContractNLI external baseline caveated</b> (benchmark-file provenance differs "
        "from paper's generation pipeline).",
        "<b>Affirmative-only evaluation.</b> Correctness = recall, not false-positive rate.",
        "<b>Synthesis variance \u00b12-4pp</b> (Finding 40). Sub-band deltas are noise.",
        "<b>External multipliers are system-level,</b> not method-level.",
    ]
    for lim in lims:
        story.append(Paragraph(lim, styles["Bul"], bulletText="\u2022"))

    story.append(Spacer(1, 10))
    story.append(Paragraph("What Meridian is not", styles["H2"]))
    story.append(Paragraph(
        "Not an autonomous research agent (v1 direction, abandoned). "
        "Not a SOTA-everywhere claim (beats classic baselines, not modern dense SOTA). "
        "Not a framework-transfer claim (span taxonomy ran on LegalBench-RAG only; "
        "NFCorpus was retrieval-transfer). Not a tutorial.", styles["B"]))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.4, color=BORDER))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "47 findings with 7 corrections in docs/DECISIONS.md. "
        "Full methodology in docs/REPORT.md. "
        "Figures and reproducible generation in figures/.", styles["Small"]))

    doc.build(story)
    print(f"  {OUT}")


if __name__ == "__main__":
    print("Generating Autopsy PDF...")
    build()
    print("Done.")
