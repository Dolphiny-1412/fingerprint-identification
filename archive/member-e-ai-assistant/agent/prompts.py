# agent/prompts.py
"""
Prompt templates and builders for the Fingerprint AI Assistant.

Provides:
 - build_short_summary_prompt(context)
 - build_long_summary_prompt(context)
 - build_qa_prompt(context, question)
 - build_db_entry_prompt(key, entry)   # for iterating DB entries
 - A short usage note.
"""

from typing import Dict, Any, Optional

SHORT_INSTRUCTIONS = (
    "You are an expert forensic fingerprint assistant. Produce a concise "
    "summary (2–3 short sentences) focused on the facts, key numerical "
    "values, and a single-line actionable note (if any). Do NOT invent facts."
)

LONG_INSTRUCTIONS = (
    "You are an expert forensic fingerprint assistant. Produce a detailed, "
    "forensic-style explanation suitable for another expert. Cover: classifier "
    "interpretation (strengths/weaknesses), minutiae quality & distribution, "
    "matcher confidence, possible spoof indicators, and recommended next steps. "
    "Be precise and highlight uncertainties. Do NOT invent facts."
)

QA_INSTRUCTIONS = (
    "You are an expert forensic fingerprint assistant. Given the context, answer "
    "the user's question clearly and directly. If the question asks for an opinion, "
    "state uncertainty and provide reasoning. Be concise unless the user explicitly "
    "asks for more detail."
)


def _format_context(context: Dict[str, Any]) -> str:
    """
    Render the classifier/minutiae/matcher context into a compact bullet list
    to include in prompts. Keep it short to reduce tokens.
    """
    lines = []
    c = context.get("Classifier") or context.get("classifier") or {}
    m = context.get("Minutiae") or context.get("minutiae") or {}
    ma = context.get("Matcher") or context.get("matcher") or {}

    # Classifier
    label = c.get("label") or c.get("classification") or c.get("pattern") or "Unknown"
    conf = c.get("confidence") or c.get("score") or c.get("confidence_score")
    lines.append(f"- Pattern: {label} (confidence: {conf})")

    # Minutiae
    if isinstance(m, dict):
        mc = m.get("minutiae_count") or m.get("total") or m.get("minutiae_total")
        endings = m.get("endings") or m.get("ridge_endings") or m.get("endings_count")
        bif = m.get("bifurcations")
        qs = m.get("quality_score")
        lines.append(f"- Minutiae: count={mc}, endings={endings}, bifurcations={bif}, quality={qs}")
    elif isinstance(m, list):
        lines.append(f"- Minutiae: {len(m)} points (list provided)")

    # Matcher
    score = ma.get("match_score") or ma.get("score") or ma.get("similarity")
    mid = ma.get("matched_id") or ma.get("matched_with") or ma.get("id")
    spoof = ma.get("spoof_score") or ma.get("spoof_risk") or ma.get("spoof_flag")
    lines.append(f"- Matcher: top_score={score}, matched_id={mid}, spoof={spoof}")

    # optional short metadata
    ts = ma.get("processed_timestamp") or m.get("processed_timestamp") or None
    if ts:
        lines.append(f"- Processed at: {ts}")

    return "\n".join(lines)


def build_short_summary_prompt(context: Dict[str, Any]) -> str:
    """
    Build a compact prompt for a short (2-3 sentence) AI summary.
    """
    ctx = _format_context(context)
    prompt = (
        f"{SHORT_INSTRUCTIONS}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Concise summary:"
    )
    return prompt


def build_long_summary_prompt(context: Dict[str, Any]) -> str:
    """
    Build a detailed forensic prompt.
    """
    ctx = _format_context(context)
    prompt = (
        f"{LONG_INSTRUCTIONS}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Detailed explanation:"
    )
    return prompt


def build_qa_prompt(context: Dict[str, Any], question: str) -> str:
    """
    Build a question-answering prompt using the available context.
    """
    ctx = _format_context(context)
    prompt = (
        f"{QA_INSTRUCTIONS}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )
    return prompt


def build_db_entry_prompt(key: str, entry: Dict[str, Any]) -> str:
    """
    Brief prompt describing a single DB entry (for iterative per-fingerprint review).
    Keeps the fields small and focused (image size, minutiae_count, quality/spoof scores).
    """
    parts = [f"Fingerprint ID: {key}"]
    m = entry.get("minutiae") or {}
    if isinstance(m, dict):
        parts.append(f"Minutiae count: {m.get('minutiae_count')} (endings={m.get('endings')}, bifurcations={m.get('bifurcations')})")
        parts.append(f"Quality score: {m.get('quality_score')}, Spoof score: {m.get('spoof_score')}")
    ma = entry.get("matcher") or entry.get("match") or {}
    if isinstance(ma, dict):
        parts.append(f"Matcher top_score: {ma.get('match_score')}, matched_id: {ma.get('matched_id')}")
    prompt = (
        f"You are reviewing one fingerprint entry for triage.\n\n"
        f"{' | '.join(parts)}\n\n"
        f"Provide a short triage note (1-2 sentences) describing whether the sample is high-quality, suspicious (possible spoof), or requires re-capture."
    )
    return prompt


# Usage note (not executed): call the appropriate builder, then send the resulting
# prompt as the user's message to your LLM client. Keep prompts short and include
# only the fields required for the question to reduce token cost.
