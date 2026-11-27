# agent/explain_rules.py
"""
Deterministic rule-based explanation logic.

This module provides functions that create human-readable explanations
from classifier, minutiae and matcher outputs. They are defensive about
input shapes so they work with:
 - per-fingerprint dicts produced by matcher (detailed)
 - summary dicts (minutiae C / person-D summaries)
 - lists of minutiae points
"""

from typing import Dict, List, Any, Union, Optional


def _normalize_confidence(conf: Any) -> Optional[float]:
    """
    Normalize confidence into a float in range 0.0-1.0 when possible.
    Accepts None, 0-1 floats, or 0-100 percents.
    Returns None if unknown.
    """
    if conf is None:
        return None
    try:
        f = float(conf)
    except Exception:
        return None
    # Heuristic: if > 1, assume percent (0-100)
    if f > 1:
        f = max(0.0, min(100.0, f)) / 100.0
    # clamp
    return max(0.0, min(1.0, f))


def classification_explanation(classifier: Dict[str, Any]) -> str:
    """
    classifier may contain keys: label, classification, pattern, confidence, score.
    """
    if not isinstance(classifier, dict):
        return "No classifier information available."

    label = (
        classifier.get("label")
        or classifier.get("classification")
        or classifier.get("pattern")
        or classifier.get("pattern_type")
        or "Unknown"
    )

    raw_conf = classifier.get("confidence") or classifier.get("score") or classifier.get("confidence_score")
    conf = _normalize_confidence(raw_conf)

    if conf is None:
        return f"Classifier returned label: **{label}** (confidence unknown)."
    return f"The fingerprint was classified as **{label}** with {conf * 100:.1f}% confidence."


def minutiae_explanation(minutiae: Union[List[Dict[str, Any]], Dict[str, Any]]) -> str:
    """
    Accepts:
      - list of minutiae points: [{'type':'ending','x':..,'y':..}, ...]
      - summary dict containing minutiae_count, endings, bifurcations, or nested shapes
    Returns a concise description.
    """
    # If it's a list of minutiae points
    if isinstance(minutiae, list):
        total = len(minutiae)
        bif = sum(1 for m in minutiae if m.get("type") == "bifurcation")
        end = sum(1 for m in minutiae if m.get("type") == "ending")
        other = total - bif - end
        if total == 0:
            return "No minutiae points were detected."
        parts = [
            f"{total} minutiae detected",
            f"{bif} bifurcation{'s' if bif != 1 else ''}",
            f"{end} ridge ending{'s' if end != 1 else ''}"
        ]
        if other:
            parts.append(f"{other} other feature{'s' if other != 1 else ''}")
        return ", ".join(parts) + "."

    # If it's a dict summary
    if isinstance(minutiae, dict):
        # Try many common key names
        total = (
            minutiae.get("minutiae_count")
            or minutiae.get("minutiae_total")
            or minutiae.get("minutiaeCount")
            or None
        )
        endings = minutiae.get("endings") or minutiae.get("ridge_endings") or minutiae.get("endings_count") or 0
        bif = minutiae.get("bifurcations") or minutiae.get("bifurcations_count") or 0

        # Coerce to ints when possible
        try:
            total_i = int(total) if total is not None else None
        except Exception:
            total_i = None
        try:
            endings_i = int(endings) if endings is not None else 0
        except Exception:
            endings_i = 0
        try:
            bif_i = int(bif) if bif is not None else 0
        except Exception:
            bif_i = 0

        if total_i is None or total_i == 0:
            # maybe points array exists inside
            points = minutiae.get("minutiae_points") or minutiae.get("minutiae_list") or minutiae.get("minutiae")
            if isinstance(points, list) and len(points) > 0:
                return minutiae_explanation(points)
            return "No minutiae points were detected."

        other = max(0, total_i - endings_i - bif_i)
        parts = [
            f"{total_i} minutiae detected",
            f"{bif_i} bifurcation{'s' if bif_i != 1 else ''}",
            f"{endings_i} ridge ending{'s' if endings_i != 1 else ''}"
        ]
        if other:
            parts.append(f"{other} other feature{'s' if other != 1 else ''}")
        return ", ".join(parts) + "."

    return "No minutiae information available."


def quality_checks(minutiae: Union[List[Dict[str, Any]], Dict[str, Any]], image_meta: Optional[Dict[str, Any]] = None) -> str:
    """
    Basic heuristic checks indicating quality/spoof issues.
    Accepts either the minutiae list or a summary dict (which may contain quality_score/spoof_score).
    """
    notes = []

    # If summary dict with quality/spoof
    if isinstance(minutiae, dict):
        q = minutiae.get("quality_score") or minutiae.get("quality") or minutiae.get("quality_score_mean")
        s = minutiae.get("spoof_score") or minutiae.get("spoof_risk") or minutiae.get("spoofness")
        try:
            if q is not None:
                qf = float(q)
                if qf < 0.5:
                    notes.append("Low overall quality score.")
            if s is not None:
                sf = float(s)
                if sf > 0.5:
                    notes.append("Spoof risk is elevated.")
        except Exception:
            pass

        # If total minutiae reported and too low
        tc = minutiae.get("minutiae_count")
        if tc is not None:
            try:
                if int(tc) < 10:
                    notes.append("Low minutiae count — matching confidence may be low.")
            except Exception:
                pass

    else:
        # minutiae is a list
        try:
            if len(minutiae) < 10:
                notes.append("Low minutiae count — matching confidence may be low.")
        except Exception:
            pass

    # Image metadata checks (if provided)
    if image_meta and isinstance(image_meta, dict):
        blur = image_meta.get("blur")
        contrast = image_meta.get("contrast")
        dpi = image_meta.get("dpi") or image_meta.get("resolution_dpi")
        try:
            if blur is not None and float(blur) > 0.6:
                notes.append("Image appears blurred.")
            if contrast is not None and float(contrast) < 0.5:
                notes.append("Low contrast detected.")
            if dpi is not None and float(dpi) < 300:
                notes.append("Low DPI; features may be unreliable.")
        except Exception:
            pass

    if not notes:
        return "No immediate quality issues detected."
    return " ".join(notes)


def matcher_explanation(matcher: Dict[str, Any]) -> str:
    """
    Explain matcher output. Accepts dicts with keys like match_score, matched_id, matched_with,
    spoof_flag, spoof_score, access_granted, threshold, etc.
    """
    if not isinstance(matcher, dict):
        return "No matcher information available."

    # Fetch score
    score = matcher.get("match_score") or matcher.get("score") or matcher.get("similarity")
    matched_id = matcher.get("matched_id") or matcher.get("matched_with") or matcher.get("id")
    spoof_flag = matcher.get("spoof_flag")
    spoof_score = matcher.get("spoof_score") or matcher.get("spoof_risk")
    access = matcher.get("access_granted")

    # Normalize score to 0..1 if possible
    try:
        if score is not None:
            s = float(score)
            if s > 1:
                s = max(0.0, min(100.0, s)) / 100.0
            score = s
    except Exception:
        score = None

    # Decide spoof
    if spoof_flag is None and spoof_score is not None:
        try:
            spoof_flag = float(spoof_score) > 0.5
        except Exception:
            spoof_flag = False

    if spoof_flag:
        return "Spoof indicators flagged. The sample may be fake or tampered with."

    # If no score but access decision present
    if score is None:
        if access is True:
            return f"Access granted to **{matched_id}** (score not provided)."
        if access is False:
            return "Access denied (score not provided)."
        return "Matcher returned no score or decision."

    # With score -> format
    perc = score * 100
    if access is True:
        return f"Matched to ID **{matched_id}** with a score of {perc:.1f}%. Access granted."
    if access is False:
        return f"No reliable match (top score {perc:.1f}%). Access denied."

    # If access not provided, use threshold if exists
    threshold = matcher.get("threshold")
    if threshold is not None:
        try:
            t = float(threshold)
            # normalize threshold if >1
            if t > 1:
                t = max(0.0, min(100.0, t)) / 100.0
            if score >= t:
                return f"Matched to ID **{matched_id}** with a score of {perc:.1f}% (threshold {t*100:.1f}%). Access likely granted."
            else:
                return f"Match score {perc:.1f}% below threshold {t*100:.1f}% — access likely denied."
        except Exception:
            pass

    # Default wording if access unknown
    return f"Top match score: {perc:.1f}%. Matched ID: {matched_id or 'unknown'}."


def summarize_all(classifier: Dict[str, Any], minutiae: Union[List[Dict[str, Any]], Dict[str, Any]],
                  matcher: Dict[str, Any], image_meta: Optional[Dict[str, Any]] = None) -> str:
    """
    High level summary assembling all parts together.
    """
    parts = [
        classification_explanation(classifier),
        minutiae_explanation(minutiae),
        quality_checks(minutiae, image_meta),
        matcher_explanation(matcher)
    ]
    return "\n\n".join(parts)
