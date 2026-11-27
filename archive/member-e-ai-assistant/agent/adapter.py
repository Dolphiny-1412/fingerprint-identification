# agent/adapter.py
import os
import math
from typing import Dict, List, Any, Optional, Tuple


def _make_key_from_db_and_filename(db_name: str, filename: str) -> str:
    """Create canonical key like DB1_B_101_1 from db name and filename '101_1.tif'"""
    name_no_ext = os.path.splitext(filename)[0]
    return f"{db_name}_{name_no_ext}"


def build_matcher_catalog(matcher_db: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Robustly build a per-fingerprint matcher catalog.
    """
    catalog: Dict[str, Dict[str, Any]] = {}
    if not isinstance(matcher_db, dict):
        return catalog

    sample_keys = list(matcher_db.keys())[:6]
    if sample_keys and all(isinstance(k, str) and "_" in k for k in sample_keys):
        for k, v in matcher_db.items():
            if isinstance(v, dict):
                catalog[k] = v
        if catalog:
            return catalog

    if "processed_databases" in matcher_db and isinstance(matcher_db["processed_databases"], list):
        for db in matcher_db["processed_databases"]:
            db_name = db.get("name")
            results = db.get("results") or {}
            for item in results.get("individual_results", []) or []:
                filename = item.get("filename")
                if filename and db_name:
                    k = _make_key_from_db_and_filename(db_name, filename)
                    entry = dict(item)
                    catalog[k] = entry
        if catalog:
            return catalog

    for maybe_list_key in matcher_db:
        maybe_list = matcher_db.get(maybe_list_key)
        if isinstance(maybe_list, list) and len(maybe_list) > 0 and isinstance(maybe_list[0], dict):
            for rec in maybe_list:
                fname = rec.get("filename") or rec.get("file") or None
                if not fname:
                    imgp = rec.get("image_path") or rec.get("file_path") or rec.get("path")
                    if isinstance(imgp, str):
                        fname = os.path.basename(imgp)
                if fname:
                    dbname = rec.get("database") or maybe_list_key
                    if dbname and isinstance(dbname, str):
                        k = _make_key_from_db_and_filename(dbname, fname)
                    else:
                        name_no_ext = os.path.splitext(fname)[0]
                        k = name_no_ext
                    catalog[k] = rec
            if catalog:
                return catalog

    def _walk_and_extract(d):
        if isinstance(d, dict):
            fname = d.get("filename") or None
            imgp = d.get("image_path") or d.get("file_path") or d.get("path")
            dbname = d.get("database") or d.get("db") or None
            if not fname and isinstance(imgp, str):
                fname = os.path.basename(imgp)
            if fname:
                if dbname:
                    k = _make_key_from_db_and_filename(dbname, fname)
                else:
                    k = os.path.splitext(fname)[0]
                catalog.setdefault(k, {}).update(d)
            for v in d.values():
                _walk_and_extract(v)
        elif isinstance(d, list):
            for it in d:
                _walk_and_extract(it)

    _walk_and_extract(matcher_db)
    return catalog


def build_minutiae_catalog(minutiae_db: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build mapping key -> stats for minutiae C JSON format.
    """
    catalog: Dict[str, Dict[str, Any]] = {}
    if not isinstance(minutiae_db, dict):
        return catalog

    processed = minutiae_db.get("processed_databases") or []
    for db in processed:
        db_name = db.get("name")
        results = db.get("results", {}) or {}
        individual_results = results.get("individual_results") or []
        minutiae_counts = results.get("minutiae_counts") or []
        quality_scores = results.get("quality_scores") or []
        spoof_scores = results.get("spoof_scores") or []
        processing_times = results.get("processing_times") or []

        for idx, item in enumerate(individual_results):
            filename = item.get("filename") if isinstance(item, dict) else None
            if not filename:
                continue
            key = _make_key_from_db_and_filename(db_name, filename)
            entry = {
                "database": db_name,
                "filename": filename,
                "result_file": item.get("result_file"),
                "success": item.get("success", False),
                "minutiae_count": minutiae_counts[idx] if idx < len(minutiae_counts) else None,
                "quality_score": quality_scores[idx] if idx < len(quality_scores) else None,
                "spoof_score": spoof_scores[idx] if idx < len(spoof_scores) else None,
                "processing_time": processing_times[idx] if idx < len(processing_times) else None,
            }
            catalog[key] = entry

    return catalog


def build_classifier_catalog(classifier_db: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build mapping from classifier B if classifier_db is per-fingerprint.
    """
    if not isinstance(classifier_db, dict):
        return {}
    sample_keys = list(classifier_db.keys())[:5]
    if sample_keys and all(isinstance(k, str) and "_" in k for k in sample_keys):
        return {k: classifier_db[k] for k in classifier_db.keys()}
    return {}


def build_merged_catalog(
    classifier_db: Optional[Dict[str, Any]],
    minutiae_db: Optional[Dict[str, Any]],
    matcher_db: Optional[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Create a merged catalog keyed by fingerprint id.
    """
    matcher_catalog = build_matcher_catalog(matcher_db or {})
    minutiae_catalog = build_minutiae_catalog(minutiae_db or {})
    classifier_catalog = build_classifier_catalog(classifier_db or {})

    keys = set(matcher_catalog.keys()) | set(minutiae_catalog.keys()) | set(classifier_catalog.keys())
    keys_list = sorted(list(keys))

    merged: Dict[str, Dict[str, Any]] = {}
    for k in keys_list:
        merged[k] = {
            "id": k,
            "classifier": classifier_catalog.get(k),
            "minutiae": minutiae_catalog.get(k),
            "matcher": matcher_catalog.get(k),
        }

    return keys_list, merged


# -----------------------------
# Helper: sample large minutiae arrays
# -----------------------------
def _evenly_sample_points(points: List[Dict[str, Any]], max_points: int) -> List[Dict[str, Any]]:
    """
    Evenly sample a large minutiae list to at most max_points entries.
    Preserves order-ish and types; deterministic selection.
    """
    if not isinstance(points, list):
        return []
    n = len(points)
    if n <= max_points:
        return points
    # compute indices evenly spaced
    stride = n / float(max_points)
    indices = [min(n - 1, int(math.floor(i * stride))) for i in range(max_points)]
    # deduplicate indices while preserving order
    seen = set()
    sampled = []
    for i in indices:
        if i not in seen:
            sampled.append(points[i])
            seen.add(i)
        if len(sampled) >= max_points:
            break
    return sampled


def _normalize_score_to_01(val: Any) -> Optional[float]:
    """Normalize a score that may be 0..1 or 0..100 into 0..1 float; return None if unknown."""
    if val is None:
        return None
    try:
        f = float(val)
    except Exception:
        return None
    if f > 1.0:  # likely percentage
        f = max(0.0, min(100.0, f)) / 100.0
    return max(0.0, min(1.0, f))


def _resolve_access_decision(match_score: Optional[float], threshold: Optional[float], explicit_access: Any) -> Optional[bool]:
    """
    Decide access_granted:
      - If explicit_access (True/False) present, use it.
      - Else if match_score and threshold present, compare.
      - Else None (unknown).
    """
    if explicit_access is True:
        return True
    if explicit_access is False:
        return False
    if match_score is not None and threshold is not None:
        try:
            return float(match_score) >= float(threshold)
        except Exception:
            return None
    return None


def get_fingerprint_payload(
    merged_catalog: Dict[str, Dict[str, Any]],
    key: str,
    *,
    max_points: int = 80
) -> Dict[str, Any]:
    """
    Create a compact payload for AIAssistant.analyze by merging Person C (minutiae summary)
    and Person D (matcher + rich minutiae) for a single fingerprint key.

    The app workflow pages fingerprints one-by-one, so this function returns a budgeted
    payload (samples large point arrays).
    """
    rec = merged_catalog.get(key, {}) or {}

    # classifier can be missing (Person B). Keep empty dict if so.
    classifier = rec.get("classifier") or {}

    # From Person C (minutiae summary): may contain quality/spoof/minutiae_count info
    minutiae_summary_c = rec.get("minutiae") or {}

    # From Person D (matcher record): may contain minutiae points, image info, match results
    matcher_rec = rec.get("matcher") or {}

    # -------------------------
    # Minutiae: prefer detailed points from matcher_rec if present (Person D),
    # otherwise fall back to summary from Person C.
    # -------------------------
    minutiae_points = []
    minutiae_summary = {
        "minutiae_count": None,
        "endings": None,
        "bifurcations": None,
        "quality_score": None,
        "spoof_score": None,
    }

    # Try many common locations for minutiae points
    candidate_points = None
    # matcher_rec may contain a nested 'minutiae' dict with minutiae_points
    if isinstance(matcher_rec, dict):
        candidate_points = (
            matcher_rec.get("minutiae", {}).get("minutiae_points")
            or matcher_rec.get("minutiae_points")
            or matcher_rec.get("minutiae", {}).get("points")
            or None
        )

    # If no points in matcher, check the C summary for an embedded list
    if not candidate_points and isinstance(minutiae_summary_c, dict):
        candidate_points = (
            minutiae_summary_c.get("minutiae_points")
            or minutiae_summary_c.get("minutiae_list")
            or minutiae_summary_c.get("minutiae")
            or None
        )

    if isinstance(candidate_points, list):
        # sample down to max_points (keeps type info)
        minutiae_points = _evenly_sample_points(candidate_points, max_points)
        # counts inferred
        minutiae_summary["minutiae_count"] = len(candidate_points)
        # attempt to count endings/bifurcations from sampled list if possible (best-effort)
        try:
            endings = sum(1 for p in candidate_points if isinstance(p, dict) and p.get("type") == "ending")
            bifs = sum(1 for p in candidate_points if isinstance(p, dict) and p.get("type") == "bifurcation")
            minutiae_summary["endings"] = endings
            minutiae_summary["bifurcations"] = bifs
        except Exception:
            pass
    else:
        # No point array — rely on summary fields from C or nested in D
        # total
        total = (
            minutiae_summary_c.get("minutiae_count")
            or minutiae_summary_c.get("minutiae_total")
            or (matcher_rec.get("minutiae") or {}).get("minutiae_count")
            or matcher_rec.get("minutiae_count")
            or None
        )
        try:
            minutiae_summary["minutiae_count"] = int(total) if total is not None else None
        except Exception:
            minutiae_summary["minutiae_count"] = None

        # endings/bifurcations
        minutiae_summary["endings"] = (
            minutiae_summary_c.get("endings")
            or minutiae_summary_c.get("ridge_endings")
            or (matcher_rec.get("minutiae") or {}).get("endings")
            or None
        )
        minutiae_summary["bifurcations"] = (
            minutiae_summary_c.get("bifurcations")
            or minutiae_summary_c.get("bifurcations_count")
            or (matcher_rec.get("minutiae") or {}).get("bifurcations")
            or None
        )

    # pull quality/spoof info from C (prefer) then matcher_rec
    minutiae_summary["quality_score"] = (
        minutiae_summary_c.get("quality_score") or minutiae_summary_c.get("quality") or (matcher_rec.get("minutiae") or {}).get("quality_score") or None
    )
    minutiae_summary["spoof_score"] = (
        minutiae_summary_c.get("spoof_score") or (matcher_rec.get("minutiae") or {}).get("spoof_score") or matcher_rec.get("spoof_score") or None
    )

    # -------------------------
    # Matcher summary normalization (Person D)
    # -------------------------
    matcher_output = {}
    if isinstance(matcher_rec, dict):
        raw_score = matcher_rec.get("match_score") or matcher_rec.get("score") or matcher_rec.get("similarity")
        norm_score = _normalize_score_to_01(raw_score)
        matcher_output["match_score"] = norm_score
        matcher_output["matched_id"] = matcher_rec.get("matched_id") or matcher_rec.get("matched_with") or matcher_rec.get("id")
        matcher_output["threshold"] = _normalize_score_to_01(matcher_rec.get("threshold"))
        matcher_output["explicit_access"] = matcher_rec.get("access_granted") if "access_granted" in matcher_rec else None
        matcher_output["spoof_flag"] = bool(matcher_rec.get("spoof_flag", False))
        # Resolve access decision when possible
        matcher_output["access_granted"] = _resolve_access_decision(matcher_output["match_score"], matcher_output["threshold"], matcher_output["explicit_access"])

        # small diagnostics (image path/size) if present
        matcher_output["_diagnostics"] = {
            "raw_image_path": matcher_rec.get("image_path") or matcher_rec.get("file_path") or matcher_rec.get("path"),
            "image_size": matcher_rec.get("image_size") or (matcher_rec.get("image", {}) or {}).get("size"),
        }
    else:
        matcher_output = {"match_score": None, "matched_id": None, "access_granted": None, "spoof_flag": False, "_diagnostics": {}}

    # -------------------------
    # Classifier normalization (Person B optional)
    # -------------------------
    classifier_output = {}
    if isinstance(classifier, dict):
        classifier_output["label"] = classifier.get("label") or classifier.get("classification") or classifier.get("pattern") or "Unknown"
        classifier_output["confidence"] = _normalize_score_to_01(classifier.get("confidence") or classifier.get("score"))
    else:
        classifier_output = {"label": "Unknown", "confidence": None}

    # -------------------------
    # Build compact payload
    # -------------------------
    minutiae_output_compact = {
        "minutiae_summary": minutiae_summary,
        "minutiae_points_sample": minutiae_points,  # may be empty
    }

    payload = {
        "classifier_output": classifier_output,
        "minutiae_output": minutiae_output_compact,
        "matcher_output": matcher_output,
    }

    return payload


# -----------------------------
# Session pager helpers (mapping-friendly) - FIXED
# -----------------------------
def init_pager_in_session(session_state: dict, keys: List[str]) -> None:
    """
    Initialize or reset pager keys and index in a Streamlit-like session_state.
    This will always set fp_keys to the provided keys list. If the keys changed,
    reset fp_index to 0. Otherwise, ensure fp_index is within valid range.
    """
    # Accept any mapping-like session_state (Streamlit's session_state behaves like a dict)
    if session_state is None:
        return

    # Always set keys to the normalized list (copy)
    keys_list = list(keys) if keys is not None else []
    prev_keys = session_state.get("fp_keys")

    session_state["fp_keys"] = keys_list

    # If keys changed (or no previous index), reset index to 0
    if prev_keys != keys_list or "fp_index" not in session_state:
        session_state["fp_index"] = 0
        return

    # Otherwise clamp existing index to new range
    try:
        idx = int(session_state.get("fp_index", 0))
    except Exception:
        idx = 0
    max_idx = max(0, len(keys_list) - 1)
    session_state["fp_index"] = max(0, min(idx, max_idx))


def get_current_index(session_state: dict) -> int:
    """Return current fp_index as int, safe against bad values."""
    try:
        return int(session_state.get("fp_index", 0))
    except Exception:
        return 0


def set_index(session_state: dict, idx: int) -> None:
    """Set fp_index clamped to valid range given current fp_keys."""
    if session_state is None:
        return
    keys = session_state.get("fp_keys", []) or []
    try:
        idx_i = int(idx)
    except Exception:
        idx_i = 0
    clamped = max(0, min(idx_i, max(0, len(keys) - 1)))
    session_state["fp_index"] = clamped


def next_index(session_state: dict) -> int:
    """Advance index by one (clamped) and return new index."""
    idx = get_current_index(session_state) + 1
    set_index(session_state, idx)
    return get_current_index(session_state)


def prev_index(session_state: dict) -> int:
    """Move index back by one (clamped) and return new index."""
    idx = get_current_index(session_state) - 1
    set_index(session_state, idx)
    return get_current_index(session_state)


def current_key(session_state: dict) -> Optional[str]:
    """Return current key from fp_keys at fp_index, or None."""
    keys = session_state.get("fp_keys", []) or []
    idx = get_current_index(session_state)
    if keys and 0 <= idx < len(keys):
        return keys[idx]
    return None
