# app.py
import os
import json
import io
import traceback

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw

from agent.adapter import (
    build_matcher_catalog,
    build_minutiae_catalog,
    build_classifier_catalog,
    get_fingerprint_payload,
    init_pager_in_session,
    next_index,
    prev_index,
    current_key,
)

DATA_RAW = "data_raw"

st.set_page_config(page_title="Fingerprint AI Assistant", layout="wide")
# Apply global design system for improved readability (no change to flow)
try:
    from design_system import init_design_system
    init_design_system()
except Exception:
    pass

st.title("🖐️ Fingerprint Forensics — AI Assistant (file-driven)")

# ------------------------
# Helpers
# ------------------------
def load_json_if_exists(path: str) -> Optional[Any]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        st.error(f"Failed to load JSON {path}: {e}")
        return None

def find_all_jsons(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(".json")]

def assign_roles_by_structure(paths: List[str]):
    """Simple heuristics to pick classifier/minutiae/matcher from found json files."""
    classifier = None
    minutiae = None
    matcher = None
    for p in paths:
        data = load_json_if_exists(p)
        if not isinstance(data, dict):
            continue
        if "processed_databases" in data and minutiae is None:
            minutiae = data
            continue
        keys = list(data.keys())[:8]
        if keys and all(isinstance(k, str) and "_" in k for k in keys) and matcher is None:
            matcher = data
            continue
        vals = list(data.values())[:8]
        if vals and any(isinstance(v, dict) and ("label" in v or "confidence" in v) for v in vals) and classifier is None:
            classifier = data
            continue
    # fallback by filename hints
    for p in paths:
        name = os.path.basename(p).lower()
        data = load_json_if_exists(p)
        if not isinstance(data, dict):
            continue
        if minutiae is None and ("minutiae" in name or "_c" in name):
            minutiae = data
        if matcher is None and ("matcher" in name or "_d" in name):
            matcher = data
        if classifier is None and ("classifier" in name or "_b" in name):
            classifier = data
    return classifier, minutiae, matcher

def draw_minutiae_overlay_from_path(image_path: Optional[str], points: List[Dict[str, Any]], max_size=800):
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception:
        return None
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    for p in points or []:
        try:
            x = int(p.get("x") * scale) if p.get("x") is not None else None
            y = int(p.get("y") * scale) if p.get("y") is not None else None
            t = (p.get("type") or "unknown").lower()
            if x is None or y is None:
                continue
            if t == "bifurcation":
                color = (0,166,90,220); r = 3
            elif t == "ending":
                color = (255,99,71,220); r = 2
            else:
                color = (100,149,237,200); r = 2
            draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline=(0,0,0,120))
        except Exception:
            continue
    combined = Image.alpha_composite(img, overlay)
    buf = io.BytesIO()
    combined.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf

# ------------------------
# Load JSON files and assign roles
# ------------------------
json_paths = find_all_jsons(DATA_RAW)
st.sidebar.header("Data files found")
if not json_paths:
    st.sidebar.error(f"No JSON files found in `{DATA_RAW}`. Put your Person C/D JSONs there.")
else:
    for p in json_paths:
        st.sidebar.write("-", os.path.basename(p))

classifier_raw, minutiae_raw, matcher_raw = assign_roles_by_structure(json_paths)

# Show explicit status
st.sidebar.markdown("**Assigned roles (heuristic):**")
st.sidebar.write("Classifier (B):", bool(classifier_raw))
st.sidebar.write("Minutiae (C):", bool(minutiae_raw))
st.sidebar.write("Matcher (D):", bool(matcher_raw))
st.sidebar.markdown("---")

# Build per-role raw catalogs for raw per-fingerprint display
matcher_catalog_raw = build_matcher_catalog(matcher_raw or {}) if matcher_raw else {}
minutiae_catalog_raw = build_minutiae_catalog(minutiae_raw or {}) if minutiae_raw else {}
classifier_catalog_raw = build_classifier_catalog(classifier_raw or {}) if classifier_raw else {}

# Build merged keys using intersection (C ∩ D) with union fallback
matcher_keys = set(matcher_catalog_raw.keys())
minutiae_keys = set(minutiae_catalog_raw.keys())
classifier_keys = set(classifier_catalog_raw.keys())

intersection_keys = sorted(list(matcher_keys & minutiae_keys))
union_keys = sorted(list(matcher_keys | minutiae_keys | classifier_keys))

if intersection_keys:
    keys_list = intersection_keys
else:
    keys_list = union_keys

# Build merged dictionary for the chosen keys_list
merged_catalog: Dict[str, Dict[str, Any]] = {}
for k in keys_list:
    merged_catalog[k] = {
        "id": k,
        "classifier": classifier_catalog_raw.get(k),
        "minutiae": minutiae_catalog_raw.get(k),
        "matcher": matcher_catalog_raw.get(k),
    }

# ------------------------
# Robust session initialization:
# Only initialize pager/catalog in session when the set of data files changes.
# This prevents accidental fp_index resets on every rerun.
# ------------------------
def _compute_data_signature(paths: List[str]) -> Tuple[str, ...]:
    # Sorted base filenames form a stable signature
    return tuple(sorted([os.path.basename(p) for p in (paths or [])]))

files_sig = _compute_data_signature(json_paths)
if st.session_state.get("data_sig") != files_sig:
    # Data changed -> (re)initialize
    st.session_state["data_sig"] = files_sig
    st.session_state["fp_keys"] = list(keys_list)
    st.session_state["merged_catalog"] = merged_catalog
    st.session_state["fp_index"] = 0
    # Keep a friendly flag (not required) to know we just initialized
    st.session_state["data_initialized"] = True
else:
    # Keep merged_catalog in session if already present; if absent, set it
    st.session_state.setdefault("merged_catalog", merged_catalog)
    st.session_state.setdefault("fp_keys", list(keys_list))
    st.session_state.setdefault("fp_index", 0)

# Ensure fp_index is clamped to valid range using your helper
init_pager_in_session(st.session_state, st.session_state.get("fp_keys", []))

# Diagnostics panel (very helpful)
with st.expander("🔧 Diagnostic / Debug (show to me if things fail)", expanded=True):
    st.write("data_raw files:", [os.path.basename(p) for p in json_paths])
    st.write("Assigned roles (classifier, minutiae, matcher):", bool(classifier_raw), bool(minutiae_raw), bool(matcher_raw))
    st.write("Using intersection keys (C ∩ D) count:", len(matcher_keys & minutiae_keys))
    st.write("Selected keys count (used for iteration):", len(st.session_state.get("fp_keys", [])))
    st.write("First 10 selected keys:", st.session_state.get("fp_keys", [])[:10])
    st.write("Current fp_index:", st.session_state.get("fp_index"))
    st.write("Sample merged entry (first key):", st.session_state.get("fp_keys", [None])[0])
    st.write("Matcher catalog sample keys:", list(matcher_catalog_raw.keys())[:6])
    st.write("Minutiae catalog sample keys:", list(minutiae_catalog_raw.keys())[:6])
    st.write("Classifier catalog sample keys:", list(classifier_catalog_raw.keys())[:6])

# If no keys -> bail with clear message
if not st.session_state.get("fp_keys"):
    st.warning("No fingerprint entries extracted from the provided JSONs. Check diagnostic panel above and paste here the file listing if you want help.")
    st.stop()

# ------------------------
# Navigation controls: PREV / NEXT
# ------------------------
def _on_prev():
    try:
        prev_index(st.session_state)
    except Exception as e:
        st.error(f"Prev failed: {e}")

def _on_next():
    try:
        next_index(st.session_state)
    except Exception as e:
        st.error(f"Next failed: {e}")

nav_col, info_col = st.columns([1, 4])
with nav_col:
    st.button("◀ Prev", use_container_width=True, on_click=_on_prev)
    st.button("Next ▶", use_container_width=True, on_click=_on_next)
with info_col:
    idx = int(st.session_state.get("fp_index", 0))
    total = len(st.session_state.get("fp_keys", []))
    ck = current_key(st.session_state)
    st.markdown(f"**Fingerprint {idx + 1} / {total}** — key: `{ck}`")

# ------------------------
# Show current fingerprint view
# ------------------------
sel_key = current_key(st.session_state)
merged_catalog = st.session_state.get("merged_catalog", {}) or {}

# If somehow the current key isn't in merged (recover)
if not sel_key or sel_key not in merged_catalog:
    init_pager_in_session(st.session_state, st.session_state.get("fp_keys", []))
    sel_key = current_key(st.session_state)

if not sel_key:
    st.error("No current key selected (unexpected).")
    st.stop()

rec = merged_catalog.get(sel_key) or {}

left, center, right = st.columns([1, 1.5, 1.5])

with left:
    st.subheader("Metadata (merged)")
    st.write("ID:", sel_key)
    mino = rec.get("minutiae") if isinstance(rec.get("minutiae"), dict) else {}
    matc = rec.get("matcher") if isinstance(rec.get("matcher"), dict) else {}
    st.write("Database (C):", mino.get("database"))
    st.write("Filename (C/D):", (mino.get("filename") if mino else None) or (matc.get("filename") if matc else None))
    st.write("Minutiae summary count (C):", mino.get("minutiae_count") if mino else None)
    st.write("Quality score (C):", mino.get("quality_score") if mino else None)
    st.write("Match score (D):", matc.get("match_score") if matc else None)
    st.write("Matched id (D):", matc.get("matched_id") if matc else None)

with center:
    st.subheader("Image / Minutiae overlay (if local image path available)")
    payload = get_fingerprint_payload(merged_catalog, sel_key)
    diag = (payload.get("matcher_output") or {}).get("_diagnostics", {}) or {}
    raw_image_path = diag.get("raw_image_path") or diag.get("file_path") or diag.get("path")
    if raw_image_path and not os.path.exists(raw_image_path):
        cand = os.path.join(DATA_RAW, os.path.basename(raw_image_path))
        if os.path.exists(cand):
            raw_image_path = cand
    pts = (payload.get("minutiae_output") or {}).get("minutiae_points_sample", []) or []
    try:
        buf = draw_minutiae_overlay_from_path(raw_image_path, pts)
    except Exception as e:
        buf = None
        st.warning(f"Overlay draw failed: {e}")
    if buf:
        st.image(buf, use_column_width=True)
    else:
        st.info("No local image found to display. Showing minutiae sample (if any).")
        if pts:
            st.table(pts[:50])

    # --- NEW: expose the normalized payload for this fingerprint for debugging
    if st.button("🔎 Show normalized payload for current fingerprint"):
        st.subheader("Normalized payload (sent to assistant)")
        st.json(payload)

with right:
    st.subheader("Raw JSONs (per-fingerprint)")
    c_raw = minutiae_catalog_raw.get(sel_key)
    d_raw = matcher_catalog_raw.get(sel_key)

    st.markdown("**Person C (minutiae JSON entry)**")
    if isinstance(c_raw, dict) and c_raw:
        with st.expander("Show C JSON", expanded=False):
            st.json(c_raw)
    else:
        st.info("No Person C entry found for this fingerprint key.")

    st.markdown("**Person D (matcher JSON entry)**")
    if isinstance(d_raw, dict) and d_raw:
        with st.expander("Show D JSON", expanded=False):
            st.json(d_raw)
    else:
        st.info("No Person D entry found for this fingerprint key.")

    st.markdown("---")
    st.subheader("AI Actions")
    # lazy-instantiating assistant to avoid import errors if not needed
    assistant = st.session_state.get("assistant")
    if not assistant:
        try:
            from agent.assistant import AIAssistant
            assistant = AIAssistant(openai_api_key=None)
            st.session_state.assistant = assistant
        except Exception as e:
            assistant = None
            st.session_state.setdefault("assistant_import_error", str(e))

    # Short Analyze (safe) - pass the SINGLE normalized payload dict
    if st.button("⚡ Short Analyze", use_container_width=True):
        try:
            payload = get_fingerprint_payload(st.session_state.get("merged_catalog", {}), sel_key)
            st.write("Payload preview (short):")
            st.json(payload)
            if not st.session_state.get("assistant"):
                st.error("Assistant not available (check diagnostics).")
            else:
                res = st.session_state.get("assistant").analyze(payload, explanation=True, max_length="short")
                st.session_state.last_result = res
        except Exception as e:
            st.error(f"Analyze failed: {e}")
            st.text(traceback.format_exc())

    # Long Analyze (safe)
    if st.button("🧠 Long Analyze", use_container_width=True):
        try:
            payload = get_fingerprint_payload(st.session_state.get("merged_catalog", {}), sel_key)
            st.write("Payload preview (long):")
            st.json(payload)
            if not st.session_state.get("assistant"):
                st.error("Assistant not available (check diagnostics).")
            else:
                res = st.session_state.get("assistant").analyze(payload, explanation=True, max_length="long")
                st.session_state.last_result = res
        except Exception as e:
            st.error(f"Analyze failed: {e}")
            st.text(traceback.format_exc())

# ------------------------
# Bottom: Results & Chat (fixed)
# ------------------------
st.markdown("### Analysis Result")

# assistant health snippet
assistant = st.session_state.get("assistant")
if assistant:
    try:
        client_ok = bool(getattr(assistant, "client", None))
        openai_err = getattr(assistant, "_openai_import_error", None)
        st.caption(f"Assistant loaded — OpenAI client: {client_ok}, import_error: {openai_err}")
    except Exception:
        pass
else:
    imp_err = st.session_state.get("assistant_import_error") or None
    if imp_err:
        st.caption(f"Assistant not instantiated: {imp_err}")
    else:
        st.caption("Assistant not instantiated (using deterministic fallbacks).")

if st.session_state.get("last_result"):
    with st.expander("Latest AI Result", expanded=True):
        # non-empty label + hidden for accessibility
        st.text_area("AI result (hidden label)", value=st.session_state["last_result"], height=300, label_visibility="hidden")
else:
    st.info("No analysis yet — press Short/Long Analyze for current fingerprint.")

st.markdown("### Quick Chat")
qcol1, qcol2 = st.columns([4,1])
with qcol1:
    user_q = st.text_input("Ask a question about the currently-selected fingerprint:", key="chat_input", label_visibility="visible")
with qcol2:
    if st.button("Send"):
        if not user_q:
            st.warning("Write a question first.")
        else:
            try:
                payload = get_fingerprint_payload(st.session_state.get("merged_catalog", {}), sel_key)
                st.write("Payload used for Q/A:")
                st.json(payload)
                if not st.session_state.get("assistant"):
                    st.error("Assistant not available (check diagnostics).")
                else:
                    reply = st.session_state.get("assistant").analyze(payload, user_question=user_q, explanation=True, max_length="short")
                    st.session_state.chat_history = st.session_state.get("chat_history", []) + [{"q": user_q, "a": reply}]
                    st.session_state.last_result = reply
            except Exception as e:
                st.error(f"Chat failed: {e}")
                st.text(traceback.format_exc())

if st.session_state.get("chat_history"):
    with st.expander("Chat history", expanded=False):
        for item in reversed(st.session_state.chat_history[-12:]):
            st.markdown(f"**Q:** {item['q']}\n- **A:** {item['a']}")
