# agent/assistant.py
"""
Robust AIAssistant wrapper that accepts either:
 - three separate args (classifier_output, minutiae_output, matcher_output)
 - OR a single "normalized" payload dict: {"classifier":..., "minutiae":..., "matcher":...}
 - OR a single "compat" payload dict: {"classifier_output":..., "minutiae_output":..., "matcher_output":...}
 - OR an older combined sample dict (legacy fallback).

It also supports AI explanations when OpenAI is available (gracefully degrades otherwise).
"""

from typing import Optional, Dict, Any, List
import time

from agent.explain_rules import (
    classification_explanation,
    minutiae_explanation,
    quality_checks,
    matcher_explanation,
    summarize_all,
)

# Optional prompt builders (used if present)
try:
    from agent.prompts import (
        build_short_summary_prompt,
        build_long_summary_prompt,
        build_qa_prompt,
    )
    _PROMPTS_AVAILABLE = True
except Exception:
    _PROMPTS_AVAILABLE = False

# optional config import
try:
    from config import OPENAI_API_KEY as CONFIG_OPENAI_KEY
except Exception:
    CONFIG_OPENAI_KEY = None

# optional OpenAI import (do not raise at import time)
try:
    from openai import OpenAI
    _OPENAI_IMPORT_ERROR = None
except Exception as e:
    OpenAI = None
    _OPENAI_IMPORT_ERROR = e


class AIAssistant:
    """
    Person E – AI Assistant & Explainability Architect

    Public methods:
      - analyze(...)  : main workhorse (summary/explanation or Q&A)
      - generate_response(step, input_payload, ...) : per-step explainer for UI
    """

    def __init__(self, openai_api_key: Optional[str] = None, timeout_s: float = 30.0, max_retries: int = 2):
        # Accept explicit key or fall back to config value (config.py).
        self.openai_api_key = openai_api_key or CONFIG_OPENAI_KEY
        self.client = None
        self._openai_import_error = _OPENAI_IMPORT_ERROR
        self.timeout_s = timeout_s
        self.max_retries = max_retries

        if self.openai_api_key and OpenAI is not None:
            try:
                # Instantiate the modern OpenAI client wrapper if available
                self.client = OpenAI(api_key=self.openai_api_key)
            except Exception:
                self.client = None  # fail closed if client init fails

    # -------------------------
    # Public API
    # -------------------------
    def analyze(
        self,
        classifier_output: Dict[str, Any],
        minutiae_output: Optional[Dict[str, Any]] = None,
        matcher_output: Optional[Dict[str, Any]] = None,
        explanation: bool = False,
        user_question: Optional[str] = None,
        max_length: str = "long",
    ) -> str:
        """
        Analyze fingerprint payload. Accepts:
          - analyze(classifier_dict, minutiae_dict, matcher_dict, ...)
          - analyze(normalized_payload_dict, explanation=..., ...) where dict has keys {"classifier","minutiae","matcher"}
          - analyze(compat_payload_dict, ...) where dict has keys {"classifier_output","minutiae_output","matcher_output"}
          - analyze(legacy_sample_dict, ...) where only one dict provided.

        Returns a plain-text explanation (deterministic or AI-generated).
        """

        # --- Unpack wrapper payload if user passed {"classifier":..., "minutiae":..., "matcher":...}
        if (
            isinstance(classifier_output, dict)
            and minutiae_output is None
            and matcher_output is None
            and set(classifier_output.keys()) >= {"classifier", "minutiae", "matcher"}
        ):
            wrapper = classifier_output
            classifier_output = wrapper.get("classifier") or {}
            minutiae_output = wrapper.get("minutiae") or {}
            matcher_output = wrapper.get("matcher") or {}

        # --- Unpack compat wrapper payload if user passed {"classifier_output":..., "minutiae_output":..., "matcher_output":...}
        elif (
            isinstance(classifier_output, dict)
            and minutiae_output is None
            and matcher_output is None
            and set(classifier_output.keys()) >= {"classifier_output", "minutiae_output", "matcher_output"}
        ):
            wrapper = classifier_output
            classifier_output = wrapper.get("classifier_output") or {}
            minutiae_output = wrapper.get("minutiae_output") or {}
            matcher_output = wrapper.get("matcher_output") or {}

        # --- Legacy single-dict fallback (tolerant)
        elif (
            isinstance(classifier_output, dict)
            and minutiae_output is None
            and matcher_output is None
        ):
            sample = classifier_output
            # best-effort synthesize, but prefer explicit nested inputs
            minutiae_output = {
                "ridge_endings": sample.get("minutiae_endings") or sample.get("endings") or (sample.get("minutiae_count", 0) // 2),
                "bifurcations": sample.get("minutiae_bifurcations") or sample.get("bifurcations") or (sample.get("minutiae_count", 0) // 2),
                "minutiae_count": sample.get("minutiae_count"),
            }
            matcher_output = {
                "match_score": sample.get("match_score", None),
                "matched_id": sample.get("fingerprint_id", "Unknown"),
                "access_granted": sample.get("access_granted", None),
                "spoof_flag": sample.get("spoof_flag", False),
            }
            classifier_output = {
                "label": sample.get("pattern", sample.get("label", "Unknown")),
                "confidence": sample.get("confidence", None),
            }

        # Normalize dicts (make them not None)
        classifier = classifier_output or {}
        minutiae = minutiae_output or {}
        matcher = matcher_output or {}

        # ----------------------------
        # NEW: Adapter-compact-shape compatibility:
        # get_fingerprint_payload returns minutiae_output as an adapter-compact dict:
        #   {"minutiae_summary": {...}, "minutiae_points_sample": [...]}
        # Normalize that into either a list of points or a summary dict so the rest of
        # the code (which expects 'minutiae_points' or top-level counts) will work.
        # ----------------------------
        try:
            if isinstance(minutiae, dict) and ("minutiae_points_sample" in minutiae or "minutiae_summary" in minutiae):
                pts = minutiae.get("minutiae_points_sample")
                summary = minutiae.get("minutiae_summary")
                if isinstance(pts, list) and pts:
                    # convert to a top-level points list so sampling/counting paths work
                    minutiae = {"minutiae_points": pts}
                elif isinstance(summary, dict):
                    # use the provided summary directly (contains minutiae_count, endings, bifurcations, quality_score etc.)
                    minutiae = summary.copy()
                else:
                    # fallback: leave as-is (empty compact)
                    minutiae = {}
        except Exception:
            # don't fail here; leave original minutiae
            pass

        # --- Robustness: if minutiae top-level is still empty, check common nested locations
        # (some pipelines put minutiae under matcher or results)
        if (not minutiae or (isinstance(minutiae, dict) and len(minutiae) == 0)) and isinstance(matcher, dict):
            nested_min = (
                matcher.get("minutiae")
                or matcher.get("minutiae_output")
                or matcher.get("results", {}).get("minutiae")
                or matcher.get("analysis", {}).get("minutiae")
            )
            if nested_min:
                minutiae = nested_min

        # Also check for classifier nested in matcher if classifier is missing
        if (not classifier or (isinstance(classifier, dict) and len(classifier) == 0)) and isinstance(matcher, dict):
            nested_cls = matcher.get("classifier") or matcher.get("classification") or matcher.get("pattern")
            if nested_cls:
                # If nested_cls is just a label string, convert
                if isinstance(nested_cls, str):
                    classifier = {"label": nested_cls}
                elif isinstance(nested_cls, dict):
                    classifier = nested_cls

        # --- Normalize classifier into simple fields
        label = (
            (classifier or {}).get("label")
            or (classifier or {}).get("classification")
            or (classifier or {}).get("pattern")
            or (classifier or {}).get("pattern_type")
            or "Unknown"
        )
        confidence = (classifier or {}).get("confidence") or (classifier or {}).get("score") or None
        classifier_info = {"label": label, "confidence": confidence}

        # --- Normalize minutiae into a lightweight list representation for deterministic rules
        minutiae_list: List[Dict[str, Any]] = []
        if isinstance(minutiae, dict):
            # Several formats may exist. Prefer explicit counts if present.
            try:
                mcount = (
                    minutiae.get("minutiae_count")
                    or (minutiae.get("minutiae") or {}).get("minutiae_count")
                    or None
                )
            except Exception:
                mcount = None

            endings = minutiae.get("endings") or minutiae.get("ridge_endings") or None
            bifs = minutiae.get("bifurcations") or minutiae.get("bifurcations_count") or None

            # If explicit point arrays exist under many possible keys, sample them
            points = None
            if isinstance(minutiae.get("minutiae_points"), list):
                points = minutiae.get("minutiae_points")
            elif isinstance(minutiae.get("minutiae"), dict) and isinstance(minutiae["minutiae"].get("minutiae_points"), list):
                points = minutiae["minutiae"].get("minutiae_points")
            elif isinstance(minutiae.get("minutiae"), list):
                points = minutiae.get("minutiae")

            if points:
                # Build a small list of typed minutiae for deterministic rules (we only need types)
                for p in points[:200]:  # cap for safety
                    if isinstance(p, dict) and "type" in p:
                        minutiae_list.append({"type": p.get("type")})
                # If we built a list by sampling, we can also set mcount accordingly
                if mcount is None:
                    mcount = len(points)

            # If no explicit points found but counts exist — synthesize type-only records (for rules)
            if not minutiae_list:
                if mcount is not None:
                    # If endings/bifs given, prefer those
                    if endings is not None or bifs is not None:
                        try:
                            endings_n = int(endings) if endings is not None else 0
                        except Exception:
                            endings_n = 0
                        try:
                            bifs_n = int(bifs) if bifs is not None else 0
                        except Exception:
                            bifs_n = 0
                        minutiae_list = ([{"type": "ending"}] * endings_n) + ([{"type": "bifurcation"}] * bifs_n)
                    else:
                        # synthesize with unknown types so rules can still count length
                        try:
                            mcount_i = int(mcount)
                        except Exception:
                            mcount_i = 0
                        minutiae_list = ([{"type": "unknown"}] * mcount_i)
        elif isinstance(minutiae, list):
            # Already a list of minutiae dicts
            minutiae_list = [{"type": m.get("type", "unknown")} for m in minutiae[:200]]
        else:
            minutiae_list = []

        # --- Normalize matcher into simple fields for rules
        matcher_info = {"match_score": None, "matched_id": None, "access_granted": None, "spoof_flag": False}
        if isinstance(matcher, dict):
            matcher_info["match_score"] = matcher.get("match_score") or matcher.get("score") or matcher.get("similarity")
            matcher_info["matched_id"] = matcher.get("matched_id") or matcher.get("matched_with") or matcher.get("id")
            matcher_info["access_granted"] = matcher.get("access_granted")
            matcher_info["spoof_flag"] = bool(matcher.get("spoof_flag", False))
            # threshold fallback if access not explicit
            if matcher_info["access_granted"] is None and matcher_info["match_score"] is not None:
                threshold = matcher.get("threshold")
                if threshold is not None:
                    try:
                        matcher_info["access_granted"] = float(matcher_info["match_score"]) >= float(threshold)
                    except Exception:
                        matcher_info["access_granted"] = None

        # --- If user asked a question, answer it (AI if requested and available)
        if user_question:
            return self._answer_question(classifier_info, minutiae_list, matcher_info, user_question, use_ai=explanation, max_length=max_length)

        # --- Summary / Explanation
        if explanation and self.client:
            try:
                return self._ai_summary(classifier_info, minutiae_list, matcher_info, max_length=max_length)
            except Exception:
                return summarize_all(classifier_info, minutiae_list, matcher_info)
        else:
            # Deterministic fallback
            parts = [
                classification_explanation(classifier_info),
                minutiae_explanation(minutiae_list if minutiae_list else minutiae),
                quality_checks(minutiae_list if minutiae_list else minutiae),
                matcher_explanation(matcher_info),
            ]
            if max_length == "short":
                return " ".join(p.split("\n")[0] for p in parts)
            return "\n\n".join(parts)

    def generate_response(
        self,
        step: str,
        input_payload: Dict[str, Any],
        mode: str = "short",
    ) -> str:
        """
        Public helper. step ∈ {"classify","minutiae","match","agent"}.
        input_payload can be {"classifier":..., "minutiae":..., "matcher":...} or similar.
        """
        # Normalize entry
        if set(input_payload.keys()) >= {"classifier", "minutiae", "matcher"}:
            cls = input_payload.get("classifier") or {}
            mino = input_payload.get("minutiae") or {}
            mat = input_payload.get("matcher") or {}
        else:
            cls = input_payload.get("classifier_output") or input_payload.get("classifier") or {}
            mino = input_payload.get("minutiae_output") or input_payload.get("minutiae") or {}
            mat = input_payload.get("matcher_output") or input_payload.get("matcher") or {}

        if step.lower().startswith("class"):
            question = "Explain the predicted fingerprint pattern and how confidence should be interpreted."
            return self.analyze({"classifier": cls, "minutiae": {}, "matcher": {}}, explanation=True, max_length=mode, user_question=question)
        if step.lower().startswith("minu"):
            question = "Explain the minutiae (ridge endings, bifurcations) and what their counts imply about quality."
            return self.analyze({"classifier": {}, "minutiae": mino, "matcher": {}}, explanation=True, max_length=mode, user_question=question)
        if step.lower().startswith("match"):
            question = "Explain the match result, thresholding, and whether access should be granted."
            return self.analyze({"classifier": {}, "minutiae": {}, "matcher": mat}, explanation=True, max_length=mode, user_question=question)
        # default: full agent view
        return self.analyze({"classifier": cls, "minutiae": mino, "matcher": mat}, explanation=True, max_length=mode)

    # -------------------------
    # Internals
    # -------------------------
    def _ai_summary(self, classifier_info, minutiae_list, matcher_info, max_length: str) -> str:
        """
        Use prompt builders (if available) to produce short/long explanation.
        """
        if not self.client:
            return summarize_all(classifier_info, minutiae_list, matcher_info)

        context = {
            "Classifier": classifier_info,
            "Minutiae": {
                "total": len(minutiae_list),
                "endings": sum(1 for m in minutiae_list if m.get("type") == "ending"),
                "bifurcations": sum(1 for m in minutiae_list if m.get("type") == "bifurcation"),
            },
            "Matcher": matcher_info,
        }

        if _PROMPTS_AVAILABLE:
            prompt = build_short_summary_prompt(context) if max_length == "short" else build_long_summary_prompt(context)
        else:
            prompt = (
                "You are an expert forensic fingerprint assistant.\n"
                f"Context:\nClassifier: {context['Classifier']}\nMinutiae: {context['Minutiae']}\nMatcher: {context['Matcher']}\n\n"
                + ("Provide a concise 2-3 sentence summary." if max_length == "short" else "Provide a detailed forensic explanation.")
            )

        return self._chat(prompt)

    def _answer_question(self, classifier_info, minutiae_list, matcher_info, question, use_ai: bool, max_length: str = "long"):
        if use_ai and self.client:
            context = {
                "Classifier": classifier_info,
                "Minutiae": {
                    "total": len(minutiae_list),
                    "endings": sum(1 for m in minutiae_list if m.get("type") == "ending"),
                    "bifurcations": sum(1 for m in minutiae_list if m.get("type") == "bifurcation"),
                },
                "Matcher": matcher_info,
            }

            if _PROMPTS_AVAILABLE:
                prompt = build_qa_prompt(context, question)
            else:
                prompt = (
                    "You are a forensic fingerprint analysis assistant.\n"
                    f"Context:\nClassifier: {context['Classifier']}\nMinutiae: {context['Minutiae']}\nMatcher: {context['Matcher']}\n\n"
                    f"Question: {question}\n"
                    + ("Answer in 2-3 concise sentences." if max_length == "short" else "")
                )

            try:
                return self._chat(prompt)
            except Exception:
                return summarize_all(classifier_info, minutiae_list, matcher_info)
        else:
            base = summarize_all(classifier_info, minutiae_list, matcher_info)
            if use_ai and self._openai_import_error:
                diag = f"\n\n[AI unavailable: {self._openai_import_error}]"
                return base + diag
            return base

    def _chat(self, prompt: str) -> str:
        """
        Minimal retry + timeout wrapper around OpenAI Chat Completions.
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                start = time.time()
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert in fingerprint biometrics. Never overclaim identity certainty."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=350 if "forensic" in prompt.lower() or "detailed" in prompt.lower() else 180,
                )
                if time.time() - start > self.timeout_s:
                    raise TimeoutError("OpenAI call exceeded timeout")
                try:
                    return resp.choices[0].message.content.strip()
                except Exception:
                    return str(resp)
            except Exception as e:
                last_err = e
                time.sleep(0.4 * (attempt + 1))
        raise RuntimeError(f"OpenAI chat failed after retries: {last_err}")
