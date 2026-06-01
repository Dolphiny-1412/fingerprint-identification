import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

_MODEL_CACHE: Dict[str, object] = {"model": None, "labels": None, "sk_model": None}


def _get_paths() -> Tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "model.h5"
    labels_path = base_dir / "label_map.json"
    return model_path, labels_path


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load image from path, convert to grayscale, resize, normalize to [0,1], add batch/channel dims.
    Returns a numpy array shaped (1, H, W, 1).
    """
    image = Image.open(image_path).convert("L")
    image = image.resize(target_size)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # channel
    image_array = np.expand_dims(image_array, axis=0)   # batch
    return image_array


def load_classifier():
    """
    Lazy-load and cache the Keras model and label map.
    Expects files next to this module: model.h5 and label_map.json
    """
    if _MODEL_CACHE["model"] is not None and _MODEL_CACHE["labels"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["labels"]

    model_path, labels_path = _get_paths()

    # Local import to keep import time fast when only calling preprocess
    labels: Tuple[str, ...]
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = tuple(json.load(f))
    else:
        labels = tuple()

    # Try TensorFlow model first
    try:
        from tensorflow import keras  # type: ignore
        if model_path.exists():
            model = keras.models.load_model(os.fspath(model_path))
            _MODEL_CACHE["model"] = model
            _MODEL_CACHE["labels"] = list(labels)
            return model, list(labels)
    except Exception:
        pass

    # Fallback to scikit-learn
    sk_path = Path(__file__).resolve().parent / "model_sklearn.joblib"
    if sk_path.exists():
        try:
            import joblib  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("joblib is required to load scikit-learn model.") from exc
        sk_model = joblib.load(os.fspath(sk_path))
        _MODEL_CACHE["sk_model"] = sk_model
        _MODEL_CACHE["labels"] = list(labels)
        return sk_model, list(labels)

    raise FileNotFoundError(
        f"No classifier model found. Expected one of: {model_path} (Keras) or {sk_path} (scikit-learn)."
    )


def classify_fingerprint(image_path: str) -> Dict[str, object]:
    """
    Run inference on a single fingerprint image path.
    Returns: {"label": str, "confidence": float}
    """
    model, label_list = load_classifier()
    # TensorFlow path: model has predict with 4D input
    if hasattr(model, "predict"):
        x = preprocess_image(image_path)
        preds = model.predict(x, verbose=0)
        preds = np.squeeze(preds)
        class_index = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = label_list[class_index]
        return {"label": label, "confidence": round(confidence, 6)}

    # Scikit-learn path: model expects flattened vector
    image = Image.open(image_path).convert("L").resize((128, 128))
    vec = (np.asarray(image, dtype=np.float32) / 255.0).flatten()[None, :]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        class_index = int(np.argmax(probs))
        confidence = float(np.max(probs))
    else:
        pred = int(model.predict(vec)[0])
        class_index = pred
        confidence = 1.0
    label = label_list[class_index] if label_list else str(class_index)
    return {"label": label, "confidence": round(confidence, 6)}


