"""
Person B - Fingerprint Classifier
=================================

Implements fingerprint pattern classification (Loop, Whorl, Arch)
with confidence scoring and model persistence.
"""

import os
import json
import numpy as np
import cv2
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from skimage import exposure

class FingerprintClassifier:
    """
    Fingerprint pattern classifier for Loop, Whorl, and Arch patterns
    """
    
    def __init__(self, model_path: str = "src/models/classifier.pkl", 
                 scaler_path: str = "src/models/scaler.pkl",
                 labels_path: str = "src/models/labels.json"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.labels_path = labels_path
        self.model = None
        self.scaler = None
        self.label_encoder = {"Loop": 0, "Whorl": 1, "Arch": 2}
        self.label_decoder = {0: "Loop", 1: "Whorl", 2: "Arch"}
        
        # Create models directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Load existing model if available
        self._load_model()
    
    def _load_model(self):
        """Load existing trained model and scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print("✅ Loaded existing classifier model")
            else:
                print("⚠️ No existing model found, will train new model")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            self.scaler = None
    
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract rich features from fingerprint image for classification
        - Intensity histogram + simple stats
        - Multi-scale LBP
        - GLCM (Haralick-like) texture props
        - Gabor filter bank summary
        - Gradient orientation histogram
        - FFT ring and wedge energies
        """
        # Standardize size and type
        image = cv2.resize(image, (256, 256))
        img = image.astype(np.float32)
        features: List[float] = []
        
        # 1) Histogram features (normalized)
        hist = cv2.calcHist([image], [0], None, [64], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        features.extend(hist.tolist())
        
        # 2) Simple statistics
        features.extend([
            float(np.mean(img)),
            float(np.std(img)),
            float(np.var(img)),
            float(np.median(img)),
            float(np.min(img)),
            float(np.max(img)),
            float(np.percentile(img, 25)),
            float(np.percentile(img, 75)),
        ])
        
        # 3) LBP (multi-scale, normalized histograms)
        lbp1 = local_binary_pattern(image, P=8, R=1, method='uniform')
        hist1, _ = np.histogram(lbp1.ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))
        hist1 = hist1.astype(np.float32) / (hist1.sum() + 1e-6)
        features.extend(hist1.tolist())
        
        lbp2 = local_binary_pattern(image, P=16, R=2, method='uniform')
        hist2, _ = np.histogram(lbp2.ravel(), bins=np.arange(0, 16 + 3), range=(0, 16 + 2))
        hist2 = hist2.astype(np.float32) / (hist2.sum() + 1e-6)
        features.extend(hist2.tolist())
        
        # 4) GLCM stats on 8-level quantized image
        img_q = (image // 32).astype(np.uint8)  # 0..7
        glcm = graycomatrix(img_q, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=8, symmetric=True, normed=True)
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            vals = graycoprops(glcm, prop)
            features.append(float(vals.mean()))
            features.append(float(vals.std()))
        
        # 5) Gabor filter bank (frequency, theta)
        img_f = img / 255.0
        thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        freqs = [0.1, 0.2, 0.3]
        for th in thetas:
            for fr in freqs:
                real, imag = gabor(img_f, frequency=fr, theta=th)
                mag = np.sqrt(real**2 + imag**2)
                features.append(float(np.mean(mag)))
                features.append(float(np.std(mag)))
        
        # 6) Gradient stats + orientation histogram
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        features.extend([
            float(np.mean(mag)),
            float(np.std(mag)),
            float(np.var(mag))
        ])
        # Orientation in [0, pi)
        orient = (np.arctan2(grad_y, grad_x) + np.pi) % np.pi
        hist_o, _ = np.histogram(orient.ravel(), bins=12, range=(0, np.pi), weights=mag.ravel())
        hist_o = hist_o.astype(np.float32) / (mag.sum() + 1e-6)
        features.extend(hist_o.tolist())
        
        # 7) FFT-based features: rings + wedges
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log1p(np.abs(f_shift))
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        rr = np.sqrt((x - cx)**2 + (y - cy)**2)
        # Rings
        for r in range(8, 72, 8):
            mask = (rr >= r - 4) & (rr < r + 4)
            if np.any(mask):
                features.append(float(np.mean(magnitude_spectrum[mask])))
            else:
                features.append(0.0)
        # Wedges (6 sectors)
        ang = (np.arctan2(y - cy, x - cx) + np.pi) % np.pi
        sector = np.pi / 6
        radial_mask = rr > 8
        for s in range(6):
            a0 = s * sector
            a1 = a0 + sector
            mask = radial_mask & (ang >= a0) & (ang < a1)
            if np.any(mask):
                features.append(float(np.mean(magnitude_spectrum[mask])))
            else:
                features.append(0.0)
        
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features
    
    def _create_synthetic_dataset(self, num_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic dataset for training when no real data is available.
        num_samples is per class.
        """
        print(f"🔄 Creating synthetic dataset for training... (~{num_samples * 3} images)")
        
        features_list: List[np.ndarray] = []
        labels_list: List[int] = []
        
        for class_name, class_id in self.label_encoder.items():
            for i in range(num_samples):
                # Generate base pattern
                if class_name == "Loop":
                    image = self._generate_loop_pattern()
                elif class_name == "Whorl":
                    image = self._generate_whorl_pattern()
                else:  # Arch
                    image = self._generate_arch_pattern()
                
                # Stronger augmentations
                image = self._add_variations(image)
                
                # Extract features
                features = self._extract_features(image)
                features_list.append(features)
                labels_list.append(class_id)
        
        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)
        return X, y
    
    def _generate_loop_pattern(self) -> np.ndarray:
        """Generate synthetic loop pattern (with randomness)."""
        image = np.zeros((256, 256), dtype=np.uint8)
        cx, cy = 128 + random.randint(-10, 10), 128 + random.randint(-10, 10)
        thickness = random.choice([1, 2, 3])
        a_scale = random.uniform(0.8, 1.2)
        b_scale = random.uniform(0.6, 1.1)
        phase = random.uniform(0, np.pi)
        for angle in np.linspace(0, 2*np.pi, 180):
            for r in range(20, 100, 2):
                x = int(cx + a_scale * r * np.cos(angle + 0.3*np.sin(angle + phase)))
                y = int(cy + b_scale * r * np.sin(angle))
                if 0 <= x < 256 and 0 <= y < 256:
                    for t in range(-thickness, thickness + 1):
                        yy = y + t
                        if 0 <= yy < 256:
                            image[yy, x] = 255
        return image
    
    def _generate_whorl_pattern(self) -> np.ndarray:
        """Generate synthetic whorl (spiral-like) pattern with randomness."""
        image = np.zeros((256, 256), dtype=np.uint8)
        cx, cy = 128 + random.randint(-8, 8), 128 + random.randint(-8, 8)
        thickness = random.choice([1, 2, 3])
        a = random.uniform(2.5, 4.5)
        turns = random.uniform(3.5, 5.0)
        for t in np.linspace(0, turns * np.pi, 360):
            r = 8 + a * t
            x = int(cx + r * np.cos(t))
            y = int(cy + r * np.sin(t))
            if 0 <= x < 256 and 0 <= y < 256:
                for k in range(-thickness, thickness + 1):
                    yy = y + k
                    if 0 <= yy < 256:
                        image[yy, x] = 255
        return image
    
    def _generate_arch_pattern(self) -> np.ndarray:
        """Generate synthetic arch pattern with variability."""
        image = np.zeros((256, 256), dtype=np.uint8)
        center = 128 + random.randint(-6, 6)
        width = random.randint(40, 70)
        height = random.randint(80, 120)
        thickness = random.choice([1, 2, 3])
        for y in range(40, 220):
            for x in range(40, 216):
                if abs(x - center) < width:
                    nx = (x - center) / max(width, 1)
                    arch_y = center - int(height * (1 - nx**2))
                    if abs(y - arch_y) <= thickness:
                        image[y, x] = 255
        return image
    
    def _add_variations(self, image: np.ndarray) -> np.ndarray:
        """Add stronger, diverse variations to synthetic images."""
        img = image.astype(np.float32)
        h, w = img.shape
        
        # Contrast/Brightness
        if random.random() < 0.9:
            alpha = random.uniform(0.8, 1.3)  # contrast
            beta = random.uniform(-20, 20)    # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            img = img.astype(np.float32)
        
        # CLAHE (adaptive histogram equalization)
        if random.random() < 0.7:
            clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 3.0), tileGridSize=(8, 8))
            img = clahe.apply(img.astype(np.uint8)).astype(np.float32)
        
        # Gaussian noise + speckle
        if random.random() < 0.9:
            sigma = random.uniform(5, 15)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 255)
        if random.random() < 0.5:
            speckle = img * (np.random.randn(*img.shape).astype(np.float32) * 0.02)
            img = np.clip(img + speckle, 0, 255)
        
        # Blur (gaussian or median)
        if random.random() < 0.8:
            if random.random() < 0.6:
                k = random.choice([3, 5, 7])
                img = cv2.GaussianBlur(img, (k, k), 0)
            else:
                k = random.choice([3, 5])
                img = cv2.medianBlur(img.astype(np.uint8), k).astype(np.float32)
        
        # Affine: rotation + scale + translation
        angle = random.uniform(-25, 25)
        scale = random.uniform(0.85, 1.15)
        tx = random.randint(-10, 10)
        ty = random.randint(-10, 10)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        M[:, 2] += (tx, ty)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Perspective warp
        if random.random() < 0.5:
            jitter = 12
            src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            dst = np.float32([
                [random.randint(0, jitter), random.randint(0, jitter)],
                [w - 1 - random.randint(0, jitter), random.randint(0, jitter)],
                [w - 1 - random.randint(0, jitter), h - 1 - random.randint(0, jitter)],
                [random.randint(0, jitter), h - 1 - random.randint(0, jitter)],
            ])
            P = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Elastic deformation
        if random.random() < 0.35:
            alpha = random.uniform(6, 10)
            sigma = random.uniform(6, 10)
            dx = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
            dy = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)
            img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Morphology (erode/dilate)
        if random.random() < 0.6:
            ksize = random.choice([2, 3])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            if random.random() < 0.5:
                img = cv2.erode(img, kernel, iterations=1)
            else:
                img = cv2.dilate(img, kernel, iterations=1)
        
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def train(self, data_dir: str, save_model: bool = True) -> Dict:
        """
        Train the classifier model on real labeled data only.
        Expects data_dir to contain subfolders: Loop, Whorl, Arch.
        Tries multiple models (RF, SVM-RBF, GradientBoosting) and picks the best by macro F1.
        """
        print("🚀 Training fingerprint classifier (real data only)...")
        
        # Load real data only
        if not data_dir or not os.path.exists(data_dir):
            raise ValueError("Real training data directory is required and must exist. Expected structure: <data_dir>/{Loop,Whorl,Arch}")
        X, y = self._load_real_data(data_dir)
        print(f"✅ Loaded real training data: {len(X)} samples")
        
        # Split data (handle rare classes without stratify)
        class_counts_all = np.bincount(y, minlength=3)
        do_stratify = np.all(class_counts_all >= 2)
        stratify_arg = y if do_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_arg
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Candidate models
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300, max_depth=None, random_state=42, class_weight='balanced', n_jobs=-1
            ),
            'SVM_RBF': SVC(C=3.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
        }
        
        # Optional sample weights for models that don't support class_weight
        class_counts = np.bincount(y_train, minlength=3)
        class_weights = {}
        for cls, count in enumerate(class_counts):
            if count > 0:
                class_weights[cls] = float(len(y_train) / (max(1, int((y_train.max() + 1))) * count))
            else:
                class_weights[cls] = 0.0
        sample_weight = np.array([class_weights.get(int(yy), 0.0) for yy in y_train], dtype=np.float32)
        
        best_name = None
        best_model = None
        best_macro_f1 = -1.0
        best_bal_acc = -1.0
        per_model_metrics: Dict[str, Dict[str, float]] = {}
        
        for name, model in models.items():
            if name == 'GradientBoosting':
                model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            acc = float(np.mean(y_pred == y_test))
            per_model_metrics[name] = {
                'accuracy': acc,
                'macro_f1': float(macro_f1),
                'balanced_accuracy': float(bal_acc),
            }
            if (macro_f1 > best_macro_f1) or (np.isclose(macro_f1, best_macro_f1) and bal_acc > best_bal_acc):
                best_macro_f1 = macro_f1
                best_bal_acc = bal_acc
                best_name = name
                best_model = model
        
        # Use best model
        self.model = best_model
        
        # Evaluate best model in detail
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        y_pred = self.model.predict(X_test_scaled)
        class_names = [self.label_decoder[i] for i in range(3)]
        report = classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        
        # Save artifacts
        if save_model:
            self._save_model()
            self._save_evaluation_plots(cm, class_names)
        
        results = {
            'best_model': best_name,
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'macro_f1': float(best_macro_f1),
            'balanced_accuracy': float(best_bal_acc),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'per_model_metrics': per_model_metrics,
            'model_saved': save_model
        }
        
        print(f"✅ Training complete! Best model: {best_name}")
        print(f"   Train Acc: {train_score:.3f} | Test Acc: {test_score:.3f}")
        print(f"   Macro F1: {best_macro_f1:.3f} | Balanced Acc: {best_bal_acc:.3f}")
        
        return results
    
    def _load_real_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load real training data from directory"""
        features_list = []
        labels_list = []
        
        # Look for subdirectories named after pattern types
        for pattern_name in ["Loop", "Whorl", "Arch"]:
            pattern_dir = os.path.join(data_dir, pattern_name)
            if os.path.exists(pattern_dir):
                for filename in os.listdir(pattern_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        image_path = os.path.join(pattern_dir, filename)
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            features = self._extract_features(image)
                            features_list.append(features)
                            labels_list.append(self.label_encoder[pattern_name])
        
        if not features_list:
            raise ValueError("No valid training images found")
        
        return np.array(features_list), np.array(labels_list)
    
    def _save_model(self):
        """Save trained model and scaler"""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        # Save label mapping
        with open(self.labels_path, 'w') as f:
            json.dump(self.label_encoder, f, indent=2)
        
        print(f"💾 Model saved to {self.model_path}")
    
    def _save_evaluation_plots(self, cm: np.ndarray, class_names: List[str]):
        """Save evaluation plots"""
        os.makedirs("reports", exist_ok=True)
        
        # Confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Fingerprint Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Evaluation plots saved to reports/")
    
    def classify_fingerprint(self, image_path: str) -> Dict:
        """
        Classify a fingerprint image
        
        Args:
            image_path: Path to fingerprint image
            
        Returns:
            Classification result dictionary
        """
        if self.model is None or self.scaler is None:
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "error": "Model not trained"
            }
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return {
                    "label": "Unknown",
                    "confidence": 0.0,
                    "error": "Could not load image"
                }
            
            # Extract features
            features = self._extract_features(image)
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = int(self.model.predict(features_scaled)[0])
            proba_raw = self.model.predict_proba(features_scaled)[0]
            # Map probabilities to full 3-class vector using model.classes_
            full_proba = np.zeros(3, dtype=np.float32)
            if hasattr(self.model, "classes_"):
                classes = list(self.model.classes_)
                for idx, cls in enumerate(classes):
                    if 0 <= int(cls) <= 2:
                        full_proba[int(cls)] = float(proba_raw[idx])
            else:
                # Fallback: if classes_ not available, assume order 0..len-1
                for i in range(min(3, len(proba_raw))):
                    full_proba[i] = float(proba_raw[i])
            
            # Get label and confidence
            label = self.label_decoder.get(prediction, "Unknown")
            confidence = float(np.max(full_proba))
            
            return {
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    "Loop": float(full_proba[0]),
                    "Whorl": float(full_proba[1]),
                    "Arch": float(full_proba[2])
                }
            }
            
        except Exception as e:
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "error": str(e)
            }

# Global classifier instance
_classifier_instance = None

def get_classifier() -> FingerprintClassifier:
    """Get or create global classifier instance"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = FingerprintClassifier()
    return _classifier_instance

def classify_fingerprint(image_path: str) -> Dict:
    """
    Main function to classify a fingerprint image
    
    Args:
        image_path: Path to fingerprint image
        
    Returns:
        Classification result dictionary
    """
    classifier = get_classifier()
    return classifier.classify_fingerprint(image_path)

def train_classifier(data_dir: str) -> Dict:
    """
    Train the classifier model on real labeled data only.
    Args:
        data_dir: Directory containing subfolders Loop/Whorl/Arch
    Returns:
        Training results
    """
    if not data_dir:
        raise ValueError("data_dir is required")
    classifier = get_classifier()
    return classifier.train(data_dir)

if __name__ == "__main__":
    print("Usage: python train_real.py --data-dir <path_with_Loop_Whorl_Arch>")
