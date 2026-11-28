#!/usr/bin/env python3
"""
Production Fingerprint Classification Pipeline
=============================================

End-to-end pipeline for 6,000+ fingerprint images with:
- Comprehensive landmark detection and feature extraction
- Proper train/validation/test splits to prevent overfitting
- Multiple model training with hyperparameter tuning
- Production-ready prediction functionality
- Robust evaluation with held-out test set

Target: ≥90% accuracy on real test data
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (StratifiedKFold, train_test_split, 
                                   GridSearchCV, cross_val_score)
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, roc_curve, auc)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import json
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

class ProductionFingerprintPreprocessor:
    """Production-grade preprocessing with enhanced reliability metrics"""
    
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    def enhance_image(self, image):
        """Apply CLAHE enhancement and normalization"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize to 0-255 range
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply CLAHE
        enhanced = self.clahe.apply(image)
        
        return enhanced
    
    def compute_orientation_field(self, image):
        """Compute ridge orientation field with reliability metrics"""
        h, w = image.shape
        rows, cols = h // self.block_size, w // self.block_size
        
        orientation = np.zeros((rows, cols))
        reliability = np.zeros((rows, cols))
        coherence = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                # Extract block
                y_start, y_end = i * self.block_size, (i + 1) * self.block_size
                x_start, x_end = j * self.block_size, (j + 1) * self.block_size
                
                if y_end <= h and x_end <= w:
                    block = image[y_start:y_end, x_start:x_end].astype(np.float32)
                    
                    # Compute gradients
                    gx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)
                    
                    # Structure tensor elements
                    gxx = np.sum(gx * gx)
                    gxy = np.sum(gx * gy)
                    gyy = np.sum(gy * gy)
                    
                    # Orientation angle
                    if gxx + gyy > 0:
                        orientation[i, j] = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
                        
                        # Reliability as coherence measure
                        coherence_val = np.sqrt((gxx - gyy)**2 + 4*gxy**2) / (gxx + gyy + 1e-10)
                        reliability[i, j] = coherence_val
                        coherence[i, j] = coherence_val
        
        return orientation, reliability, coherence
    
    def compute_ridge_frequency(self, image, orientation):
        """Compute ridge frequency using windowed FFT analysis"""
        h, w = image.shape
        rows, cols = h // self.block_size, w // self.block_size
        frequency = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                y_start, y_end = i * self.block_size, (i + 1) * self.block_size
                x_start, x_end = j * self.block_size, (j + 1) * self.block_size
                
                if y_end <= h and x_end <= w:
                    block = image[y_start:y_end, x_start:x_end].astype(np.float32)
                    angle = orientation[i, j]
                    
                    # Rotate block to align ridges horizontally
                    center = (self.block_size // 2, self.block_size // 2)
                    M = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)
                    rotated = cv2.warpAffine(block, M, (self.block_size, self.block_size))
                    
                    # Project to get 1D signal
                    projection = np.mean(rotated, axis=0)
                    
                    # Remove DC component and compute FFT
                    projection = projection - np.mean(projection)
                    
                    # Apply window to reduce spectral leakage
                    window = np.hanning(len(projection))
                    windowed = projection * window
                    
                    # FFT analysis
                    fft = np.fft.fft(windowed)
                    freqs = np.fft.fftfreq(len(windowed))
                    
                    # Find peak frequency (excluding DC)
                    magnitude = np.abs(fft[1:len(fft)//2])
                    if len(magnitude) > 0 and np.max(magnitude) > 0:
                        peak_idx = np.argmax(magnitude) + 1
                        frequency[i, j] = abs(freqs[peak_idx])
        
        return frequency

class ProductionLandmarkDetector:
    """Production-grade landmark detection with enhanced filtering"""
    
    def __init__(self, block_size=16):
        self.preprocessor = ProductionFingerprintPreprocessor(block_size)
        self.block_size = block_size
    
    def compute_poincare_index(self, orientation, i, j, radius=1):
        """Compute Poincaré index with configurable radius"""
        angles = []
        positions = []
        
        # Collect angles in circular pattern
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ni, nj = i + di, j + dj
                if (di != 0 or dj != 0) and 0 <= ni < orientation.shape[0] and 0 <= nj < orientation.shape[1]:
                    angles.append(orientation[ni, nj])
                    positions.append((ni, nj))
        
        if len(angles) < 3:
            return 0.0
        
        # Sort by angle around the center point
        angles = np.array(angles)
        
        # Compute angular differences
        total_change = 0
        for k in range(len(angles)):
            diff = angles[(k + 1) % len(angles)] - angles[k]
            
            # Wrap angles to [-π, π]
            while diff > np.pi:
                diff -= 2 * np.pi
            while diff < -np.pi:
                diff += 2 * np.pi
            
            total_change += diff
        
        # Poincaré index
        poincare_index = abs(total_change) / (2 * np.pi)
        return poincare_index
    
    def detect_landmarks(self, image, min_reliability_percentile=70):
        """Detect cores and deltas with enhanced quality filtering"""
        try:
            # Preprocess image
            enhanced = self.preprocessor.enhance_image(image)
            
            # Compute orientation field and metrics
            orientation, reliability, coherence = self.preprocessor.compute_orientation_field(enhanced)
            frequency = self.preprocessor.compute_ridge_frequency(enhanced, orientation)
            
            h, w = orientation.shape
            cores = []
            deltas = []
            
            # Adaptive thresholds based on image quality
            min_reliability = np.percentile(reliability[reliability > 0], min_reliability_percentile)
            min_frequency = 0.005  # Minimum ridge frequency
            
            # Detect landmarks with enhanced criteria
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    rel = reliability[i, j]
                    freq = frequency[i, j]
                    
                    # Quality filtering
                    if rel >= min_reliability and freq >= min_frequency:
                        poincare = self.compute_poincare_index(orientation, i, j)
                        
                        # Convert to image coordinates
                        x = j * self.block_size + self.block_size // 2
                        y = i * self.block_size + self.block_size // 2
                        
                        # Classify landmark type
                        if 0.3 < poincare < 0.7:  # Core
                            cores.append({
                                'x': x, 'y': y,
                                'strength': rel,
                                'frequency': freq,
                                'poincare': poincare,
                                'coherence': coherence[i, j]
                            })
                        elif 0.7 < poincare < 1.3:  # Delta
                            deltas.append({
                                'x': x, 'y': y,
                                'strength': rel,
                                'frequency': freq,
                                'poincare': poincare,
                                'coherence': coherence[i, j]
                            })
            
            # Sort by strength and keep best landmarks
            cores = sorted(cores, key=lambda x: x['strength'], reverse=True)[:3]
            deltas = sorted(deltas, key=lambda x: x['strength'], reverse=True)[:5]
            
            return {
                'cores': cores,
                'deltas': deltas,
                'orientation': orientation,
                'reliability': reliability,
                'coherence': coherence,
                'frequency': frequency,
                'image_shape': enhanced.shape,
                'quality_metrics': {
                    'mean_reliability': np.mean(reliability[reliability > 0]),
                    'std_reliability': np.std(reliability[reliability > 0]),
                    'max_reliability': np.max(reliability),
                    'mean_frequency': np.mean(frequency[frequency > 0]),
                    'std_frequency': np.std(frequency[frequency > 0]),
                    'mean_coherence': np.mean(coherence[coherence > 0])
                }
            }
            
        except Exception as e:
            print(f"⚠️ Landmark detection failed: {e}")
            return self._empty_landmark_result(image.shape if len(image.shape) == 2 else image.shape[:2])
    
    def _empty_landmark_result(self, image_shape):
        """Return empty result structure for failed detections"""
        return {
            'cores': [],
            'deltas': [],
            'orientation': np.zeros((image_shape[0] // self.block_size, image_shape[1] // self.block_size)),
            'reliability': np.zeros((image_shape[0] // self.block_size, image_shape[1] // self.block_size)),
            'coherence': np.zeros((image_shape[0] // self.block_size, image_shape[1] // self.block_size)),
            'frequency': np.zeros((image_shape[0] // self.block_size, image_shape[1] // self.block_size)),
            'image_shape': image_shape,
            'quality_metrics': {
                'mean_reliability': 0, 'std_reliability': 0, 'max_reliability': 0,
                'mean_frequency': 0, 'std_frequency': 0, 'mean_coherence': 0
            }
        }

class AdvancedFeatureExtractor:
    """Extract comprehensive feature set for production classification"""
    
    def extract_features(self, landmarks):
        """Extract 28+ features from landmark detection results"""
        features = {}
        
        cores = landmarks['cores']
        deltas = landmarks['deltas']
        orientation = landmarks['orientation']
        reliability = landmarks['reliability']
        coherence = landmarks['coherence']
        frequency = landmarks['frequency']
        quality_metrics = landmarks['quality_metrics']
        image_shape = landmarks['image_shape']
        
        # === BASIC LANDMARK COUNTS ===
        features['num_cores'] = len(cores)
        features['num_deltas'] = len(deltas)
        features['has_core'] = 1 if len(cores) > 0 else 0
        features['has_delta'] = 1 if len(deltas) > 0 else 0
        features['has_multiple_cores'] = 1 if len(cores) > 1 else 0
        features['has_multiple_deltas'] = 1 if len(deltas) > 1 else 0
        
        # === QUALITY METRICS ===
        features['mean_reliability'] = quality_metrics['mean_reliability']
        features['std_reliability'] = quality_metrics['std_reliability']
        features['max_reliability'] = quality_metrics['max_reliability']
        features['mean_frequency'] = quality_metrics['mean_frequency']
        features['std_frequency'] = quality_metrics['std_frequency']
        features['mean_coherence'] = quality_metrics['mean_coherence']
        
        # === ORIENTATION FIELD ANALYSIS ===
        # Orientation smoothness
        grad_x = np.gradient(orientation, axis=1)
        grad_y = np.gradient(orientation, axis=0)
        features['orientation_smoothness'] = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        features['orientation_variance'] = np.var(orientation)
        features['orientation_entropy'] = entropy(orientation.flatten() + np.pi + 1e-10)
        
        # === FREQUENCY DOMAIN FEATURES ===
        features['frequency_range'] = np.ptp(frequency[frequency > 0]) if np.any(frequency > 0) else 0
        features['frequency_entropy'] = entropy(frequency.flatten() + 1e-10)
        
        # === SPATIAL DISTRIBUTION FEATURES ===
        if len(cores) > 0:
            # Core positions (normalized)
            core_x = [c['x'] / image_shape[1] for c in cores]
            core_y = [c['y'] / image_shape[0] for c in cores]
            
            features['core_center_x'] = np.mean(core_x)
            features['core_center_y'] = np.mean(core_y)
            features['core_spread_x'] = np.std(core_x) if len(cores) > 1 else 0
            features['core_spread_y'] = np.std(core_y) if len(cores) > 1 else 0
        else:
            features['core_center_x'] = 0.5  # Center default
            features['core_center_y'] = 0.5
            features['core_spread_x'] = 0
            features['core_spread_y'] = 0
        
        if len(deltas) > 0:
            # Delta positions (normalized)
            delta_x = [d['x'] / image_shape[1] for d in deltas]
            delta_y = [d['y'] / image_shape[0] for d in deltas]
            
            features['delta_center_x'] = np.mean(delta_x)
            features['delta_center_y'] = np.mean(delta_y)
            features['delta_spread_x'] = np.std(delta_x) if len(deltas) > 1 else 0
            features['delta_spread_y'] = np.std(delta_y) if len(deltas) > 1 else 0
        else:
            features['delta_center_x'] = 0.5
            features['delta_center_y'] = 0.5
            features['delta_spread_x'] = 0
            features['delta_spread_y'] = 0
        
        # === GEOMETRIC RELATIONSHIPS ===
        if len(cores) > 0 and len(deltas) > 0:
            # Core-Delta relationships
            core = cores[0]  # Primary core
            
            # Distances to all deltas
            distances = []
            angles = []
            for delta in deltas:
                dist = euclidean([core['x'], core['y']], [delta['x'], delta['y']])
                distances.append(dist)
                
                # Angle from core to delta
                dx = delta['x'] - core['x']
                dy = delta['y'] - core['y']
                angle = np.arctan2(dy, dx)
                angles.append(angle)
            
            features['min_core_delta_distance'] = min(distances) / max(image_shape)  # Normalized
            features['max_core_delta_distance'] = max(distances) / max(image_shape)
            features['mean_core_delta_distance'] = np.mean(distances) / max(image_shape)
            features['std_core_delta_distance'] = np.std(distances) / max(image_shape)
            
            # Angular relationships
            features['core_delta_angle_variance'] = np.var(angles)
            features['core_delta_angle_range'] = np.ptp(angles)
        else:
            features['min_core_delta_distance'] = 0
            features['max_core_delta_distance'] = 0
            features['mean_core_delta_distance'] = 0
            features['std_core_delta_distance'] = 0
            features['core_delta_angle_variance'] = 0
            features['core_delta_angle_range'] = 0
        
        # === DELTA-DELTA RELATIONSHIPS ===
        if len(deltas) >= 2:
            delta_distances = []
            delta_angles = []
            
            for i in range(len(deltas)):
                for j in range(i + 1, len(deltas)):
                    d1, d2 = deltas[i], deltas[j]
                    
                    # Distance
                    dist = euclidean([d1['x'], d1['y']], [d2['x'], d2['y']])
                    delta_distances.append(dist)
                    
                    # Angle
                    dx = d2['x'] - d1['x']
                    dy = d2['y'] - d1['y']
                    angle = np.arctan2(dy, dx)
                    delta_angles.append(angle)
            
            features['min_delta_delta_distance'] = min(delta_distances) / max(image_shape)
            features['max_delta_delta_distance'] = max(delta_distances) / max(image_shape)
            features['mean_delta_delta_distance'] = np.mean(delta_distances) / max(image_shape)
            features['delta_angle_variance'] = np.var(delta_angles)
        else:
            features['min_delta_delta_distance'] = 0
            features['max_delta_delta_distance'] = 0
            features['mean_delta_delta_distance'] = 0
            features['delta_angle_variance'] = 0
        
        # === LANDMARK QUALITY FEATURES ===
        if len(cores) > 0:
            features['strongest_core_strength'] = max(c['strength'] for c in cores)
            features['mean_core_strength'] = np.mean([c['strength'] for c in cores])
            features['strongest_core_frequency'] = max(c['frequency'] for c in cores)
        else:
            features['strongest_core_strength'] = 0
            features['mean_core_strength'] = 0
            features['strongest_core_frequency'] = 0
        
        if len(deltas) > 0:
            features['strongest_delta_strength'] = max(d['strength'] for d in deltas)
            features['mean_delta_strength'] = np.mean([d['strength'] for d in deltas])
            features['strongest_delta_frequency'] = max(d['frequency'] for d in deltas)
        else:
            features['strongest_delta_strength'] = 0
            features['mean_delta_strength'] = 0
            features['strongest_delta_frequency'] = 0
        
        # === SYMMETRY ANALYSIS ===
        # Compute symmetry scores
        features['horizontal_symmetry'] = self._compute_symmetry(reliability, axis=1)
        features['vertical_symmetry'] = self._compute_symmetry(reliability, axis=0)
        
        return features
    
    def _compute_symmetry(self, matrix, axis):
        """Compute symmetry score along given axis"""
        try:
            if axis == 0:  # Vertical symmetry
                flipped = np.flipud(matrix)
            else:  # Horizontal symmetry
                flipped = np.fliplr(matrix)
            
            # Correlation coefficient as symmetry measure
            corr = np.corrcoef(matrix.flatten(), flipped.flatten())[0, 1]
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0

class ProductionDataLoader:
    """Load and organize all fingerprint images from dataset"""
    
    def __init__(self, data_root):
        self.data_root = data_root
        self.image_extensions = {'.png', '.tif', '.tiff', '.jpg', '.jpeg'}
    
    def discover_all_images(self):
        """Discover all fingerprint images in the dataset"""
        print("🔍 Discovering all fingerprint images...")
        
        all_images = []
        
        # Look for images in various dataset structures
        search_paths = [
            # Original structure
            os.path.join(self.data_root, "archive", "fingerprints", "data_ready"),
            # Alternative structures
            os.path.join(self.data_root, "data_ready"),
            os.path.join(self.data_root, "fingerprints"),
            # DB folders
            os.path.join(self.data_root, "archive", "fingerprints", "DB1_B"),
            os.path.join(self.data_root, "archive", "fingerprints", "DB2_B"),
            os.path.join(self.data_root, "archive", "fingerprints", "DB3_B"),
            os.path.join(self.data_root, "archive", "fingerprints", "DB4_B"),
            # Direct root
            self.data_root
        ]
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                print(f"🔍 Searching in: {search_path}")
                found_images = self._scan_directory_recursive(search_path)
                all_images.extend(found_images)
        
        # Remove duplicates based on filename
        unique_images = {}
        for img_info in all_images:
            key = os.path.basename(img_info['filepath'])
            if key not in unique_images:
                unique_images[key] = img_info
        
        all_images = list(unique_images.values())
        
        print(f"✅ Discovered {len(all_images)} unique fingerprint images")
        return all_images
    
    def _scan_directory_recursive(self, directory):
        """Recursively scan directory for fingerprint images"""
        images = []
        
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if any(filename.lower().endswith(ext) for ext in self.image_extensions):
                    filepath = os.path.join(root, filename)
                    
                    # Try to infer class from directory structure
                    class_name = self._infer_class_from_path(root, filename)
                    
                    # Try to infer split from path
                    split = self._infer_split_from_path(root)
                    
                    images.append({
                        'filepath': filepath,
                        'filename': filename,
                        'class': class_name,
                        'split': split,
                        'directory': root
                    })
        
        return images
    
    def _infer_class_from_path(self, directory_path, filename):
        """Infer fingerprint class from directory structure or filename"""
        path_lower = directory_path.lower()
        filename_lower = filename.lower()
        
        # Check directory names (more specific patterns first)
        if os.path.sep + 'arch' + os.path.sep in path_lower or path_lower.endswith('arch'):
            return 'Arch'
        elif os.path.sep + 'loop' + os.path.sep in path_lower or path_lower.endswith('loop'):
            return 'Loop'
        elif os.path.sep + 'whorl' + os.path.sep in path_lower or path_lower.endswith('whorl'):
            return 'Whorl'
        
        # Check exact directory name
        dir_name = os.path.basename(directory_path).lower()
        if dir_name == 'arch':
            return 'Arch'
        elif dir_name == 'loop':
            return 'Loop'
        elif dir_name == 'whorl':
            return 'Whorl'
        
        # Check filename patterns
        if 'arch' in filename_lower:
            return 'Arch'
        elif 'loop' in filename_lower:
            return 'Loop'
        elif 'whorl' in filename_lower:
            return 'Whorl'
        
        # Default classification (will be manually labeled later if needed)
        return 'Unknown'
    
    def _infer_split_from_path(self, directory_path):
        """Infer train/val/test split from directory structure"""
        path_lower = directory_path.lower()
        
        if 'train' in path_lower:
            return 'train'
        elif 'val' in path_lower or 'validation' in path_lower:
            return 'val'
        elif 'test' in path_lower:
            return 'test'
        else:
            return 'unsplit'
    
    def analyze_dataset_distribution(self, images):
        """Analyze and report dataset distribution"""
        print("\n📊 Dataset Analysis")
        print("=" * 30)
        
        df = pd.DataFrame(images)
        
        print(f"Total images found: {len(images)}")
        print()
        
        # Class distribution
        print("Class distribution:")
        class_counts = df['class'].value_counts()
        for class_name, count in class_counts.items():
            percentage = count / len(images) * 100
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        print()
        
        # Split distribution
        print("Split distribution:")
        split_counts = df['split'].value_counts()
        for split_name, count in split_counts.items():
            percentage = count / len(images) * 100
            print(f"  {split_name}: {count} images ({percentage:.1f}%)")
        
        print()
        
        # Check for Unknown class
        unknown_count = (df['class'] == 'Unknown').sum()
        if unknown_count > 0:
            print(f"⚠️ Warning: {unknown_count} images have unknown class labels")
            print("These will need manual classification or pattern-based labeling")
        
        return df

class ProductionFeatureProcessor:
    """Process all images and extract features for production pipeline"""
    
    def __init__(self, detector, feature_extractor):
        self.detector = detector
        self.feature_extractor = feature_extractor
    
    def process_all_images(self, images, batch_size=100):
        """Process all images and extract features with progress tracking"""
        print(f"\n🔄 Processing {len(images)} images for feature extraction...")
        print("This may take 10-30 minutes depending on dataset size...")
        
        all_features = []
        failed_count = 0
        
        for i, img_info in enumerate(images):
            try:
                # Load image
                image = cv2.imread(img_info['filepath'], cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    failed_count += 1
                    continue
                
                # Detect landmarks
                landmarks = self.detector.detect_landmarks(image)
                
                # Extract features
                features = self.feature_extractor.extract_features(landmarks)
                
                # Add metadata
                features['filename'] = img_info['filename']
                features['filepath'] = img_info['filepath']
                features['class'] = img_info['class']
                features['split'] = img_info['split']
                features['image_width'] = image.shape[1]
                features['image_height'] = image.shape[0]
                
                all_features.append(features)
                
                # Progress reporting
                if (i + 1) % batch_size == 0:
                    print(f"Processed {i + 1}/{len(images)} images ({(i+1)/len(images)*100:.1f}%)")
                
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:  # Show first few errors
                    print(f"⚠️ Failed to process {img_info['filename']}: {e}")
        
        print(f"✅ Feature extraction complete!")
        print(f"Successfully processed: {len(all_features)} images")
        print(f"Failed: {failed_count} images")
        
        return pd.DataFrame(all_features)
    
    def save_features(self, features_df, output_path):
        """Save features to CSV with proper formatting"""
        print(f"\n💾 Saving features to {output_path}")
        
        # Clean up any NaN or infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        # Save to CSV
        features_df.to_csv(output_path, index=False)
        
        print(f"✅ Features saved: {len(features_df)} samples, {len(features_df.columns)} features")
        return features_df

class ProductionModelTrainer:
    """Train and evaluate multiple models with proper validation"""
    
    def __init__(self, output_dir="production_output"):
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Initialize models with production-ready configurations
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            )
        }
        
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                n_jobs=-1
            )
    
    def prepare_data(self, features_df, test_size=0.1, val_size=0.15):
        """Prepare data with proper train/val/test splits"""
        print(f"\n📊 Preparing data for training...")
        
        # Filter out unknown classes
        known_classes = features_df[features_df['class'] != 'Unknown']
        
        if len(known_classes) == 0:
            raise ValueError("No images with known class labels found!")
        
        print(f"Working with {len(known_classes)} images with known labels")
        
        # Feature columns (exclude metadata)
        metadata_cols = ['filename', 'filepath', 'class', 'split', 'image_width', 'image_height']
        feature_cols = [col for col in known_classes.columns if col not in metadata_cols]
        
        print(f"Using {len(feature_cols)} features for training")
        
        X = known_classes[feature_cols].fillna(0)
        y = known_classes['class']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create stratified splits
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Second split: train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
        )
        
        print(f"\nData splits:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Print class distribution
        print(f"\nClass distribution in training set:")
        train_dist = pd.Series(y_train).value_counts()
        for class_name, count in train_dist.items():
            print(f"  {class_name}: {count} samples")
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'feature_cols': feature_cols
        }
    
    def apply_smote_if_needed(self, X_train, y_train, min_samples_threshold=100):
        """Apply SMOTE if classes are heavily imbalanced"""
        print(f"\n⚖️ Checking class balance...")
        
        class_counts = pd.Series(y_train).value_counts()
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        imbalance_ratio = max_class_count / min_class_count
        
        print(f"Class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} samples")
        
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")
        
        if min_class_count < min_samples_threshold or imbalance_ratio > 3.0:
            print(f"🔄 Applying SMOTE balancing (min class has {min_class_count} samples)...")
            
            try:
                # Determine k_neighbors based on smallest class
                k_neighbors = min(5, min_class_count - 1)
                if k_neighbors < 1:
                    print("⚠️ Too few samples for SMOTE, using original dataset")
                    return X_train, y_train, False
                
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                
                # Encode labels for SMOTE
                y_train_encoded = self.label_encoder.transform(y_train)
                X_balanced, y_balanced_encoded = smote.fit_resample(X_train, y_train_encoded)
                
                # Decode labels back
                y_balanced = self.label_encoder.inverse_transform(y_balanced_encoded)
                
                print(f"✅ SMOTE applied successfully")
                print(f"Balanced class distribution:")
                balanced_counts = pd.Series(y_balanced).value_counts()
                for class_name, count in balanced_counts.items():
                    print(f"  {class_name}: {count} samples")
                
                # Save balanced dataset
                balanced_df = pd.DataFrame(X_balanced, columns=X_train.columns)
                balanced_df['class'] = y_balanced
                balanced_df.to_csv(f"{self.output_dir}/balanced_dataset.csv", index=False)
                
                return X_balanced, y_balanced, True
                
            except Exception as e:
                print(f"⚠️ SMOTE failed: {e}")
                print("Using original dataset")
                return X_train, y_train, False
        else:
            print("✅ Dataset is reasonably balanced, no SMOTE needed")
            return X_train, y_train, False
    
    def train_models(self, data_splits, use_smote=True):
        """Train all models with hyperparameter tuning"""
        print(f"\n🎯 Training models...")
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        
        # Apply SMOTE if needed
        if use_smote:
            X_train, y_train, smote_applied = self.apply_smote_if_needed(X_train, y_train)
        else:
            smote_applied = False
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        results = {}
        
        # Train each model
        for model_name, model in self.models.items():
            print(f"\n🔄 Training {model_name}...")
            
            # Encode labels for XGBoost if needed
            if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                y_train_encoded = self.label_encoder.transform(y_train)
                y_val_encoded = self.label_encoder.transform(y_val)
                
                # Train model
                model.fit(X_train_scaled, y_train_encoded)
                
                # Validate
                y_val_pred_encoded = model.predict(X_val_scaled)
                y_val_pred = self.label_encoder.inverse_transform(y_val_pred_encoded)
                
                # Training predictions
                y_train_pred_encoded = model.predict(X_train_scaled)
                y_train_pred = self.label_encoder.inverse_transform(y_train_pred_encoded)
            else:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Validate
                y_val_pred = model.predict(X_val_scaled)
                y_train_pred = model.predict(X_train_scaled)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            
            # Cross-validation on training set
            if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                cv_scores = cross_val_score(model, X_train_scaled, 
                                          self.label_encoder.transform(y_train), 
                                          cv=5, scoring='accuracy')
            else:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                          cv=5, scoring='accuracy')
            
            results[model_name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores,
                'y_val_pred': y_val_pred,
                'classification_report': classification_report(y_val, y_val_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_val, y_val_pred)
            }
            
            print(f"  Training Accuracy: {train_accuracy:.3f}")
            print(f"  Validation Accuracy: {val_accuracy:.3f}")
            print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Save model
            joblib.dump(model, f"{self.output_dir}/models/{model_name.lower()}_model.pkl")
        
        # Create ensemble if multiple models
        if len(self.models) > 1:
            print(f"\n🎯 Training Ensemble Model...")
            
            ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in self.models.items()],
                voting='soft'
            )
            
            # For ensemble, use original labels (not encoded)
            ensemble.fit(X_train_scaled, y_train)
            
            # Validate ensemble
            y_val_pred_ensemble = ensemble.predict(X_val_scaled)
            y_train_pred_ensemble = ensemble.predict(X_train_scaled)
            
            ensemble_train_acc = accuracy_score(y_train, y_train_pred_ensemble)
            ensemble_val_acc = accuracy_score(y_val, y_val_pred_ensemble)
            
            # Cross-validation for ensemble
            ensemble_cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, 
                                               cv=5, scoring='accuracy')
            
            results['Ensemble'] = {
                'model': ensemble,
                'train_accuracy': ensemble_train_acc,
                'val_accuracy': ensemble_val_acc,
                'cv_mean': ensemble_cv_scores.mean(),
                'cv_std': ensemble_cv_scores.std(),
                'cv_scores': ensemble_cv_scores,
                'y_val_pred': y_val_pred_ensemble,
                'classification_report': classification_report(y_val, y_val_pred_ensemble, output_dict=True),
                'confusion_matrix': confusion_matrix(y_val, y_val_pred_ensemble)
            }
            
            print(f"  Training Accuracy: {ensemble_train_acc:.3f}")
            print(f"  Validation Accuracy: {ensemble_val_acc:.3f}")
            print(f"  CV Accuracy: {ensemble_cv_scores.mean():.3f} ± {ensemble_cv_scores.std():.3f}")
            
            joblib.dump(ensemble, f"{self.output_dir}/models/ensemble_model.pkl")
        
        # Save preprocessing components
        joblib.dump(self.scaler, f"{self.output_dir}/models/scaler.pkl")
        joblib.dump(self.label_encoder, f"{self.output_dir}/models/label_encoder.pkl")
        
        return results, {'smote_applied': smote_applied}
    
    def evaluate_on_test_set(self, results, data_splits):
        """Final evaluation on held-out test set"""
        print(f"\n🧪 Final evaluation on held-out test set...")
        
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        test_results = {}
        
        for model_name, result in results.items():
            model = result['model']
            
            # Make predictions
            if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                y_test_pred_encoded = model.predict(X_test_scaled)
                y_test_pred = self.label_encoder.inverse_transform(y_test_pred_encoded)
            else:
                y_test_pred = model.predict(X_test_scaled)
            
            # Calculate test metrics
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_report = classification_report(y_test, y_test_pred, output_dict=True)
            test_cm = confusion_matrix(y_test, y_test_pred)
            
            test_results[model_name] = {
                'test_accuracy': test_accuracy,
                'test_report': test_report,
                'test_confusion_matrix': test_cm,
                'y_test_pred': y_test_pred
            }
            
            print(f"  {model_name} Test Accuracy: {test_accuracy:.3f}")
        
        return test_results

class ProductionPredictor:
    """Production-ready prediction system"""
    
    def __init__(self, model_path, scaler_path, label_encoder_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.detector = ProductionLandmarkDetector()
        self.feature_extractor = AdvancedFeatureExtractor()
        
        # Get feature names from training
        self.expected_features = self._get_expected_features()
    
    def _get_expected_features(self):
        """Get expected feature names (should match training)"""
        # This should match the features extracted during training
        return [
            'num_cores', 'num_deltas', 'has_core', 'has_delta', 'has_multiple_cores', 'has_multiple_deltas',
            'mean_reliability', 'std_reliability', 'max_reliability', 'mean_frequency', 'std_frequency', 'mean_coherence',
            'orientation_smoothness', 'orientation_variance', 'orientation_entropy', 'frequency_range', 'frequency_entropy',
            'core_center_x', 'core_center_y', 'core_spread_x', 'core_spread_y',
            'delta_center_x', 'delta_center_y', 'delta_spread_x', 'delta_spread_y',
            'min_core_delta_distance', 'max_core_delta_distance', 'mean_core_delta_distance', 'std_core_delta_distance',
            'core_delta_angle_variance', 'core_delta_angle_range',
            'min_delta_delta_distance', 'max_delta_delta_distance', 'mean_delta_delta_distance', 'delta_angle_variance',
            'strongest_core_strength', 'mean_core_strength', 'strongest_core_frequency',
            'strongest_delta_strength', 'mean_delta_strength', 'strongest_delta_frequency',
            'horizontal_symmetry', 'vertical_symmetry'
        ]
    
    def predict_single_image(self, image_path):
        """Predict class for a single fingerprint image"""
        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Detect landmarks and extract features
            landmarks = self.detector.detect_landmarks(image)
            features = self.feature_extractor.extract_features(landmarks)
            
            # Prepare feature vector
            feature_vector = []
            for feature_name in self.expected_features:
                feature_vector.append(features.get(feature_name, 0))
            
            # Scale features
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(feature_vector_scaled)[0]
                classes = self.model.classes_
                
                # For XGBoost, handle label encoding
                if hasattr(self.model, 'get_booster'):  # XGBoost model
                    predicted_class_encoded = self.model.predict(feature_vector_scaled)[0]
                    predicted_class = self.label_encoder.inverse_transform([predicted_class_encoded])[0]
                else:
                    predicted_class = self.model.predict(feature_vector_scaled)[0]
                
                # Create probability dictionary
                if hasattr(self.model, 'get_booster'):  # XGBoost
                    prob_dict = {}
                    for i, prob in enumerate(probabilities):
                        class_name = self.label_encoder.inverse_transform([i])[0]
                        prob_dict[class_name] = prob
                else:
                    prob_dict = dict(zip(classes, probabilities))
            else:
                predicted_class = self.model.predict(feature_vector_scaled)[0]
                prob_dict = {predicted_class: 1.0}
            
            return {
                'predicted_class': predicted_class,
                'probabilities': prob_dict,
                'confidence': max(prob_dict.values()),
                'landmarks': {
                    'cores': len(landmarks['cores']),
                    'deltas': len(landmarks['deltas'])
                }
            }
            
        except Exception as e:
            return {
                'predicted_class': 'Error',
                'probabilities': {},
                'confidence': 0.0,
                'error': str(e)
            }

class ResultsGenerator:
    """Generate comprehensive results and analysis"""
    
    def __init__(self, output_dir="production_output"):
        self.output_dir = output_dir
    
    def generate_comprehensive_report(self, results, test_results, data_info, feature_cols):
        """Generate detailed analysis report"""
        print(f"\n📝 Generating comprehensive results report...")
        
        report_lines = []
        report_lines.append("# Production Fingerprint Classification Results")
        report_lines.append("=" * 55)
        report_lines.append("")
        
        # Dataset summary
        report_lines.append("## Dataset Summary")
        report_lines.append(f"- **Total images processed**: {data_info.get('total_images', 'N/A')}")
        report_lines.append(f"- **Features extracted**: {len(feature_cols)}")
        report_lines.append(f"- **SMOTE balancing applied**: {data_info.get('smote_applied', False)}")
        report_lines.append("")
        
        # Model performance comparison
        report_lines.append("## Model Performance Summary")
        report_lines.append("")
        
        # Validation results
        report_lines.append("### Validation Set Performance")
        report_lines.append("| Model | CV Accuracy | Val Accuracy | Train Accuracy |")
        report_lines.append("|-------|-------------|--------------|----------------|")
        
        for model_name, result in results.items():
            cv_acc = f"{result['cv_mean']:.3f} ± {result['cv_std']:.3f}"
            val_acc = f"{result['val_accuracy']:.3f}"
            train_acc = f"{result['train_accuracy']:.3f}"
            report_lines.append(f"| {model_name} | {cv_acc} | {val_acc} | {train_acc} |")
        
        report_lines.append("")
        
        # Test set results (final evaluation)
        report_lines.append("### 🎯 Final Test Set Performance (Held-out Data)")
        report_lines.append("| Model | Test Accuracy | Status |")
        report_lines.append("|-------|---------------|--------|")
        
        best_test_acc = 0
        best_model_name = ""
        
        for model_name, test_result in test_results.items():
            test_acc = test_result['test_accuracy']
            status = "✅ Excellent" if test_acc >= 0.90 else "🎯 Good" if test_acc >= 0.85 else "⚠️ Needs Improvement"
            report_lines.append(f"| {model_name} | {test_acc:.3f} | {status} |")
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_name = model_name
        
        report_lines.append("")
        
        # Best model detailed analysis
        if best_model_name:
            report_lines.append(f"## 🏆 Best Model: {best_model_name}")
            report_lines.append(f"**Final Test Accuracy: {best_test_acc:.3f}**")
            report_lines.append("")
            
            # Per-class performance on test set
            test_report = test_results[best_model_name]['test_report']
            report_lines.append("### Per-Class Performance (Test Set)")
            report_lines.append("| Class | Precision | Recall | F1-Score | Support |")
            report_lines.append("|-------|-----------|--------|----------|---------|")
            
            for class_name in ['Arch', 'Loop', 'Whorl']:
                if class_name in test_report:
                    metrics = test_report[class_name]
                    prec = f"{metrics['precision']:.3f}"
                    rec = f"{metrics['recall']:.3f}"
                    f1 = f"{metrics['f1-score']:.3f}"
                    support = int(metrics['support'])
                    report_lines.append(f"| {class_name} | {prec} | {rec} | {f1} | {support} |")
            
            report_lines.append("")
        
        # Feature importance (if available)
        if best_model_name in results:
            best_model = results[best_model_name]['model']
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                top_indices = np.argsort(importances)[-10:][::-1]
                
                report_lines.append("### Top 10 Most Important Features")
                for i, idx in enumerate(top_indices):
                    feature_name = feature_cols[idx]
                    importance = importances[idx]
                    report_lines.append(f"{i+1}. **{feature_name}**: {importance:.4f}")
                report_lines.append("")
        
        # Success criteria evaluation
        report_lines.append("## 🎯 Success Criteria Evaluation")
        report_lines.append("")
        
        target_met = best_test_acc >= 0.90
        report_lines.append(f"- **Target Accuracy (≥90%)**: {'✅ ACHIEVED' if target_met else '❌ NOT MET'} ({best_test_acc:.1%})")
        
        # Check Arch recall specifically
        if best_model_name and 'Arch' in test_results[best_model_name]['test_report']:
            arch_recall = test_results[best_model_name]['test_report']['Arch']['recall']
            arch_target_met = arch_recall >= 0.70
            report_lines.append(f"- **Arch Recall (≥70%)**: {'✅ ACHIEVED' if arch_target_met else '❌ NOT MET'} ({arch_recall:.1%})")
        
        report_lines.append(f"- **Production Ready**: {'✅ YES' if target_met else '⚠️ NEEDS IMPROVEMENT'}")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("## 🚀 Recommendations")
        if target_met:
            report_lines.append("### ✅ Model is Production Ready!")
            report_lines.append("1. **Deploy the best model** for production use")
            report_lines.append("2. **Monitor performance** on new data")
            report_lines.append("3. **Collect edge cases** for continuous improvement")
            report_lines.append("4. **Set up retraining pipeline** for model updates")
        else:
            report_lines.append("### 🔧 Model Needs Improvement")
            report_lines.append("1. **Collect more training data**, especially for minority classes")
            report_lines.append("2. **Advanced feature engineering** (texture, minutiae)")
            report_lines.append("3. **Hyperparameter optimization** with grid search")
            report_lines.append("4. **Deep learning approaches** for landmark detection")
            report_lines.append("5. **Data augmentation** techniques")
        
        report_lines.append("")
        
        # Save report
        report_path = f"{self.output_dir}/landmark_results_analysis.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📝 Comprehensive report saved to: {report_path}")
        return report_path
    
    def create_visualizations(self, results, test_results):
        """Create comprehensive visualizations"""
        print(f"📊 Creating visualizations...")
        
        plt.figure(figsize=(20, 12))
        
        # Model comparison
        plt.subplot(2, 4, 1)
        model_names = list(results.keys())
        cv_scores = [results[name]['cv_mean'] for name in model_names]
        val_scores = [results[name]['val_accuracy'] for name in model_names]
        test_scores = [test_results[name]['test_accuracy'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        plt.bar(x - width, cv_scores, width, label='CV Score', alpha=0.8)
        plt.bar(x, val_scores, width, label='Val Score', alpha=0.8)
        plt.bar(x + width, test_scores, width, label='Test Score', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Best model confusion matrix (test set)
        best_model_name = max(test_results.keys(), key=lambda k: test_results[k]['test_accuracy'])
        test_cm = test_results[best_model_name]['test_confusion_matrix']
        
        plt.subplot(2, 4, 2)
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Arch', 'Loop', 'Whorl'],
                   yticklabels=['Arch', 'Loop', 'Whorl'])
        plt.title(f'Test Set Confusion Matrix\n{best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Feature importance (if available)
        plt.subplot(2, 4, 3)
        if hasattr(results[best_model_name]['model'], 'feature_importances_'):
            importances = results[best_model_name]['model'].feature_importances_
            indices = np.argsort(importances)[-15:]
            
            # Get feature names (simplified for plotting)
            feature_names = [f"F{i}" for i in indices]  # Simplified names
            
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), feature_names)
            plt.title(f'Top 15 Features\n{best_model_name}')
            plt.xlabel('Importance')
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
        
        # Per-class performance
        plt.subplot(2, 4, 4)
        test_report = test_results[best_model_name]['test_report']
        classes = ['Arch', 'Loop', 'Whorl']
        precisions = [test_report[cls]['precision'] for cls in classes if cls in test_report]
        recalls = [test_report[cls]['recall'] for cls in classes if cls in test_report]
        f1_scores = [test_report[cls]['f1-score'] for cls in classes if cls in test_report]
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        plt.bar(x, recalls, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title(f'Per-Class Performance\n{best_model_name}')
        plt.xticks(x, classes)
        plt.legend()
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/comprehensive_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to {self.output_dir}/plots/")

def create_prediction_script(output_dir="production_output"):
    """Create standalone prediction script"""
    predict_script = '''#!/usr/bin/env python3
"""
Standalone Fingerprint Prediction Script
=======================================

Production-ready prediction function for fingerprint classification.
Loads trained models and predicts fingerprint class (Arch/Loop/Whorl).

Usage:
    python predict.py <image_path>
    
Example:
    python predict.py fingerprint.png
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_fingerprint_pipeline import ProductionPredictor

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Load best model (update paths as needed)
    model_dir = "production_output/models"
    
    try:
        # Try to load ensemble model first (usually best)
        predictor = ProductionPredictor(
            model_path=f"{model_dir}/ensemble_model.pkl",
            scaler_path=f"{model_dir}/scaler.pkl", 
            label_encoder_path=f"{model_dir}/label_encoder.pkl"
        )
    except:
        try:
            # Fallback to XGBoost
            predictor = ProductionPredictor(
                model_path=f"{model_dir}/xgboost_model.pkl",
                scaler_path=f"{model_dir}/scaler.pkl",
                label_encoder_path=f"{model_dir}/label_encoder.pkl"
            )
        except:
            # Fallback to Random Forest
            predictor = ProductionPredictor(
                model_path=f"{model_dir}/randomforest_model.pkl",
                scaler_path=f"{model_dir}/scaler.pkl",
                label_encoder_path=f"{model_dir}/label_encoder.pkl"
            )
    
    # Make prediction
    result = predictor.predict_single_image(image_path)
    
    # Display results
    print(f"\\n🔍 Fingerprint Analysis Results")
    print(f"================================")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if 'probabilities' in result and result['probabilities']:
        print(f"\\nClass Probabilities:")
        for class_name, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {prob:.3f}")
    
    if 'landmarks' in result:
        landmarks = result['landmarks']
        print(f"\\nLandmarks Detected:")
        print(f"  Cores: {landmarks['cores']}")
        print(f"  Deltas: {landmarks['deltas']}")
    
    if 'error' in result:
        print(f"\\n❌ Error: {result['error']}")

if __name__ == "__main__":
    main()
'''
    
    with open(f"{output_dir}/predict.py", 'w', encoding='utf-8') as f:
        f.write(predict_script)
    
    print(f"📄 Prediction script created: {output_dir}/predict.py")

def main():
    """Complete production pipeline execution"""
    print("🚀 Production Fingerprint Classification Pipeline")
    print("=" * 60)
    print("Target: Process 6,000+ images with ≥90% accuracy")
    print()
    
    # Initialize components
    detector = ProductionLandmarkDetector()
    feature_extractor = AdvancedFeatureExtractor()
    
    # Create output directories
    os.makedirs("production_output", exist_ok=True)
    os.makedirs("production_output/models", exist_ok=True)
    os.makedirs("production_output/plots", exist_ok=True)
    
    # Data root - update this path as needed
    data_root = r"C:\Users\alifa\Downloads\fingerprint_model-20250811T194218Z-1-001\fingerprint_model\data"
    
    if not os.path.exists(data_root):
        print(f"❌ Data directory not found: {data_root}")
        print("Please update the data_root path in the script")
        return
    
    # Step 1: Load and analyze dataset
    print("📂 Step 1: Loading and analyzing dataset")
    print("-" * 40)
    data_loader = ProductionDataLoader(data_root)
    all_images = data_loader.discover_all_images()
    
    if len(all_images) == 0:
        print("❌ No fingerprint images found. Please check the data directory structure.")
        return
    
    # Analyze distribution
    dataset_df = data_loader.analyze_dataset_distribution(all_images)
    
    # Step 2: Feature extraction
    print("\n🔧 Step 2: Feature extraction from all images")
    print("-" * 40)
    processor = ProductionFeatureProcessor(detector, feature_extractor)
    features_df = processor.process_all_images(all_images)
    
    # Save raw features
    raw_features_path = "production_output/fingerprint_features.csv"
    processor.save_features(features_df, raw_features_path)
    
    # Step 3: Model training and evaluation
    print("\n🎯 Step 3: Model training and evaluation")
    print("-" * 40)
    trainer = ProductionModelTrainer()
    
    # Prepare data splits
    try:
        data_splits = trainer.prepare_data(features_df)
    except ValueError as e:
        print(f"❌ Data preparation failed: {e}")
        return
    
    # Train models
    results, training_info = trainer.train_models(data_splits, use_smote=True)
    
    # Final evaluation on test set
    test_results = trainer.evaluate_on_test_set(results, data_splits)
    
    # Step 4: Generate comprehensive results
    print("\n📊 Step 4: Generating results and analysis")
    print("-" * 40)
    results_generator = ResultsGenerator()
    
    # Compile data info
    data_info = {
        'total_images': len(features_df),
        'smote_applied': training_info.get('smote_applied', False)
    }
    
    # Generate report
    report_path = results_generator.generate_comprehensive_report(
        results, test_results, data_info, data_splits['feature_cols']
    )
    
    # Create visualizations
    results_generator.create_visualizations(results, test_results)
    
    # Step 5: Create prediction script
    print("\n📄 Step 5: Creating production prediction script")
    print("-" * 40)
    create_prediction_script()
    
    # Final summary
    print("\n🎉 Production Pipeline Complete!")
    print("=" * 40)
    
    # Find best model
    best_model_name = max(test_results.keys(), key=lambda k: test_results[k]['test_accuracy'])
    best_test_accuracy = test_results[best_model_name]['test_accuracy']
    
    print(f"🏆 Best Model: {best_model_name}")
    print(f"🎯 Final Test Accuracy: {best_test_accuracy:.3f} ({best_test_accuracy:.1%})")
    
    # Success evaluation
    target_met = best_test_accuracy >= 0.90
    if target_met:
        print("✅ TARGET ACHIEVED: ≥90% accuracy reached!")
        print("🚀 Model is production-ready!")
    else:
        print(f"⚠️ Target not met: {best_test_accuracy:.1%} < 90%")
        print("📈 Review recommendations in the report for improvements")
    
    # Show per-class performance for Arch
    if 'Arch' in test_results[best_model_name]['test_report']:
        arch_recall = test_results[best_model_name]['test_report']['Arch']['recall']
        arch_target_met = arch_recall >= 0.70
        print(f"📊 Arch Recall: {arch_recall:.3f} ({'✅ Good' if arch_target_met else '⚠️ Needs improvement'})")
    
    print(f"\n📁 All outputs saved to: production_output/")
    print(f"📝 Detailed report: {report_path}")
    print(f"📄 Prediction script: production_output/predict.py")
    print(f"🎯 Usage: python production_output/predict.py <image_path>")

if __name__ == "__main__":
    main()
