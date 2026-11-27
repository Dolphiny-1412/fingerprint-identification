#!/usr/bin/env python3
"""
Fingerprint Spoof Detection Module
=================================

Implements feature-based spoof detection using:
1. Ridge clarity analysis
2. Perspiration pore detection
3. Texture analysis
4. ML-based classification

Detects presentation attacks (PAD) vs live fingerprints.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional
import joblib
import json

class SpoofDetector:
    """
    Fingerprint spoof detection using multiple feature-based approaches
    """
    
    def __init__(self):
        """Initialize spoof detector"""
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.is_trained = False
        
        # Feature extraction parameters
        self.texture_params = {
            'lbp_radius': 3,
            'lbp_points': 24,
            'glcm_distances': [1, 2, 3],
            'glcm_angles': [0, 45, 90, 135]
        }
    
    def extract_ridge_clarity_features(self, image: np.ndarray) -> Dict:
        """
        Extract ridge clarity features for spoof detection
        
        Args:
            image: Input grayscale fingerprint image
            
        Returns:
            Dictionary of ridge clarity features
        """
        # Enhance image
        enhanced = cv2.equalizeHist(image)
        
        # Compute orientation field
        orientation_field = self._compute_orientation_field(enhanced)
        
        # Compute ridge frequency
        frequency_map = self._compute_ridge_frequency(enhanced, orientation_field)
        
        # Ridge clarity metrics
        features = {}
        
        # 1. Orientation consistency
        valid_orientations = orientation_field[~np.isnan(orientation_field)]
        if len(valid_orientations) > 0:
            features['orientation_variance'] = float(np.var(valid_orientations))
            features['orientation_coherence'] = float(self._compute_coherence(orientation_field))
        else:
            features['orientation_variance'] = 0.0
            features['orientation_coherence'] = 0.0
        
        # 2. Ridge frequency consistency
        valid_frequencies = frequency_map[frequency_map > 0]
        if len(valid_frequencies) > 0:
            features['frequency_mean'] = float(np.mean(valid_frequencies))
            features['frequency_variance'] = float(np.var(valid_frequencies))
            features['frequency_range'] = float(np.ptp(valid_frequencies))
        else:
            features['frequency_mean'] = 0.0
            features['frequency_variance'] = 0.0
            features['frequency_range'] = 0.0
        
        # 3. Ridge continuity
        features['ridge_continuity'] = float(self._compute_ridge_continuity(enhanced))
        
        # 4. Ridge-valley contrast
        features['ridge_valley_contrast'] = float(self._compute_ridge_valley_contrast(enhanced))
        
        return features
    
    def extract_pore_features(self, image: np.ndarray) -> Dict:
        """
        Extract perspiration pore features
        
        Args:
            image: Input grayscale fingerprint image
            
        Returns:
            Dictionary of pore-related features
        """
        features = {}
        
        # Enhance image for pore detection
        enhanced = cv2.equalizeHist(image)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
        
        # Detect potential pores using blob detection
        # Set up blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 3
        params.maxArea = 30
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = True
        params.minConvexity = 0.7
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(blurred)
        
        # Pore features
        features['pore_count'] = len(keypoints)
        features['pore_density'] = len(keypoints) / (image.shape[0] * image.shape[1]) * 10000
        
        if keypoints:
            # Pore size statistics
            pore_sizes = [kp.size for kp in keypoints]
            features['pore_size_mean'] = float(np.mean(pore_sizes))
            features['pore_size_variance'] = float(np.var(pore_sizes))
            
            # Pore distribution
            pore_positions = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
            if len(pore_positions) > 1:
                # Compute nearest neighbor distances
                from scipy.spatial.distance import pdist
                distances = pdist(pore_positions)
                features['pore_spacing_mean'] = float(np.mean(distances))
                features['pore_spacing_variance'] = float(np.var(distances))
            else:
                features['pore_spacing_mean'] = 0.0
                features['pore_spacing_variance'] = 0.0
        else:
            features['pore_size_mean'] = 0.0
            features['pore_size_variance'] = 0.0
            features['pore_spacing_mean'] = 0.0
            features['pore_spacing_variance'] = 0.0
        
        return features
    
    def extract_texture_features(self, image: np.ndarray) -> Dict:
        """
        Extract texture-based features for spoof detection
        
        Args:
            image: Input grayscale fingerprint image
            
        Returns:
            Dictionary of texture features
        """
        features = {}
        
        # 1. Local Binary Pattern (LBP)
        lbp_features = self._compute_lbp_features(image)
        features.update(lbp_features)
        
        # 2. Gray Level Co-occurrence Matrix (GLCM)
        glcm_features = self._compute_glcm_features(image)
        features.update(glcm_features)
        
        # 3. Gabor filter responses
        gabor_features = self._compute_gabor_features(image)
        features.update(gabor_features)
        
        # 4. Statistical texture measures
        stat_features = self._compute_statistical_features(image)
        features.update(stat_features)
        
        return features
    
    def _compute_orientation_field(self, image: np.ndarray, block_size: int = 16) -> np.ndarray:
        """Compute orientation field of ridges"""
        h, w = image.shape
        orientation_field = np.full((h // block_size, w // block_size), np.nan)
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = image[i:i+block_size, j:j+block_size].astype(np.float32)
                
                # Compute gradients
                gx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)
                
                # Compute orientation
                if np.sum(gx**2 + gy**2) > 0:
                    orientation = 0.5 * np.arctan2(2 * np.sum(gx * gy), 
                                                 np.sum(gx**2 - gy**2))
                    orientation_field[i//block_size, j//block_size] = orientation
        
        return orientation_field
    
    def _compute_ridge_frequency(self, image: np.ndarray, orientation_field: np.ndarray, 
                                block_size: int = 16) -> np.ndarray:
        """Compute ridge frequency map"""
        h, w = image.shape
        frequency_map = np.zeros((h // block_size, w // block_size))
        
        for i in range(orientation_field.shape[0]):
            for j in range(orientation_field.shape[1]):
                if not np.isnan(orientation_field[i, j]):
                    # Extract block
                    y1, y2 = i * block_size, (i + 1) * block_size
                    x1, x2 = j * block_size, (j + 1) * block_size
                    
                    if y2 <= h and x2 <= w:
                        block = image[y1:y2, x1:x2]
                        
                        # Estimate frequency using projection
                        freq = self._estimate_frequency_in_block(block, orientation_field[i, j])
                        frequency_map[i, j] = freq
        
        return frequency_map
    
    def _estimate_frequency_in_block(self, block: np.ndarray, orientation: float) -> float:
        """Estimate ridge frequency in a block"""
        # Project block along perpendicular to ridge direction
        cos_o = np.cos(orientation + np.pi/2)
        sin_o = np.sin(orientation + np.pi/2)
        
        # Simple projection sum
        projection = []
        center_x, center_y = block.shape[1] // 2, block.shape[0] // 2
        
        for t in range(-min(center_x, center_y), min(center_x, center_y)):
            x = int(center_x + t * cos_o)
            y = int(center_y + t * sin_o)
            
            if 0 <= x < block.shape[1] and 0 <= y < block.shape[0]:
                projection.append(block[y, x])
        
        if len(projection) < 10:
            return 0.0
        
        # Find peaks to estimate frequency
        projection = np.array(projection)
        diff = np.diff(projection)
        peaks = []
        
        for i in range(1, len(diff) - 1):
            if diff[i-1] > 0 and diff[i] < 0:  # Peak
                peaks.append(i)
        
        if len(peaks) < 2:
            return 0.0
        
        # Average distance between peaks
        peak_distances = np.diff(peaks)
        return 1.0 / np.mean(peak_distances) if len(peak_distances) > 0 else 0.0
    
    def _compute_coherence(self, orientation_field: np.ndarray) -> float:
        """Compute orientation coherence"""
        valid_orientations = orientation_field[~np.isnan(orientation_field)]
        if len(valid_orientations) < 2:
            return 0.0
        
        # Compute local coherence
        cos_sum = np.sum(np.cos(2 * valid_orientations))
        sin_sum = np.sum(np.sin(2 * valid_orientations))
        
        coherence = np.sqrt(cos_sum**2 + sin_sum**2) / len(valid_orientations)
        return coherence
    
    def _compute_ridge_continuity(self, image: np.ndarray) -> float:
        """Compute ridge continuity measure"""
        # Apply morphological operations to detect ridges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Threshold image
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply opening to break discontinuous ridges
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Compute continuity as ratio of pixels surviving opening
        continuity = np.sum(opened > 0) / max(np.sum(binary > 0), 1)
        return continuity
    
    def _compute_ridge_valley_contrast(self, image: np.ndarray) -> float:
        """Compute contrast between ridges and valleys"""
        # Apply Otsu threshold to separate ridges and valleys
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        ridge_pixels = image[binary > 0]
        valley_pixels = image[binary == 0]
        
        if len(ridge_pixels) > 0 and len(valley_pixels) > 0:
            contrast = abs(np.mean(ridge_pixels) - np.mean(valley_pixels))
            return contrast / 255.0
        
        return 0.0
    
    def _compute_lbp_features(self, image: np.ndarray) -> Dict:
        """Compute Local Binary Pattern features"""
        # Simplified LBP implementation
        h, w = image.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                code = 0
                
                # 8-neighbor LBP
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i-1, j-1] = code
        
        # Compute LBP histogram
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        # Extract statistical features from histogram
        features = {
            'lbp_mean': float(np.mean(hist)),
            'lbp_variance': float(np.var(hist)),
            'lbp_skewness': float(self._compute_skewness(hist)),
            'lbp_kurtosis': float(self._compute_kurtosis(hist)),
            'lbp_entropy': float(-np.sum(hist * np.log(hist + 1e-7)))
        }
        
        return features
    
    def _compute_glcm_features(self, image: np.ndarray) -> Dict:
        """Compute Gray Level Co-occurrence Matrix features"""
        # Simplified GLCM computation for one distance and angle
        distance = 1
        angle = 0  # 0 degrees
        
        # Quantize image to reduce computation
        quantized = (image // 32).astype(np.uint8)  # 8 levels
        max_val = quantized.max()
        
        if max_val == 0:
            return {'glcm_contrast': 0.0, 'glcm_homogeneity': 0.0, 
                   'glcm_energy': 0.0, 'glcm_correlation': 0.0}
        
        # Compute GLCM
        glcm = np.zeros((max_val + 1, max_val + 1), dtype=np.float32)
        
        h, w = quantized.shape
        dx, dy = 1, 0  # Horizontal pairs
        
        for i in range(h):
            for j in range(w - dx):
                val1 = quantized[i, j]
                val2 = quantized[i, j + dx]
                glcm[val1, val2] += 1
                glcm[val2, val1] += 1  # Symmetric
        
        # Normalize
        glcm /= (glcm.sum() + 1e-7)
        
        # Compute features
        features = {}
        
        # Contrast
        contrast = 0.0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                contrast += glcm[i, j] * (i - j) ** 2
        features['glcm_contrast'] = float(contrast)
        
        # Homogeneity
        homogeneity = 0.0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                homogeneity += glcm[i, j] / (1 + abs(i - j))
        features['glcm_homogeneity'] = float(homogeneity)
        
        # Energy
        energy = np.sum(glcm ** 2)
        features['glcm_energy'] = float(energy)
        
        # Correlation (simplified)
        mean_i = np.sum(np.arange(glcm.shape[0]) * np.sum(glcm, axis=1))
        mean_j = np.sum(np.arange(glcm.shape[1]) * np.sum(glcm, axis=0))
        
        correlation = 0.0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                correlation += (i - mean_i) * (j - mean_j) * glcm[i, j]
        
        features['glcm_correlation'] = float(correlation)
        
        return features
    
    def _compute_gabor_features(self, image: np.ndarray) -> Dict:
        """Compute Gabor filter responses"""
        features = {}
        
        # Apply Gabor filters with different orientations
        orientations = [0, 45, 90, 135]
        
        responses = []
        for angle in orientations:
            # Create Gabor kernel
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 
                                      2*np.pi*0.1, 0.5, 0, ktype=cv2.CV_32F)
            
            # Apply filter
            response = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            responses.append(response)
        
        # Compute statistics of responses
        all_responses = np.concatenate([r.flatten() for r in responses])
        
        features['gabor_mean'] = float(np.mean(all_responses))
        features['gabor_variance'] = float(np.var(all_responses))
        features['gabor_skewness'] = float(self._compute_skewness(all_responses))
        features['gabor_kurtosis'] = float(self._compute_kurtosis(all_responses))
        
        return features
    
    def _compute_statistical_features(self, image: np.ndarray) -> Dict:
        """Compute basic statistical features"""
        flat = image.flatten().astype(np.float32)
        
        features = {
            'stat_mean': float(np.mean(flat)),
            'stat_variance': float(np.var(flat)),
            'stat_skewness': float(self._compute_skewness(flat)),
            'stat_kurtosis': float(self._compute_kurtosis(flat)),
            'stat_entropy': float(self._compute_entropy(flat))
        }
        
        return features
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return np.mean(((data - mean_val) / std_val) ** 4) - 3
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute entropy"""
        hist, _ = np.histogram(data, bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        entropy = -np.sum(hist * np.log(hist + 1e-7))
        return entropy
    
    def extract_all_features(self, image: np.ndarray) -> Dict:
        """
        Extract all spoof detection features
        
        Args:
            image: Input grayscale fingerprint image
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # Ridge clarity features
        ridge_features = self.extract_ridge_clarity_features(image)
        features.update({f'ridge_{k}': v for k, v in ridge_features.items()})
        
        # Pore features
        pore_features = self.extract_pore_features(image)
        features.update({f'pore_{k}': v for k, v in pore_features.items()})
        
        # Texture features
        texture_features = self.extract_texture_features(image)
        features.update({f'texture_{k}': v for k, v in texture_features.items()})
        
        return features
    
    def predict_spoof(self, image: np.ndarray) -> Dict:
        """
        Predict if fingerprint is spoofed or live
        
        Args:
            image: Input grayscale fingerprint image
            
        Returns:
            Dictionary containing prediction results
        """
        # Extract features
        features = self.extract_all_features(image)
        
        # For demo purposes, use rule-based approach since we don't have training data
        spoof_score = self._compute_rule_based_spoof_score(features)
        
        # Convert to live probability
        live_probability = 1.0 - spoof_score
        
        # Make decision
        is_live = live_probability > 0.5
        confidence = max(live_probability, spoof_score)
        
        return {
            'is_live': is_live,
            'live_probability': live_probability,
            'spoof_probability': spoof_score,
            'confidence': confidence,
            'prediction': 'Live' if is_live else 'Spoof',
            'features': features
        }
    
    def _compute_rule_based_spoof_score(self, features: Dict) -> float:
        """
        Compute spoof score using rule-based approach
        
        Args:
            features: Extracted features
            
        Returns:
            Spoof probability (0-1)
        """
        spoof_indicators = []
        
        # Ridge clarity indicators
        ridge_clarity = features.get('ridge_ridge_continuity', 0.5)
        if ridge_clarity < 0.3:  # Low ridge continuity suggests spoof
            spoof_indicators.append(0.3)
        
        ridge_contrast = features.get('ridge_ridge_valley_contrast', 0.5)
        if ridge_contrast < 0.2:  # Low contrast suggests spoof
            spoof_indicators.append(0.2)
        
        # Pore indicators
        pore_density = features.get('pore_pore_density', 0.0)
        if pore_density < 0.1:  # Very few pores suggest spoof
            spoof_indicators.append(0.4)
        elif pore_density > 2.0:  # Too many pores might be artifacts
            spoof_indicators.append(0.2)
        
        # Texture indicators
        texture_variance = features.get('texture_stat_variance', 1000.0)
        if texture_variance < 100:  # Very low texture variance suggests spoof
            spoof_indicators.append(0.3)
        
        lbp_entropy = features.get('texture_lbp_entropy', 3.0)
        if lbp_entropy < 2.0:  # Low texture complexity
            spoof_indicators.append(0.2)
        
        # Compute overall spoof score
        if spoof_indicators:
            spoof_score = np.mean(spoof_indicators)
        else:
            spoof_score = 0.1  # Default low spoof probability for live images
        
        return min(spoof_score, 0.9)  # Cap at 90%

def test_spoof_detection():
    """Test spoof detection system"""
    print("🛡️ Testing Fingerprint Spoof Detection System")
    print("=" * 60)
    
    # Load test image
    sample_path = r"C:\Users\alifa\Downloads\fingerprint_model-20250811T194218Z-1-001\fingerprint_model\data\archive\fingerprints\DB1_B\101_1.tif"
    
    image = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("❌ Failed to load test image")
        return
    
    print(f"📷 Testing on: {sample_path}")
    
    # Initialize detector
    detector = SpoofDetector()
    
    # Predict spoof/live
    print("🔍 Analyzing image for spoof detection...")
    result = detector.predict_spoof(image)
    
    # Display results
    print(f"\n📊 Spoof Detection Results:")
    print(f"   Prediction: {result['prediction']}")
    print(f"   Live Probability: {result['live_probability']:.3f}")
    print(f"   Spoof Probability: {result['spoof_probability']:.3f}")
    print(f"   Confidence: {result['confidence']:.3f}")
    
    # Show key features
    print(f"\n🔍 Key Features:")
    features = result['features']
    key_features = [
        'ridge_ridge_continuity', 'ridge_ridge_valley_contrast',
        'pore_pore_density', 'texture_lbp_entropy'
    ]
    
    for feature in key_features:
        if feature in features:
            print(f"   {feature}: {features[feature]:.3f}")
    
    # Save results
    result_for_json = {
        'filename': 'test_image.tif',
        'prediction': result['prediction'],
        'live_probability': float(result['live_probability']),
        'spoof_probability': float(result['spoof_probability']),
        'confidence': float(result['confidence']),
        'key_features': {k: float(features[k]) for k in key_features if k in features}
    }
    
    with open('spoof_detection_test.json', 'w') as f:
        json.dump(result_for_json, f, indent=2)
    
    print(f"\n✅ Spoof detection test complete!")
    print(f"💾 Results saved: spoof_detection_test.json")

if __name__ == "__main__":
    test_spoof_detection()
