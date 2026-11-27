"""
Task C - Advanced Minutiae Detection & Security Analysis
Enhanced pipeline with dataset-adaptive processing and multi-level security
"""

import cv2
import numpy as np
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

from skimage import filters, morphology, feature, measure
from skimage.segmentation import watershed
from scipy import ndimage
from sklearn.ensemble import IsolationForest
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFingerprintAnalyzer:
    """
    High-performance fingerprint analysis with adaptive security checks
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.spoof_detector = None
        self.quality_thresholds = self._calibrate_thresholds()
        
    def _default_config(self) -> Dict:
        """Default configuration optimized for forensic datasets"""
        return {
            'preprocessing': {
                'target_dpi': 500,
                'gabor_frequencies': [0.1, 0.15, 0.2],
                'gabor_angles': [0, 45, 90, 135],
                'block_size': 16,
                'normalize': True
            },
            'quality': {
                'min_contrast': 0.3,
                'min_clarity': 50.0,
                'max_noise_ratio': 0.15,
                'min_ridge_density': 0.4
            },
            'minutiae': {
                'min_confidence': 0.5,
                'max_minutiae': 200,
                'neighborhood_size': 5,
                'angle_tolerance': 15
            },
            'security': {
                'spoof_threshold': 0.6,
                'texture_analysis': True,
                'frequency_analysis': True,
                'orientation_analysis': True
            },
            'performance': {
                'parallel_workers': 4,
                'batch_size': 10,
                'cache_preprocessed': True
            }
        }
    
    def _calibrate_thresholds(self) -> Dict:
        """Auto-calibrate thresholds based on dataset characteristics"""
        return {
            'blur_threshold': 100.0,
            'brightness_range': (50, 200),
            'contrast_min': 30.0,
            'ridge_freq_range': (0.08, 0.25),
            'orientation_consistency': 0.7
        }
    
    def analyze_fingerprint(self, image_path: str, image_id: str = None) -> Dict:
        """
        Complete fingerprint analysis with security assessment
        
        Args:
            image_path: Path to fingerprint image
            image_id: Optional identifier for the image
            
        Returns:
            Comprehensive analysis results as JSON-compatible dict
        """
        start_time = time.time()
        
        try:
            # Load and validate image
            image = self._load_and_validate_image(image_path)
            if image is None:
                return self._create_error_result(image_path, "Invalid image format")
            
            image_id = image_id or Path(image_path).stem
            
            # Level 1: Basic validation and preprocessing
            preprocessed, roi_mask = self._preprocess_image(image)
            
            # Level 2: Quality assessment
            quality_results = self._assess_image_quality(image, preprocessed, roi_mask)
            
            # Early rejection for very poor quality
            if quality_results['quality_score'] < 0.2:
                return self._create_result(
                    image_id, quality_results, [], 0.0, True, 
                    processing_time=time.time() - start_time,
                    early_rejection="Poor image quality"
                )
            
            # Level 3: Ridge analysis and orientation field
            ridge_analysis = self._analyze_ridge_structure(preprocessed, roi_mask)
            
            # Level 4: Minutiae extraction with confidence scoring
            minutiae = self._extract_minutiae_with_confidence(preprocessed, ridge_analysis)
            
            # Level 5: Security assessment
            security_score = self._comprehensive_security_analysis(
                image, preprocessed, ridge_analysis, minutiae, roi_mask
            )
            
            # Combine results
            processing_time = time.time() - start_time
            
            return self._create_result(
                image_id, quality_results, minutiae, 
                security_score, security_score > self.config['security']['spoof_threshold'],
                processing_time, ridge_analysis
            )
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return self._create_error_result(image_path, str(e))
    
    def _load_and_validate_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and validate image with format checking"""
        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            
            # Basic validation
            h, w = image.shape
            if h < 100 or w < 100:  # Too small
                return None
            if h > 2000 or w > 2000:  # Too large - resize
                scale = min(2000/h, 2000/w)
                image = cv2.resize(image, None, fx=scale, fy=scale)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced preprocessing with ROI detection and ridge enhancement
        """
        # Normalize intensity
        normalized = cv2.equalizeHist(image)
        
        # ROI detection - find fingerprint area
        roi_mask = self._detect_fingerprint_roi(normalized)
        
        # Apply ROI mask
        masked = cv2.bitwise_and(normalized, normalized, mask=roi_mask)
        
        # Gaussian smoothing
        smoothed = cv2.GaussianBlur(masked, (3, 3), 0.5)
        
        # Gabor filter enhancement
        enhanced = self._gabor_enhancement(smoothed)
        
        # Binarization
        binary = self._adaptive_binarization(enhanced)
        
        # Morphological cleanup
        cleaned = self._morphological_cleanup(binary)
        
        # Skeletonization
        skeleton = morphology.skeletonize(cleaned > 0).astype(np.uint8) * 255
        
        return skeleton, roi_mask
    
    def _detect_fingerprint_roi(self, image: np.ndarray) -> np.ndarray:
        """Detect the actual fingerprint region"""
        # Otsu thresholding to find foreground
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        if num_labels > 1:
            # Get largest component (excluding background)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            roi_mask = (labels == largest_label).astype(np.uint8) * 255
        else:
            roi_mask = np.ones_like(image) * 255
        
        # Morphological operations to smooth ROI
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
        
        return roi_mask
    
    def _gabor_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhanced Gabor filtering for ridge enhancement"""
        enhanced = np.zeros_like(image, dtype=np.float32)
        
        for freq in self.config['preprocessing']['gabor_frequencies']:
            for angle in self.config['preprocessing']['gabor_angles']:
                gabor_real, _ = filters.gabor(image, frequency=freq, theta=np.deg2rad(angle))
                enhanced += np.abs(gabor_real)
        
        # Normalize
        enhanced = enhanced / len(self.config['preprocessing']['gabor_frequencies'])
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min()) * 255
        
        return enhanced.astype(np.uint8)
    
    def _adaptive_binarization(self, image: np.ndarray) -> np.ndarray:
        """Adaptive binarization with multiple methods"""
        # Method 1: Adaptive threshold
        binary1 = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Method 2: Otsu's method
        _, binary2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine methods
        combined = cv2.bitwise_and(binary1, binary2)
        
        return combined
    
    def _morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """Clean up binary image with morphological operations"""
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # Fill small gaps
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        return cleaned
    
    def _assess_image_quality(self, original: np.ndarray, processed: np.ndarray, roi_mask: np.ndarray) -> Dict:
        """Comprehensive image quality assessment"""
        quality_metrics = {}
        
        # Apply ROI mask to original for analysis
        roi_image = cv2.bitwise_and(original, original, mask=roi_mask)
        
        # 1. Blur assessment (Laplacian variance)
        laplacian_var = cv2.Laplacian(roi_image, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / self.quality_thresholds['blur_threshold'])
        quality_metrics['blur_score'] = blur_score
        
        # 2. Brightness assessment
        mean_brightness = roi_image.mean()
        brightness_score = self._normalize_to_range(
            mean_brightness, self.quality_thresholds['brightness_range']
        )
        quality_metrics['brightness_score'] = brightness_score
        
        # 3. Contrast assessment
        contrast = roi_image.std()
        contrast_score = min(1.0, contrast / self.quality_thresholds['contrast_min'])
        quality_metrics['contrast_score'] = contrast_score
        
        # 4. Ridge density in ROI
        ridge_pixels = np.sum(processed > 0)
        total_roi_pixels = np.sum(roi_mask > 0)
        ridge_density = ridge_pixels / max(total_roi_pixels, 1)
        quality_metrics['ridge_density'] = ridge_density
        
        # 5. Overall quality score
        weights = [0.3, 0.2, 0.3, 0.2]  # blur, brightness, contrast, density
        scores = [blur_score, brightness_score, contrast_score, ridge_density]
        quality_score = sum(w * s for w, s in zip(weights, scores))
        
        quality_metrics['quality_score'] = quality_score
        quality_metrics['quality_reasons'] = self._generate_quality_reasons(quality_metrics)
        
        return quality_metrics
    
    def _normalize_to_range(self, value: float, optimal_range: Tuple[float, float]) -> float:
        """Normalize value to [0,1] based on optimal range"""
        min_val, max_val = optimal_range
        if value < min_val or value > max_val:
            # Outside optimal range - penalize
            distance = min(abs(value - min_val), abs(value - max_val))
            return max(0.0, 1.0 - distance / 50.0)  # Gradual penalty
        else:
            # Within optimal range
            return 1.0
    
    def _generate_quality_reasons(self, metrics: Dict) -> List[str]:
        """Generate human-readable quality assessment reasons"""
        reasons = []
        
        if metrics['blur_score'] < 0.5:
            reasons.append("Image appears blurry")
        if metrics['brightness_score'] < 0.5:
            reasons.append("Poor lighting conditions")
        if metrics['contrast_score'] < 0.5:
            reasons.append("Low ridge-valley contrast")
        if metrics['ridge_density'] < 0.3:
            reasons.append("Insufficient ridge information")
            
        return reasons
    
    def _analyze_ridge_structure(self, skeleton: np.ndarray, roi_mask: np.ndarray) -> Dict:
        """Comprehensive ridge structure analysis"""
        ridge_info = {}
        
        # 1. Orientation field analysis
        orientation_field = self._compute_orientation_field(skeleton)
        ridge_info['orientation_field'] = orientation_field
        ridge_info['orientation_consistency'] = self._assess_orientation_consistency(orientation_field)
        
        # 2. Ridge frequency analysis
        ridge_frequency = self._compute_ridge_frequency(skeleton, roi_mask)
        ridge_info['ridge_frequency'] = ridge_frequency
        ridge_info['frequency_score'] = self._assess_frequency_naturalness(ridge_frequency)
        
        # 3. Ridge continuity
        ridge_info['continuity_score'] = self._assess_ridge_continuity(skeleton)
        
        return ridge_info
    
    def _compute_orientation_field(self, skeleton: np.ndarray) -> np.ndarray:
        """Compute local ridge orientation field using gradients"""
        # Sobel gradients
        gx = cv2.Sobel(skeleton, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(skeleton, cv2.CV_64F, 0, 1, ksize=3)
        
        # Local orientation (perpendicular to gradient)
        orientation = np.arctan2(gy, gx) + np.pi/2
        
        # Smooth orientation field
        block_size = self.config['preprocessing']['block_size']
        h, w = skeleton.shape
        
        smoothed_orientation = np.zeros_like(orientation)
        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block = orientation[i:i+block_size, j:j+block_size]
                mean_orientation = np.angle(np.mean(np.exp(1j * 2 * block))) / 2
                smoothed_orientation[i:i+block_size, j:j+block_size] = mean_orientation
        
        return smoothed_orientation
    
    def _assess_orientation_consistency(self, orientation_field: np.ndarray) -> float:
        """Assess consistency of ridge orientation field"""
        # Compute local orientation variance
        block_size = self.config['preprocessing']['block_size']
        h, w = orientation_field.shape
        
        variances = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = orientation_field[i:i+block_size, j:j+block_size]
                # Convert to complex numbers for circular statistics
                complex_angles = np.exp(1j * 2 * block)
                variance = 1 - np.abs(np.mean(complex_angles))
                variances.append(variance)
        
        # Lower variance means higher consistency
        mean_variance = np.mean(variances)
        consistency_score = 1.0 - mean_variance
        
        return max(0.0, consistency_score)
    
    def _compute_ridge_frequency(self, skeleton: np.ndarray, roi_mask: np.ndarray) -> float:
        """Compute dominant ridge frequency using FFT"""
        # Apply ROI mask
        masked_skeleton = cv2.bitwise_and(skeleton, skeleton, mask=roi_mask)
        
        # Compute 2D FFT
        f_transform = np.fft.fft2(masked_skeleton)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Find dominant frequency
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Create radial frequency profile
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Average magnitude at each radius
        max_radius = min(center_x, center_y)
        radial_profile = []
        
        for r in range(1, max_radius):
            mask = (radius >= r - 0.5) & (radius < r + 0.5)
            radial_profile.append(np.mean(magnitude_spectrum[mask]))
        
        # Find peak frequency (excluding DC component)
        if len(radial_profile) > 5:
            peak_idx = np.argmax(radial_profile[2:]) + 2  # Skip low frequencies
            dominant_frequency = peak_idx / max_radius
        else:
            dominant_frequency = 0.15  # Default value
        
        return dominant_frequency
    
    def _assess_frequency_naturalness(self, frequency: float) -> float:
        """Assess if ridge frequency is within natural human range"""
        min_freq, max_freq = self.quality_thresholds['ridge_freq_range']
        
        if min_freq <= frequency <= max_freq:
            return 1.0
        else:
            # Distance from natural range
            if frequency < min_freq:
                distance = min_freq - frequency
            else:
                distance = frequency - max_freq
            
            # Exponential penalty
            return max(0.0, np.exp(-distance * 10))
    
    def _assess_ridge_continuity(self, skeleton: np.ndarray) -> float:
        """Assess ridge continuity and smoothness"""
        # Find all ridge pixels
        ridge_pixels = np.where(skeleton > 0)
        
        if len(ridge_pixels[0]) == 0:
            return 0.0
        
        # Trace ridge lines and assess continuity
        ridge_image = (skeleton > 0).astype(np.uint8)
        
        # Find connected components (ridge lines)
        num_labels, labels = cv2.connectedComponents(ridge_image)
        
        if num_labels <= 1:
            return 0.0
        
        # Analyze each ridge component
        continuity_scores = []
        
        for label in range(1, num_labels):
            component = (labels == label)
            component_size = np.sum(component)
            
            if component_size < 10:  # Too small to analyze
                continue
            
            # Compute component's aspect ratio and curvature
            y_coords, x_coords = np.where(component)
            
            # Fit line to component
            if len(x_coords) > 2:
                coeffs = np.polyfit(x_coords, y_coords, 1)
                fitted_y = np.polyval(coeffs, x_coords)
                mse = np.mean((y_coords - fitted_y) ** 2)
                
                # Lower MSE indicates better continuity
                continuity_score = 1.0 / (1.0 + mse / 10.0)
                continuity_scores.append(continuity_score)
        
        return np.mean(continuity_scores) if continuity_scores else 0.0
    
    def _extract_minutiae_with_confidence(self, skeleton: np.ndarray, ridge_info: Dict) -> List[Dict]:
        """Extract minutiae points with confidence scoring"""
        minutiae = []
        
        # Find ridge pixels
        ridge_pixels = np.where(skeleton > 0)
        
        for i in range(len(ridge_pixels[0])):
            y, x = ridge_pixels[0][i], ridge_pixels[1][i]
            
            # Extract 3x3 neighborhood
            neighborhood = skeleton[max(0, y-1):y+2, max(0, x-1):x+2]
            
            if neighborhood.shape != (3, 3):
                continue
            
            # Compute crossing number
            cn = self._compute_crossing_number(neighborhood)
            
            minutiae_type = None
            if cn == 1:
                minutiae_type = "ending"
            elif cn == 3:
                minutiae_type = "bifurcation"
            else:
                continue  # Not a minutiae
            
            # Compute minutiae angle
            angle = self._compute_minutiae_angle(skeleton, x, y, minutiae_type)
            
            # Compute confidence score
            confidence = self._compute_minutiae_confidence(
                skeleton, x, y, minutiae_type, ridge_info
            )
            
            if confidence >= self.config['minutiae']['min_confidence']:
                minutiae.append({
                    'x': int(x),
                    'y': int(y),
                    'type': minutiae_type,
                    'angle': float(angle),
                    'confidence': float(confidence)
                })
        
        # Remove duplicate minutiae (too close to each other)
        minutiae = self._remove_duplicate_minutiae(minutiae)
        
        # Limit number of minutiae
        if len(minutiae) > self.config['minutiae']['max_minutiae']:
            # Sort by confidence and keep top ones
            minutiae.sort(key=lambda m: m['confidence'], reverse=True)
            minutiae = minutiae[:self.config['minutiae']['max_minutiae']]
        
        return minutiae
    
    def _compute_crossing_number(self, neighborhood: np.ndarray) -> int:
        """Compute crossing number for 3x3 neighborhood"""
        if neighborhood.shape != (3, 3):
            return 0
        
        # Define 8-neighborhood in clockwise order
        neighbors = [
            neighborhood[0, 1],  # Top
            neighborhood[0, 2],  # Top-right
            neighborhood[1, 2],  # Right
            neighborhood[2, 2],  # Bottom-right
            neighborhood[2, 1],  # Bottom
            neighborhood[2, 0],  # Bottom-left
            neighborhood[1, 0],  # Left
            neighborhood[0, 0]   # Top-left
        ]
        
        # Convert to binary
        neighbors = [(1 if p > 0 else 0) for p in neighbors]
        
        # Compute crossing number
        cn = 0
        for i in range(8):
            cn += abs(neighbors[i] - neighbors[(i + 1) % 8])
        
        return cn // 2
    
    def _compute_minutiae_angle(self, skeleton: np.ndarray, x: int, y: int, minutiae_type: str) -> float:
        """Compute orientation angle of minutiae point"""
        h, w = skeleton.shape
        
        # Define search radius
        radius = self.config['minutiae']['neighborhood_size']
        
        # Extract larger neighborhood for angle computation
        y1, y2 = max(0, y - radius), min(h, y + radius + 1)
        x1, x2 = max(0, x - radius), min(w, x + radius + 1)
        
        neighborhood = skeleton[y1:y2, x1:x2]
        
        # Find ridge direction using moments or gradient
        ridge_pixels = np.where(neighborhood > 0)
        
        if len(ridge_pixels[0]) < 3:
            return 0.0  # Not enough points
        
        # Convert to relative coordinates
        ridge_y = ridge_pixels[0] - (y - y1)
        ridge_x = ridge_pixels[1] - (x - x1)
        
        # Compute principal direction using PCA
        if len(ridge_x) > 2:
            coords = np.column_stack((ridge_x, ridge_y))
            coords_centered = coords - coords.mean(axis=0)
            
            try:
                cov_matrix = np.cov(coords_centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                
                # Principal direction (largest eigenvalue)
                principal_vector = eigenvectors[:, np.argmax(eigenvalues)]
                angle = np.arctan2(principal_vector[1], principal_vector[0])
                
                # Normalize angle to [0, 2π]
                if angle < 0:
                    angle += 2 * np.pi
                
                return angle
            except:
                return 0.0
        
        return 0.0
    
    def _compute_minutiae_confidence(self, skeleton: np.ndarray, x: int, y: int, 
                                   minutiae_type: str, ridge_info: Dict) -> float:
        """Compute confidence score for minutiae point"""
        h, w = skeleton.shape
        radius = self.config['minutiae']['neighborhood_size']
        
        # Extract neighborhood
        y1, y2 = max(0, y - radius), min(h, y + radius + 1)
        x1, x2 = max(0, x - radius), min(w, x + radius + 1)
        neighborhood = skeleton[y1:y2, x1:x2]
        
        confidence_factors = []
        
        # Factor 1: Local ridge density
        ridge_density = np.sum(neighborhood > 0) / neighborhood.size
        confidence_factors.append(ridge_density)
        
        # Factor 2: Ridge thickness consistency
        ridge_pixels = neighborhood > 0
        if np.sum(ridge_pixels) > 0:
            # Check if ridges have consistent thickness
            thickness_variance = self._compute_ridge_thickness_variance(neighborhood)
            thickness_score = 1.0 / (1.0 + thickness_variance)
            confidence_factors.append(thickness_score)
        else:
            confidence_factors.append(0.0)
        
        # Factor 3: Orientation consistency with local field
        if 'orientation_field' in ridge_info and ridge_info['orientation_field'] is not None:
            if 0 <= y < ridge_info['orientation_field'].shape[0] and 0 <= x < ridge_info['orientation_field'].shape[1]:
                local_orientation = ridge_info['orientation_field'][y, x]
                # Compute minutiae orientation
                minutiae_angle = self._compute_minutiae_angle(skeleton, x, y, minutiae_type)
                
                # Angular difference
                angle_diff = abs(local_orientation - minutiae_angle)
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)  # Circular distance
                
                orientation_score = 1.0 - (angle_diff / np.pi)
                confidence_factors.append(orientation_score)
            else:
                confidence_factors.append(0.5)  # Neutral score for boundary cases
        else:
            confidence_factors.append(0.5)
        
        # Factor 4: Distance from image boundaries
        boundary_distance = min(x, y, w - x - 1, h - y - 1)
        boundary_score = min(1.0, boundary_distance / 20.0)  # Penalize minutiae near edges
        confidence_factors.append(boundary_score)
        
        # Combine factors with weights
        weights = [0.3, 0.25, 0.25, 0.2]
        confidence = sum(w * f for w, f in zip(weights, confidence_factors))
        
        return max(0.0, min(1.0, confidence))
    
    def _compute_ridge_thickness_variance(self, neighborhood: np.ndarray) -> float:
        """Compute variance in ridge thickness within neighborhood"""
        # This is a simplified implementation
        # In practice, you might want more sophisticated ridge width analysis
        
        ridge_pixels = neighborhood > 0
        if np.sum(ridge_pixels) == 0:
            return 1.0  # High variance (bad)
        
        # Compute distance transform to estimate ridge widths
        distance_transform = ndimage.distance_transform_edt(ridge_pixels)
        
        # Extract ridge centerline (skeleton)
        ridge_center = morphology.skeletonize(ridge_pixels)
        center_pixels = np.where(ridge_center)
        
        if len(center_pixels[0]) == 0:
            return 1.0
        
        # Get ridge widths at centerline points
        widths = []
        for i in range(len(center_pixels[0])):
            y, x = center_pixels[0][i], center_pixels[1][i]
            width = distance_transform[y, x] * 2  # Diameter
            widths.append(width)
        
        if len(widths) < 2:
            return 0.0  # No variance
        
        # Compute coefficient of variation
        mean_width = np.mean(widths)
        std_width = np.std(widths)
        
        if mean_width == 0:
            return 1.0
        
        cv = std_width / mean_width
        return cv
    
    def _remove_duplicate_minutiae(self, minutiae: List[Dict]) -> List[Dict]:
        """Remove minutiae points that are too close to each other"""
        if len(minutiae) <= 1:
            return minutiae
        
        filtered_minutiae = []
        min_distance = 10  # Minimum distance between minutiae
        
        # Sort by confidence (highest first)
        sorted_minutiae = sorted(minutiae, key=lambda m: m['confidence'], reverse=True)
        
        for minutia in sorted_minutiae:
            too_close = False
            
            for existing in filtered_minutiae:
                distance = np.sqrt(
                    (minutia['x'] - existing['x']) ** 2 + 
                    (minutia['y'] - existing['y']) ** 2
                )
                
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered_minutiae.append(minutia)
        
        return filtered_minutiae
    
    def _comprehensive_security_analysis(self, original: np.ndarray, skeleton: np.ndarray,
                                       ridge_info: Dict, minutiae: List[Dict], 
                                       roi_mask: np.ndarray) -> float:
        """Comprehensive spoof detection analysis"""
        security_scores = []
        
        # 1. Texture analysis using LBP
        if self.config['security']['texture_analysis']:
            texture_score = self._analyze_texture_authenticity(original, roi_mask)
            security_scores.append(('texture', texture_score, 0.3))
        
        # 2. Frequency domain analysis
        if self.config['security']['frequency_analysis']:
            frequency_score = self._analyze_frequency_authenticity(original, ridge_info)
            security_scores.append(('frequency', frequency_score, 0.25))
        
        # 3. Orientation field analysis
        if self.config['security']['orientation_analysis']:
            orientation_score = ridge_info.get('orientation_consistency', 0.5)
            security_scores.append(('orientation', orientation_score, 0.2))
        
        # 4. Minutiae pattern analysis
        minutiae_score = self._analyze_minutiae_authenticity(minutiae)
        security_scores.append(('minutiae', minutiae_score, 0.15))
        
        # 5. Ridge quality analysis
        ridge_score = ridge_info.get('continuity_score', 0.5)
        security_scores.append(('ridge_quality', ridge_score, 0.1))
        
        # Compute weighted security score
        if security_scores:
            weighted_sum = sum(score * weight for _, score, weight in security_scores)
            total_weight = sum(weight for _, _, weight in security_scores)
            overall_security = weighted_sum / total_weight
        else:
            overall_security = 0.5
        
        # Convert to spoof risk (invert the score)
        spoof_risk_score = 1.0 - overall_security
        
        return spoof_risk_score
    
    def _analyze_texture_authenticity(self, image: np.ndarray, roi_mask: np.ndarray) -> float:
        """Analyze texture patterns for spoof detection using LBP"""
        # Apply ROI mask
        roi_image = cv2.bitwise_and(image, image, mask=roi_mask)
        
        # Compute Local Binary Pattern
        lbp = feature.local_binary_pattern(roi_image, P=8, R=1, method='uniform')
        
        # Compute LBP histogram
        n_bins = 8 + 2  # Uniform patterns + non-uniform
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-6)
        
        # Real fingerprints have characteristic LBP distributions
        # This is a simplified check - in practice, you'd train a classifier
        
        # Check for uniform distribution (often indicates fake texture)
        entropy = -np.sum(hist * np.log2(hist + 1e-6))
        max_entropy = np.log2(n_bins)
        normalized_entropy = entropy / max_entropy
        
        # Real fingerprints have medium entropy (not too uniform, not too random)
        if 0.3 <= normalized_entropy <= 0.8:
            texture_authenticity = 1.0
        else:
            # Penalize extreme entropy values
            if normalized_entropy < 0.3:
                texture_authenticity = normalized_entropy / 0.3
            else:
                texture_authenticity = (1.0 - normalized_entropy) / 0.2
        
        return max(0.0, min(1.0, texture_authenticity))
    
    def _analyze_frequency_authenticity(self, image: np.ndarray, ridge_info: Dict) -> float:
        """Analyze frequency domain characteristics for authenticity"""
        frequency_score = ridge_info.get('frequency_score', 0.5)
        
        # Additional frequency analysis could include:
        # - Power spectrum analysis
        # - Harmonic content analysis
        # - Frequency stability across the image
        
        return frequency_score
    
    def _analyze_minutiae_authenticity(self, minutiae: List[Dict]) -> float:
        """Analyze minutiae patterns for authenticity"""
        if len(minutiae) == 0:
            return 0.0  # No minutiae is suspicious
        
        authenticity_factors = []
        
        # 1. Minutiae density check
        # Real fingerprints typically have 20-100 minutiae
        minutiae_count = len(minutiae)
        if 20 <= minutiae_count <= 100:
            density_score = 1.0
        elif minutiae_count < 20:
            density_score = minutiae_count / 20.0
        else:
            density_score = max(0.0, 1.0 - (minutiae_count - 100) / 50.0)
        
        authenticity_factors.append(density_score)
        
        # 2. Minutiae type distribution
        endings = sum(1 for m in minutiae if m['type'] == 'ending')
        bifurcations = sum(1 for m in minutiae if m['type'] == 'bifurcation')
        
        if minutiae_count > 0:
            ending_ratio = endings / minutiae_count
            # Real fingerprints typically have more endings than bifurcations
            if 0.6 <= ending_ratio <= 0.8:
                type_score = 1.0
            else:
                type_score = 1.0 - abs(ending_ratio - 0.7) / 0.3
        else:
            type_score = 0.0
        
        authenticity_factors.append(type_score)
        
        # 3. Minutiae confidence distribution
        if minutiae:
            confidences = [m['confidence'] for m in minutiae]
            avg_confidence = np.mean(confidences)
            confidence_std = np.std(confidences)
            
            # Real minutiae should have reasonable confidence distribution
            confidence_score = avg_confidence * (1.0 - min(0.5, confidence_std))
        else:
            confidence_score = 0.0
        
        authenticity_factors.append(confidence_score)
        
        # 4. Spatial distribution analysis
        if len(minutiae) >= 3:
            spatial_score = self._analyze_minutiae_spatial_distribution(minutiae)
        else:
            spatial_score = 0.5
        
        authenticity_factors.append(spatial_score)
        
        # Combine factors
        weights = [0.3, 0.25, 0.25, 0.2]
        overall_authenticity = sum(w * f for w, f in zip(weights, authenticity_factors))
        
        return max(0.0, min(1.0, overall_authenticity))
    
    def _analyze_minutiae_spatial_distribution(self, minutiae: List[Dict]) -> float:
        """Analyze spatial distribution of minutiae for authenticity"""
        if len(minutiae) < 3:
            return 0.5
        
        # Extract coordinates
        coords = np.array([[m['x'], m['y']] for m in minutiae])
        
        # Compute nearest neighbor distances
        from scipy.spatial.distance import pdist
        distances = pdist(coords)
        
        if len(distances) == 0:
            return 0.5
        
        # Analyze distance distribution
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Real minutiae should not be too clustered or too regular
        # Check coefficient of variation
        if mean_distance > 0:
            cv = std_distance / mean_distance
            # Moderate variation is expected in real fingerprints
            if 0.3 <= cv <= 1.0:
                spatial_score = 1.0
            elif cv < 0.3:
                # Too regular (suspicious)
                spatial_score = cv / 0.3
            else:
                # Too random
                spatial_score = max(0.0, 1.0 - (cv - 1.0) / 0.5)
        else:
            spatial_score = 0.0
        
        return spatial_score
    
    def _create_result(self, image_id: str, quality_results: Dict, minutiae: List[Dict],
                      spoof_risk_score: float, spoof_suspected: bool, processing_time: float,
                      ridge_analysis: Dict = None, early_rejection: str = None) -> Dict:
        """Create standardized result dictionary"""
        result = {
            'image_id': image_id,
            'quality_score': quality_results['quality_score'],
            'quality_reasons': quality_results.get('quality_reasons', []),
            'spoof_risk_score': spoof_risk_score,
            'spoof_suspected': spoof_suspected,
            'minutiae_count': len(minutiae),
            'minutiae': minutiae,
            'processing_time_ms': round(processing_time * 1000, 2),
            'timestamp': time.time()
        }
        
        # Add ridge analysis if available
        if ridge_analysis:
            result.update({
                'orientation_consistency': ridge_analysis.get('orientation_consistency', 0.0),
                'ridge_frequency': ridge_analysis.get('ridge_frequency', 0.0),
                'frequency_score': ridge_analysis.get('frequency_score', 0.0),
                'continuity_score': ridge_analysis.get('continuity_score', 0.0)
            })
        
        # Add early rejection reason if applicable
        if early_rejection:
            result['early_rejection'] = early_rejection
        
        return result
    
    def _create_error_result(self, image_path: str, error_message: str) -> Dict:
        """Create error result dictionary"""
        return {
            'image_id': Path(image_path).stem,
            'error': error_message,
            'quality_score': 0.0,
            'spoof_risk_score': 1.0,
            'spoof_suspected': True,
            'minutiae_count': 0,
            'minutiae': [],
            'processing_time_ms': 0.0,
            'timestamp': time.time()
        }
    
    def batch_analyze(self, input_dir: str, output_dir: str, 
                     file_pattern: str = "*.tif") -> Dict:
        """
        Batch process multiple fingerprint images
        
        Args:
            input_dir: Directory containing fingerprint images
            output_dir: Directory to save results
            file_pattern: File pattern to match (e.g., "*.tif", "*.png")
            
        Returns:
            Summary statistics of the batch processing
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all matching image files
        image_files = list(input_path.rglob(file_pattern))
        
        if not image_files:
            logger.warning(f"No files found matching pattern {file_pattern} in {input_dir}")
            return {'processed': 0, 'errors': 0, 'results': []}
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images in parallel
        results = []
        errors = 0
        
        with ThreadPoolExecutor(max_workers=self.config['performance']['parallel_workers']) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(self.analyze_fingerprint, str(img_file), img_file.stem): img_file 
                for img_file in image_files
            }
            
            # Collect results
            for future in future_to_file:
                img_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Save individual result
                    result_file = output_path / f"{img_file.stem}_analysis.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
                        
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")
                    errors += 1
        
        # Save batch summary
        summary = {
            'processed': len(results),
            'errors': errors,
            'total_files': len(image_files),
            'results': results,
            'statistics': self._compute_batch_statistics(results)
        }
        
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Batch processing complete: {len(results)} processed, {errors} errors")
        
        return summary
    
    def _compute_batch_statistics(self, results: List[Dict]) -> Dict:
        """Compute statistics across batch results"""
        if not results:
            return {}
        
        # Filter out error results
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results to analyze'}
        
        stats = {}
        
        # Quality statistics
        quality_scores = [r['quality_score'] for r in valid_results]
        stats['quality'] = {
            'mean': np.mean(quality_scores),
            'std': np.std(quality_scores),
            'min': np.min(quality_scores),
            'max': np.max(quality_scores),
            'median': np.median(quality_scores)
        }
        
        # Spoof detection statistics
        spoof_scores = [r['spoof_risk_score'] for r in valid_results]
        spoof_flags = [r['spoof_suspected'] for r in valid_results]
        
        stats['security'] = {
            'mean_spoof_risk': np.mean(spoof_scores),
            'spoof_detection_rate': np.mean(spoof_flags),
            'total_suspected_spoofs': sum(spoof_flags)
        }
        
        # Minutiae statistics
        minutiae_counts = [r['minutiae_count'] for r in valid_results]
        stats['minutiae'] = {
            'mean_count': np.mean(minutiae_counts),
            'std_count': np.std(minutiae_counts),
            'min_count': np.min(minutiae_counts),
            'max_count': np.max(minutiae_counts)
        }
        
        # Performance statistics
        processing_times = [r['processing_time_ms'] for r in valid_results]
        stats['performance'] = {
            'mean_time_ms': np.mean(processing_times),
            'total_time_ms': np.sum(processing_times),
            'throughput_images_per_second': len(valid_results) / (np.sum(processing_times) / 1000)
        }
        
        return stats


def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Fingerprint Analysis')
    parser.add_argument('--input', required=True, help='Input image file or directory')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    parser.add_argument('--pattern', default='*.tif', help='File pattern for batch processing')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AdvancedFingerprintAnalyzer()
    
    if args.batch:
        # Batch processing
        summary = analyzer.batch_analyze(args.input, args.output, args.pattern)
        print(f"Processed {summary['processed']} images with {summary['errors']} errors")
        print(f"Results saved to {args.output}")
    else:
        # Single image processing
        result = analyzer.analyze_fingerprint(args.input)
        
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        result_file = output_path / f"{Path(args.input).stem}_analysis.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Analysis complete. Result saved to {result_file}")
        print(f"Quality Score: {result['quality_score']:.3f}")
        print(f"Spoof Risk: {result['spoof_risk_score']:.3f}")
        print(f"Minutiae Count: {result['minutiae_count']}")


if __name__ == "__main__":
    main()
