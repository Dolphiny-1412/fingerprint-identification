"""
Person C - Minutiae Detection & Analysis
========================================

Implements minutiae extraction with overlay generation and quality assessment.
"""

import os
import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from PIL import Image, ImageDraw
import time

class MinutiaeDetector:
    """
    Advanced minutiae detection with quality assessment and overlay generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        """Default configuration for minutiae detection"""
        return {
            'preprocessing': {
                'target_size': (256, 256),
                'gaussian_blur': (3, 3),
                'gaussian_sigma': 0.5,
                'morphology_kernel_size': 3
            },
            'minutiae': {
                'min_confidence': 0.5,
                'max_minutiae': 200,
                'neighborhood_size': 5,
                'min_distance': 10
            },
            'quality': {
                'min_contrast': 0.3,
                'min_clarity': 50.0,
                'max_noise_ratio': 0.15
            }
        }
    
    def detect_minutiae(self, image_path: str) -> Dict:
        """
        Detect minutiae in a fingerprint image
        
        Args:
            image_path: Path to fingerprint image
            
        Returns:
            Dictionary containing minutiae data and overlay path
        """
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return self._create_error_result("Could not load image")
            
            # Store original image dimensions
            original_height, original_width = image.shape[:2]
            
            # Preprocess image
            processed_image, roi_mask = self._preprocess_image(image)
            
            # Assess image quality
            quality_metrics = self._assess_image_quality(image, processed_image, roi_mask)
            
            # Extract minutiae
            minutiae_points = self._extract_minutiae_points(processed_image, roi_mask)
            
            # Scale minutiae coordinates back to original image size
            target_size = self.config['preprocessing']['target_size']
            scale_x = original_width / target_size[0]
            scale_y = original_height / target_size[1]
            
            scaled_minutiae = []
            for minutia in minutiae_points:
                scaled_minutiae.append({
                    'x': int(minutia['x'] * scale_x),
                    'y': int(minutia['y'] * scale_y),
                    'type': minutia['type'],
                    'angle': minutia['angle'],
                    'confidence': minutia['confidence']
                })
            
            # Generate overlay image using scaled coordinates
            overlay_path = self._generate_overlay_image(image, scaled_minutiae, image_path)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Ensure all minutiae points are JSON-serializable
            serializable_minutiae = []
            for minutia in scaled_minutiae:
                serializable_minutiae.append({
                    'x': int(minutia['x']),
                    'y': int(minutia['y']),
                    'type': str(minutia['type']),
                    'angle': float(minutia['angle']),
                    'confidence': float(minutia['confidence'])
                })
            
            return {
                "minutiae": serializable_minutiae,
                "quality_score": float(quality_metrics['overall_quality']),
                "spoof_suspected": bool(quality_metrics['spoof_suspected']),
                "spoof_risk_score": float(quality_metrics.get('spoof_risk_score', 0.0)),
                "overlay_path": str(overlay_path),
                "processing_time_ms": round(processing_time * 1000, 2),
                "minutiae_count": int(len(minutiae_points)),
                "quality_reasons": [str(reason) for reason in quality_metrics.get('quality_reasons', [])]
            }
            
        except Exception as e:
            return self._create_error_result(str(e))
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess fingerprint image for minutiae detection
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (processed_image, roi_mask)
        """
        # Resize to target size
        target_size = self.config['preprocessing']['target_size']
        image = cv2.resize(image, target_size)
        
        # Normalize intensity
        normalized = cv2.equalizeHist(image)
        
        # Detect ROI (Region of Interest)
        roi_mask = self._detect_roi(normalized)
        
        # Apply ROI mask - ensure mask and image have same size and type
        if roi_mask.shape != normalized.shape:
            roi_mask = cv2.resize(roi_mask, (normalized.shape[1], normalized.shape[0]))
        if roi_mask.dtype != normalized.dtype:
            roi_mask = roi_mask.astype(normalized.dtype)
        masked = cv2.bitwise_and(normalized, normalized, mask=roi_mask)
        
        # Gaussian smoothing
        blurred = cv2.GaussianBlur(
            masked, 
            self.config['preprocessing']['gaussian_blur'],
            self.config['preprocessing']['gaussian_sigma']
        )
        
        # Binarization
        binary = self._adaptive_binarization(blurred)
        
        # Morphological cleanup
        cleaned = self._morphological_cleanup(binary)
        
        # Skeletonization
        skeleton = self._skeletonize(cleaned)
        
        return skeleton, roi_mask
    
    def _detect_roi(self, image: np.ndarray) -> np.ndarray:
        """Detect region of interest (fingerprint area)"""
        # Otsu thresholding
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
    
    def _adaptive_binarization(self, image: np.ndarray) -> np.ndarray:
        """Adaptive binarization"""
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
    
    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """Skeletonize binary image"""
        from skimage.morphology import skeletonize
        
        # Convert to boolean
        binary_bool = binary > 0
        
        # Skeletonize (ensure ridges are foreground/True)
        skeleton = skeletonize(binary_bool)
        
        # Convert back to uint8
        return (skeleton * 255).astype(np.uint8)
    
    def _assess_image_quality(self, original: np.ndarray, processed: np.ndarray, roi_mask: np.ndarray) -> Dict:
        """Assess image quality metrics"""
        # Apply ROI mask to original - ensure mask and image have same size and type
        if roi_mask.shape != original.shape:
            roi_mask = cv2.resize(roi_mask, (original.shape[1], original.shape[0]))
        if roi_mask.dtype != original.dtype:
            roi_mask = roi_mask.astype(original.dtype)
        roi_image = cv2.bitwise_and(original, original, mask=roi_mask)
        
        # 1. Blur assessment (Laplacian variance)
        laplacian_var = cv2.Laplacian(roi_image, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 100.0)
        
        # 2. Contrast assessment
        contrast = roi_image.std()
        contrast_score = min(1.0, contrast / 50.0)
        
        # 3. Brightness assessment
        mean_brightness = roi_image.mean()
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
        
        # 4. Ridge density (compute from binarized ridge map, not skeleton)
        blurred = cv2.GaussianBlur(roi_image, (5, 5), 0)
        _, ridge_map = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ridge_map = cv2.bitwise_and(ridge_map, ridge_map, mask=roi_mask)
        ridge_pixels = np.sum(ridge_map > 0)
        total_roi_pixels = np.sum(roi_mask > 0)
        ridge_density = ridge_pixels / max(total_roi_pixels, 1)
        
        # 5. Overall quality score
        overall_quality = (blur_score + contrast_score + brightness_score + ridge_density) / 4.0
        
        # 6. Spoof detection (require at least two low indicators to reduce false positives)
        low_flags = [
            blur_score < 0.15,
            contrast_score < 0.15,
            brightness_score < 0.15,
            ridge_density < 0.08,
        ]
        spoof_suspected = sum(low_flags) >= 2
        
        # Generate quality reasons
        quality_reasons = []
        if blur_score < 0.5:
            quality_reasons.append("Image appears blurry")
        if contrast_score < 0.5:
            quality_reasons.append("Low ridge-valley contrast")
        if brightness_score < 0.5:
            quality_reasons.append("Poor lighting conditions")
        if ridge_density < 0.3:
            quality_reasons.append("Insufficient ridge information")
        
        return {
            'blur_score': blur_score,
            'contrast_score': contrast_score,
            'brightness_score': brightness_score,
            'ridge_density': ridge_density,
            'overall_quality': overall_quality,
            'spoof_suspected': spoof_suspected,
            'quality_reasons': quality_reasons
        }
    
    def _extract_minutiae_points(self, skeleton: np.ndarray, roi_mask: np.ndarray) -> List[Dict]:
        """Extract minutiae points from skeletonized image"""
        minutiae = []

        # Compute distance to ROI edge to suppress boundary artifacts
        # Normalize mask to {0,1} for distance transform
        roi_norm = (roi_mask > 0).astype(np.uint8)
        # Distance inside ROI to the nearest zero (edge)
        dist_to_edge = cv2.distanceTransform(roi_norm, cv2.DIST_L2, 3)
        roi_margin = self.config['minutiae'].get('roi_margin', 12)
        
        # Find ridge pixels
        ridge_pixels = np.where(skeleton > 0)
        
        for i in range(len(ridge_pixels[0])):
            y, x = ridge_pixels[0][i], ridge_pixels[1][i]
            
            # Skip if not in ROI
            if roi_mask[y, x] == 0:
                continue
            
            # Extract 3x3 neighborhood
            neighborhood = skeleton[max(0, y-1):y+2, max(0, x-1):x+2]
            
            if neighborhood.shape != (3, 3):
                continue
            
            # Compute crossing number using canonical 8-neighborhood ordering
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
            confidence = self._compute_minutiae_confidence(skeleton, x, y, minutiae_type)
            
            # Skip minutiae with very low confidence
            if confidence < self.config['minutiae']['min_confidence']:
                continue
            
            minutiae.append({
                'x': int(x),
                'y': int(y),
                'type': minutiae_type,
                'angle': float(angle),
                'confidence': float(confidence)
            })
        
        # Remove duplicate minutiae
        minutiae = self._remove_duplicate_minutiae(minutiae)
        
        # Limit number of minutiae
        if len(minutiae) > self.config['minutiae']['max_minutiae']:
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
        radius = self.config['minutiae']['neighborhood_size']
        
        # Extract larger neighborhood for angle computation
        y1, y2 = max(0, y - radius), min(h, y + radius + 1)
        x1, x2 = max(0, x - radius), min(w, x + radius + 1)
        
        neighborhood = skeleton[y1:y2, x1:x2]
        
        # Find ridge direction using gradient
        ridge_pixels = np.where(neighborhood > 0)
        
        if len(ridge_pixels[0]) < 3:
            return 0.0
        
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
    
    def _compute_minutiae_confidence(self, skeleton: np.ndarray, x: int, y: int, minutiae_type: str) -> float:
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
        
        # Factor 2: Distance from image boundaries
        boundary_distance = min(x, y, w - x - 1, h - y - 1)
        boundary_score = min(1.0, boundary_distance / 20.0)
        confidence_factors.append(boundary_score)
        
        # Factor 3: Ridge continuity
        ridge_pixels = neighborhood > 0
        if np.sum(ridge_pixels) > 0:
            # Check if ridges have consistent thickness
            continuity_score = self._assess_ridge_continuity(neighborhood)
            confidence_factors.append(continuity_score)
        else:
            confidence_factors.append(0.0)
        
        # Combine factors
        confidence = np.mean(confidence_factors)
        
        return max(0.0, min(1.0, confidence))
    
    def _assess_ridge_continuity(self, neighborhood: np.ndarray) -> float:
        """Assess ridge continuity in neighborhood"""
        # Simple continuity check based on connected components
        ridge_pixels = neighborhood > 0
        
        if np.sum(ridge_pixels) == 0:
            return 0.0
        
        # Count connected components
        num_labels, labels = cv2.connectedComponents(ridge_pixels.astype(np.uint8))
        
        # Fewer components indicate better continuity
        if num_labels <= 1:
            return 0.0
        
        # Score based on number of components (fewer is better)
        continuity_score = 1.0 / num_labels
        
        return continuity_score
    
    def _remove_duplicate_minutiae(self, minutiae: List[Dict]) -> List[Dict]:
        """Remove minutiae points that are too close to each other"""
        if len(minutiae) <= 1:
            return minutiae
        
        filtered_minutiae = []
        min_distance = self.config['minutiae']['min_distance']
        
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
    
    def _generate_overlay_image(self, original_image: np.ndarray, minutiae: List[Dict], image_path: str) -> str:
        """Generate overlay image with minutiae points marked"""
        # Create output directory
        os.makedirs("src/data/derived", exist_ok=True)
        
        # Generate output filename
        base_name = Path(image_path).stem
        timestamp = int(time.time())
        overlay_filename = f"{base_name}_overlay_{timestamp}.png"
        overlay_path = os.path.join("src/data/derived", overlay_filename)
        
        # Convert to PIL Image for drawing
        if len(original_image.shape) == 2:
            # Grayscale to RGB
            pil_image = Image.fromarray(original_image).convert('RGB')
        else:
            pil_image = Image.fromarray(original_image)
        
        # Create overlay
        overlay = Image.new('RGBA', pil_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw minutiae points
        for i, minutia in enumerate(minutiae):
            x, y = minutia['x'], minutia['y']
            minutiae_type = minutia['type']
            confidence = minutia['confidence']
            
            # Ensure coordinates are within image bounds
            if not (0 <= x < pil_image.width and 0 <= y < pil_image.height):
                continue
            
            # Color based on type
            if minutiae_type == "ending":
                color = (255, 0, 0, 255)  # Bright red for endings
                radius = 4
            else:  # bifurcation
                color = (0, 255, 0, 255)   # Bright green for bifurcations
                radius = 5
            
            # Draw filled circle
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=(0, 0, 0, 255)  # Black outline
            )
            
            # Draw a small cross to mark the exact center
            draw.line([(x-2, y), (x+2, y)], fill=(255, 255, 255, 255), width=1)
            draw.line([(x, y-2), (x, y+2)], fill=(255, 255, 255, 255), width=1)
        
        # Combine original and overlay
        combined = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
        combined = combined.convert('RGB')
        
        # Save overlay image
        combined.save(overlay_path)
        
        return overlay_path
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result dictionary"""
        return {
            "minutiae": [],
            "quality_score": 0.0,
            "spoof_suspected": True,
            "spoof_risk_score": 1.0,
            "overlay_path": "",
            "processing_time_ms": 0.0,
            "minutiae_count": 0,
            "quality_reasons": [str(error_message)],
            "error": str(error_message)
        }

# Global detector instance
_detector_instance = None

def get_detector() -> MinutiaeDetector:
    """Get or create global detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MinutiaeDetector()
    return _detector_instance

def detect_minutiae(image_path: str) -> Dict:
    """
    Main function to detect minutiae in a fingerprint image
    
    Args:
        image_path: Path to fingerprint image
        
    Returns:
        Minutiae detection result dictionary
    """
    detector = get_detector()
    return detector.detect_minutiae(image_path)

if __name__ == "__main__":
    # Test with sample images
    test_images = [
        "Fingerprint D/archive/fingerprints/DB1_B/101_1.tif",
        "Fingerprint D/archive/fingerprints/DB1_B/102_1.tif"
    ]
    
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\n🔍 Testing minutiae detection with {test_image}")
            result = detect_minutiae(test_image)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ Minutiae detected: {result['minutiae_count']}")
                print(f"   Quality score: {result['quality_score']:.3f}")
                print(f"   Spoof suspected: {result['spoof_suspected']}")
                print(f"   Overlay saved to: {result['overlay_path']}")
                
                # Show sample minutiae
                if result['minutiae']:
                    print(f"   Sample minutiae:")
                    for i, minutia in enumerate(result['minutiae'][:3]):
                        print(f"     {i+1}. {minutia['type']} at ({minutia['x']}, {minutia['y']}) - confidence: {minutia['confidence']:.3f}")
        else:
            print(f"⚠️ Test image not found: {test_image}")
