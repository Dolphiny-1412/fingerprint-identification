"""
Step 2: Enhanced Spoof Detection Logic
Comprehensive anti-spoofing system for fingerprint authentication
"""

import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from skimage import filters, morphology, feature
from skimage.filters import gaussian
from scipy import ndimage
from scipy.stats import entropy
import os
import sys

# Add the minutiae folder to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'minutiae'))
from detector import AdvancedFingerprintAnalyzer


class SpoofDetectionSystem:
    """
    Comprehensive spoof detection system with multiple security layers
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.advanced_analyzer = AdvancedFingerprintAnalyzer()
        
    def _default_config(self) -> Dict:
        """Default configuration for spoof detection"""
        return {
            'texture_analysis': {
                'lbp_radius': 3,
                'lbp_points': 8,
                'haralick_features': True,
                'gabor_filters': True
            },
            'frequency_analysis': {
                'fft_threshold': 0.1,
                'frequency_bands': [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7)]
            },
            'quality_checks': {
                'min_contrast': 0.2,
                'min_sharpness': 50.0,
                'max_noise': 0.15
            },
            'liveness_detection': {
                'pore_analysis': True,
                'ridge_continuity': True,
                'moisture_detection': True
            },
            'thresholds': {
                'spoof_high': 0.85,    # Increased from 0.7
                'spoof_medium': 0.65,  # Increased from 0.4
                'spoof_low': 0.45      # Increased from 0.2
            }
        }
    
    def detect_spoof(self, image_path: str) -> Dict:
        """
        Comprehensive spoof detection for a fingerprint image
        
        Args:
            image_path: Path to the fingerprint image
            
        Returns:
            Dict containing spoof detection results
        """
        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return self._create_error_result("Could not load image")
            
            # Run advanced analysis first
            advanced_result = self.advanced_analyzer.analyze_fingerprint(image_path)
            
            # Initialize results
            spoof_results = {
                'image_path': image_path,
                'overall_spoof_risk': 0.0,
                'spoof_detected': False,
                'confidence': 'low',
                'detection_methods': {},
                'recommendations': []
            }
            
            # 1. Texture Analysis
            texture_score = self._analyze_texture_authenticity(image)
            spoof_results['detection_methods']['texture_analysis'] = {
                'score': texture_score,
                'description': 'Analyzes texture patterns for artificial materials'
            }
            
            # 2. Frequency Domain Analysis
            frequency_score = self._analyze_frequency_domain(image)
            spoof_results['detection_methods']['frequency_analysis'] = {
                'score': frequency_score,
                'description': 'Detects artificial patterns in frequency domain'
            }
            
            # 3. Quality-based Detection
            quality_score = self._analyze_quality_indicators(image)
            spoof_results['detection_methods']['quality_analysis'] = {
                'score': quality_score,
                'description': 'Checks for quality indicators of real fingerprints'
            }
            
            # 4. Liveness Detection
            liveness_score = self._detect_liveness_features(image)
            spoof_results['detection_methods']['liveness_detection'] = {
                'score': liveness_score,
                'description': 'Detects natural skin characteristics'
            }
            
            # 5. Ridge Pattern Analysis
            ridge_score = self._analyze_ridge_patterns(image)
            spoof_results['detection_methods']['ridge_analysis'] = {
                'score': ridge_score,
                'description': 'Analyzes ridge pattern naturalness'
            }
            
            # Calculate overall spoof risk
            scores = [
                texture_score,
                frequency_score,
                quality_score,
                liveness_score,
                ridge_score
            ]
            
            # Weighted average (texture and liveness are most important, ridge less important)
            weights = [0.25, 0.20, 0.20, 0.25, 0.10]  # Reduced ridge weight from 0.15 to 0.10
            overall_risk = np.average(scores, weights=weights)
            
            spoof_results['overall_spoof_risk'] = overall_risk
            
            # Determine if spoof is detected
            if overall_risk >= self.config['thresholds']['spoof_high']:
                spoof_results['spoof_detected'] = True
                spoof_results['confidence'] = 'high'
            elif overall_risk >= self.config['thresholds']['spoof_medium']:
                spoof_results['spoof_detected'] = True
                spoof_results['confidence'] = 'medium'
            elif overall_risk >= self.config['thresholds']['spoof_low']:
                spoof_results['confidence'] = 'suspicious'
            
            # Generate recommendations
            spoof_results['recommendations'] = self._generate_recommendations(
                spoof_results['detection_methods'], overall_risk
            )
            
            return spoof_results
            
        except Exception as e:
            return self._create_error_result(f"Error in spoof detection: {str(e)}")
    
    def _analyze_texture_authenticity(self, image: np.ndarray) -> float:
        """Analyze texture patterns for authenticity"""
        try:
            # Local Binary Pattern analysis
            lbp = feature.local_binary_pattern(
                image, 
                self.config['texture_analysis']['lbp_points'],
                self.config['texture_analysis']['lbp_radius'],
                method='uniform'
            )
            
            # Calculate LBP histogram
            lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
            lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
            
            # Calculate entropy (real fingerprints have higher entropy)
            lbp_entropy = entropy(lbp_hist + 1e-10)
            
            # Normalize entropy score (0-1, higher is more authentic)
            texture_score = min(1.0, lbp_entropy / 2.0)
            
            return texture_score
            
        except Exception as e:
            print(f"Texture analysis error: {e}")
            return 0.5
    
    def _analyze_frequency_domain(self, image: np.ndarray) -> float:
        """Analyze frequency domain for artificial patterns"""
        try:
            # Apply FFT
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Analyze frequency distribution
            h, w = magnitude_spectrum.shape
            center_y, center_x = h // 2, w // 2
            
            # Check for regular patterns (artificial fingerprints often have regular patterns)
            frequency_bands = self.config['frequency_analysis']['frequency_bands']
            band_scores = []
            
            for low_freq, high_freq in frequency_bands:
                # Create frequency band mask
                y, x = np.ogrid[:h, :w]
                mask = ((x - center_x)**2 + (y - center_y)**2 >= (low_freq * min(h, w))**2) & \
                       ((x - center_x)**2 + (y - center_y)**2 <= (high_freq * min(h, w))**2)
                
                band_energy = np.mean(magnitude_spectrum[mask])
                band_scores.append(band_energy)
            
            # Calculate frequency regularity score
            freq_variance = np.var(band_scores)
            freq_score = max(0, 1 - freq_variance / 10.0)  # Lower variance = more regular = more suspicious
            
            return freq_score
            
        except Exception as e:
            print(f"Frequency analysis error: {e}")
            return 0.5
    
    def _analyze_quality_indicators(self, image: np.ndarray) -> float:
        """Analyze quality indicators that suggest real vs fake fingerprints"""
        try:
            # 1. Contrast analysis
            contrast = np.std(image) / 255.0
            contrast_score = min(1.0, contrast / self.config['quality_checks']['min_contrast'])
            
            # 2. Sharpness analysis (Laplacian variance)
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            sharpness = np.var(laplacian)
            sharpness_score = min(1.0, sharpness / self.config['quality_checks']['min_sharpness'])
            
            # 3. Noise analysis
            # Apply Gaussian blur and compare with original
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            noise = np.mean(np.abs(image.astype(float) - blurred.astype(float)))
            noise_score = max(0, 1 - noise / (255 * self.config['quality_checks']['max_noise']))
            
            # Combine quality scores
            quality_score = (contrast_score + sharpness_score + noise_score) / 3.0
            
            return quality_score
            
        except Exception as e:
            print(f"Quality analysis error: {e}")
            return 0.5
    
    def _detect_liveness_features(self, image: np.ndarray) -> float:
        """Detect liveness features characteristic of real skin"""
        try:
            # 1. Pore detection (real fingerprints have pores)
            # Apply morphological operations to detect small dots
            kernel = np.ones((2, 2), np.uint8)
            eroded = cv2.erode(image, kernel, iterations=1)
            dilated = cv2.dilate(eroded, kernel, iterations=1)
            pores = cv2.subtract(image, dilated)
            
            # Count potential pores
            pore_threshold = np.mean(pores) + 2 * np.std(pores)
            pore_count = np.sum(pores > pore_threshold)
            pore_score = min(1.0, pore_count / 1000.0)  # Normalize
            
            # 2. Ridge continuity analysis
            # Real fingerprints have more continuous ridges
            edges = cv2.Canny(image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate average contour length
            if contours:
                avg_length = np.mean([len(contour) for contour in contours])
                continuity_score = min(1.0, avg_length / 100.0)
            else:
                continuity_score = 0.0
            
            # 3. Moisture detection (real skin has moisture patterns)
            # Apply Gabor filter to detect moisture-like patterns
            moisture_score = self._detect_moisture_patterns(image)
            
            # Combine liveness scores
            liveness_score = (pore_score + continuity_score + moisture_score) / 3.0
            
            return liveness_score
            
        except Exception as e:
            print(f"Liveness detection error: {e}")
            return 0.5
    
    def _detect_moisture_patterns(self, image: np.ndarray) -> float:
        """Detect moisture patterns characteristic of real skin"""
        try:
            # Apply multiple Gabor filters to detect moisture patterns
            angles = [0, 45, 90, 135]
            frequencies = [0.1, 0.2, 0.3]
            
            moisture_responses = []
            for angle in angles:
                for freq in frequencies:
                    # Create Gabor filter
                    kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 2*np.pi*freq, 0.5, 0)
                    response = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                    moisture_responses.append(np.var(response))
            
            # Calculate moisture pattern score
            moisture_variance = np.var(moisture_responses)
            moisture_score = min(1.0, moisture_variance / 1000.0)
            
            return moisture_score
            
        except Exception as e:
            print(f"Moisture detection error: {e}")
            return 0.5
    
    def _analyze_ridge_patterns(self, image: np.ndarray) -> float:
        """Analyze ridge patterns for naturalness"""
        try:
            # 1. Ridge orientation consistency
            # Calculate gradient
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate orientation
            orientation = np.arctan2(grad_y, grad_x)
            
            # Analyze orientation consistency
            orientation_hist, _ = np.histogram(orientation, bins=36, range=(-np.pi, np.pi))
            orientation_entropy = entropy(orientation_hist + 1e-10)
            
            # Real fingerprints have more consistent orientations
            orientation_score = max(0, 1 - orientation_entropy / 3.0)
            
            # 2. Ridge spacing analysis
            # Apply distance transform to analyze ridge spacing
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            
            # Analyze spacing regularity
            spacing_variance = np.var(dist_transform)
            spacing_score = max(0, 1 - spacing_variance / 100.0)
            
            # Combine ridge pattern scores
            ridge_score = (orientation_score + spacing_score) / 2.0
            
            return ridge_score
            
        except Exception as e:
            print(f"Ridge pattern analysis error: {e}")
            return 0.5
    
    def _generate_recommendations(self, detection_methods: Dict, overall_risk: float) -> List[str]:
        """Generate security recommendations based on detection results"""
        recommendations = []
        
        if overall_risk >= 0.7:
            recommendations.append("🚨 HIGH SPOOF RISK: Implement additional authentication methods")
            recommendations.append("🔒 Require multi-factor authentication")
            recommendations.append("📱 Request secondary verification (SMS, email)")
        
        elif overall_risk >= 0.4:
            recommendations.append("⚠️ MEDIUM SPOOF RISK: Increase security measures")
            recommendations.append("🔍 Request additional fingerprint samples")
            recommendations.append("⏱️ Implement time-based access restrictions")
        
        elif overall_risk >= 0.2:
            recommendations.append("🤔 SUSPICIOUS: Monitor access patterns")
            recommendations.append("📊 Log authentication attempts for analysis")
        
        # Method-specific recommendations
        if detection_methods.get('texture_analysis', {}).get('score', 1.0) < 0.3:
            recommendations.append("🎨 Texture analysis suggests artificial material")
        
        if detection_methods.get('liveness_detection', {}).get('score', 1.0) < 0.3:
            recommendations.append("💧 Liveness detection indicates potential spoof")
        
        if detection_methods.get('quality_analysis', {}).get('score', 1.0) < 0.3:
            recommendations.append("📸 Image quality suggests potential manipulation")
        
        return recommendations
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result dictionary"""
        return {
            'error': error_message,
            'overall_spoof_risk': 1.0,  # Assume worst case on error
            'spoof_detected': True,
            'confidence': 'unknown',
            'detection_methods': {},
            'recommendations': ['❌ Error in spoof detection - manual verification required']
        }


def test_spoof_detection(image_path: str) -> Dict:
    """
    Test spoof detection on a single image
    
    Args:
        image_path: Path to the fingerprint image
        
    Returns:
        Spoof detection results
    """
    detector = SpoofDetectionSystem()
    results = detector.detect_spoof(image_path)
    
    print(f"🛡️ Spoof Detection Results for: {os.path.basename(image_path)}")
    print("=" * 60)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return results
    
    print(f"📊 Overall Spoof Risk: {results['overall_spoof_risk']:.3f}")
    print(f"🎯 Spoof Detected: {'🚨 YES' if results['spoof_detected'] else '✅ NO'}")
    print(f"🛡️ Confidence: {results['confidence'].upper()}")
    
    print(f"\n🔍 Detection Method Results:")
    for method, data in results['detection_methods'].items():
        score = data['score']
        emoji = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
        print(f"   {emoji} {method.replace('_', ' ').title()}: {score:.3f}")
        print(f"      {data['description']}")
    
    if results['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in results['recommendations']:
            print(f"   • {rec}")
    
    return results


def main():
    """Main function to test spoof detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spoof Detection System')
    parser.add_argument('--image', help='Path to fingerprint image')
    parser.add_argument('--batch', help='Directory for batch processing')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch processing
        detector = SpoofDetectionSystem()
        results = []
        
        for filename in os.listdir(args.batch):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                image_path = os.path.join(args.batch, filename)
                print(f"Processing: {filename}")
                result = detector.detect_spoof(image_path)
                results.append(result)
        
        # Summary
        spoof_detected = sum(1 for r in results if r.get('spoof_detected', False))
        avg_risk = np.mean([r.get('overall_spoof_risk', 0) for r in results])
        
        print(f"\n📈 Batch Processing Summary:")
        print(f"   • Total images: {len(results)}")
        print(f"   • Spoofs detected: {spoof_detected}")
        print(f"   • Average risk: {avg_risk:.3f}")
        
    elif args.image:
        # Single image processing
        test_spoof_detection(args.image)
    else:
        print("Please provide either --image or --batch argument")
        print("Example: python spoof_detection.py --image path/to/image.tif")
        print("Example: python spoof_detection.py --batch path/to/directory")


if __name__ == "__main__":
    main()
