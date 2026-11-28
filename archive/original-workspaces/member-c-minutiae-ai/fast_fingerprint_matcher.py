#!/usr/bin/env python3
"""
Fast Fingerprint Matcher
========================

Optimized version focusing on speed and essential functionality.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import json
import os

class FastFingerprintMatcher:
    """
    Fast fingerprint matcher using optimized feature extraction
    """
    
    def __init__(self, match_threshold=40.0):
        self.match_threshold = match_threshold
        
        # Initialize ORB detector with fewer features for speed
        self.orb = cv2.ORB_create(nfeatures=200, scaleFactor=1.2, nlevels=4)
        
        # Initialize SIFT if available
        try:
            self.sift = cv2.SIFT_create(nfeatures=100)
            self.use_sift = True
        except AttributeError:
            self.sift = None
            self.use_sift = False
    
    def extract_fast_features(self, image: np.ndarray) -> Dict:
        """
        Fast feature extraction using only ORB and basic image statistics
        
        Args:
            image: Input grayscale fingerprint image
            
        Returns:
            Dictionary containing extracted features
        """
        # Preprocess image quickly
        enhanced = cv2.equalizeHist(image)
        
        # Extract ORB features
        orb_kp, orb_desc = self.orb.detectAndCompute(enhanced, None)
        
        # Extract basic image statistics
        stats = {
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'skew': float(self._compute_skewness(image)),
            'kurtosis': float(self._compute_kurtosis(image))
        }
        
        # Simple ridge count estimation
        ridge_count = self._estimate_ridge_count(image)
        
        template = {
            'orb_keypoints': len(orb_kp),
            'orb_descriptors': orb_desc.tolist() if orb_desc is not None else [],
            'image_stats': stats,
            'ridge_count': ridge_count,
            'image_shape': image.shape
        }
        
        return template
    
    def _compute_skewness(self, image: np.ndarray) -> float:
        """Compute image skewness"""
        flat = image.flatten().astype(np.float32)
        mean_val = np.mean(flat)
        std_val = np.std(flat)
        if std_val == 0:
            return 0.0
        return np.mean(((flat - mean_val) / std_val) ** 3)
    
    def _compute_kurtosis(self, image: np.ndarray) -> float:
        """Compute image kurtosis"""
        flat = image.flatten().astype(np.float32)
        mean_val = np.mean(flat)
        std_val = np.std(flat)
        if std_val == 0:
            return 0.0
        return np.mean(((flat - mean_val) / std_val) ** 4) - 3
    
    def _estimate_ridge_count(self, image: np.ndarray) -> int:
        """Fast ridge count estimation using gradient analysis"""
        # Apply Sobel filters
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Count high gradient regions (approximate ridge count)
        threshold = np.percentile(gradient_mag, 80)
        ridge_pixels = np.sum(gradient_mag > threshold)
        
        # Normalize by image area
        normalized_count = int(ridge_pixels / 1000)  # Rough estimate
        
        return normalized_count
    
    def match_orb_features(self, template1: Dict, template2: Dict) -> float:
        """
        Fast ORB feature matching
        
        Args:
            template1: First template
            template2: Second template
            
        Returns:
            Similarity score (0-100)
        """
        desc1 = np.array(template1['orb_descriptors'], dtype=np.uint8)
        desc2 = np.array(template2['orb_descriptors'], dtype=np.uint8)
        
        if desc1.size == 0 or desc2.size == 0:
            return 0.0
        
        # Use BFMatcher for ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        
        # Filter good matches
        good_matches = [m for m in matches if m.distance < 50]
        
        # Calculate similarity
        similarity = (len(good_matches) / max(len(desc1), len(desc2))) * 100.0
        
        return similarity
    
    def match_image_stats(self, template1: Dict, template2: Dict) -> float:
        """
        Match based on image statistics
        
        Args:
            template1: First template
            template2: Second template
            
        Returns:
            Similarity score (0-100)
        """
        stats1 = template1['image_stats']
        stats2 = template2['image_stats']
        
        # Compute normalized differences
        mean_diff = abs(stats1['mean'] - stats2['mean']) / 255.0
        std_diff = abs(stats1['std'] - stats2['std']) / 255.0
        skew_diff = abs(stats1['skew'] - stats2['skew']) / 10.0  # Normalize skewness
        kurtosis_diff = abs(stats1['kurtosis'] - stats2['kurtosis']) / 10.0
        
        # Ridge count similarity
        ridge_diff = abs(template1['ridge_count'] - template2['ridge_count'])
        max_ridge = max(template1['ridge_count'], template2['ridge_count'])
        ridge_similarity = 1.0 - (ridge_diff / max(max_ridge, 1))
        
        # Combined similarity
        stat_similarity = 1.0 - np.mean([mean_diff, std_diff, skew_diff, kurtosis_diff])
        combined_similarity = (stat_similarity * 0.6 + ridge_similarity * 0.4) * 100.0
        
        return max(0.0, combined_similarity)
    
    def match_fingerprints_fast(self, image1: np.ndarray, image2: np.ndarray) -> Dict:
        """
        Fast fingerprint matching
        
        Args:
            image1: First fingerprint image
            image2: Second fingerprint image
            
        Returns:
            Matching results
        """
        print("🚀 Fast fingerprint matching...")
        
        # Extract features (fast)
        template1 = self.extract_fast_features(image1)
        template2 = self.extract_fast_features(image2)
        
        print(f"Template 1: {template1['orb_keypoints']} ORB features")
        print(f"Template 2: {template2['orb_keypoints']} ORB features")
        
        # Match ORB features
        orb_score = self.match_orb_features(template1, template2)
        
        # Match image statistics
        stats_score = self.match_image_stats(template1, template2)
        
        # Combine scores
        combined_score = orb_score * 0.7 + stats_score * 0.3
        
        # Make decision
        is_match = combined_score >= self.match_threshold
        confidence = min(combined_score / 100.0, 1.0)
        
        return {
            'is_match': is_match,
            'combined_score': combined_score,
            'confidence': confidence,
            'orb_score': orb_score,
            'stats_score': stats_score,
            'match_threshold': self.match_threshold,
            'template1': template1,
            'template2': template2
        }
    
    def visualize_fast_matching(self, image1: np.ndarray, image2: np.ndarray, 
                               result: Dict, save_path: str = None) -> None:
        """Create simple visualization of matching results"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Fast Matching: {"MATCH" if result["is_match"] else "NO MATCH"} '
                    f'({result["combined_score"]:.1f}%)', fontsize=14, fontweight='bold')
        
        # Image 1
        axes[0].imshow(image1, cmap='gray')
        axes[0].set_title('Fingerprint 1')
        axes[0].axis('off')
        
        # Image 2
        axes[1].imshow(image2, cmap='gray')
        axes[1].set_title('Fingerprint 2')
        axes[1].axis('off')
        
        # Scores
        methods = ['ORB', 'Stats', 'Combined']
        scores = [result['orb_score'], result['stats_score'], result['combined_score']]
        colors = ['lightblue', 'lightgreen', 'red' if result['is_match'] else 'gray']
        
        bars = axes[2].bar(methods, scores, color=colors, alpha=0.7)
        axes[2].axhline(y=self.match_threshold, color='red', linestyle='--', 
                       label=f'Threshold ({self.match_threshold}%)')
        axes[2].set_ylabel('Score (%)')
        axes[2].set_title('Matching Scores')
        axes[2].legend()
        axes[2].set_ylim(0, 100)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{score:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✅ Visualization saved: {save_path}")
        
        plt.close()

def test_fast_matching():
    """Test the fast matching system"""
    print("⚡ Testing Fast Fingerprint Matching System")
    print("=" * 50)
    
    # Load test images
    data_root = r"C:\Users\alifa\Downloads\fingerprint_model-20250811T194218Z-1-001\fingerprint_model\data\archive\fingerprints\DB1_B"
    
    image1_path = f"{data_root}/101_1.tif"
    image2_path = f"{data_root}/101_2.tif"  # Same finger
    image3_path = f"{data_root}/102_1.tif"  # Different finger
    
    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    image3 = cv2.imread(image3_path, cv2.IMREAD_GRAYSCALE)
    
    if any(img is None for img in [image1, image2, image3]):
        print("❌ Failed to load test images")
        return
    
    # Initialize fast matcher
    matcher = FastFingerprintMatcher(match_threshold=30.0)
    
    # Test 1: Same finger (should match)
    print("\n🧪 Test 1: Same finger (101_1 vs 101_2)")
    result1 = matcher.match_fingerprints_fast(image1, image2)
    
    print(f"Result: {'MATCH' if result1['is_match'] else 'NO MATCH'}")
    print(f"Combined Score: {result1['combined_score']:.1f}%")
    print(f"ORB Score: {result1['orb_score']:.1f}%")
    print(f"Stats Score: {result1['stats_score']:.1f}%")
    
    # Visualize
    matcher.visualize_fast_matching(image1, image2, result1, "fast_match_same.png")
    
    # Test 2: Different finger (should not match)
    print("\n🧪 Test 2: Different finger (101_1 vs 102_1)")
    result2 = matcher.match_fingerprints_fast(image1, image3)
    
    print(f"Result: {'MATCH' if result2['is_match'] else 'NO MATCH'}")
    print(f"Combined Score: {result2['combined_score']:.1f}%")
    print(f"ORB Score: {result2['orb_score']:.1f}%")
    print(f"Stats Score: {result2['stats_score']:.1f}%")
    
    # Visualize
    matcher.visualize_fast_matching(image1, image3, result2, "fast_match_different.png")
    
    # Save results
    results = {
        'same_finger': {
            'images': ['101_1.tif', '101_2.tif'],
            'is_match': bool(result1['is_match']),
            'combined_score': float(result1['combined_score']),
            'orb_score': float(result1['orb_score']),
            'stats_score': float(result1['stats_score'])
        },
        'different_finger': {
            'images': ['101_1.tif', '102_1.tif'],
            'is_match': bool(result2['is_match']),
            'combined_score': float(result2['combined_score']),
            'orb_score': float(result2['orb_score']),
            'stats_score': float(result2['stats_score'])
        }
    }
    
    with open('fast_matching_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Fast matching test complete!")
    print(f"📊 Results: fast_matching_results.json")
    print(f"🎨 Visualizations: fast_match_*.png")

if __name__ == "__main__":
    test_fast_matching()
