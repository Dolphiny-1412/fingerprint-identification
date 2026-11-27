#!/usr/bin/env python3
"""
Fingerprint Identity Matching Engine
===================================

Implements geometric minutiae-based matching with:
1. Minutiae template comparison
2. Translation and rotation invariant alignment
3. ORB/SIFT features as backup
4. Similarity scoring and match decision

Based on forensic standards for identity verification.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import json
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
import math

from minutiae_extraction import MinutiaeExtractor

class FingerprintMatcher:
    """
    Professional fingerprint identity matching using minutiae and backup features
    """
    
    def __init__(self, match_threshold=40.0, min_minutiae=8, max_distance=20):
        """
        Initialize fingerprint matcher
        
        Args:
            match_threshold: Similarity threshold for match decision (0-100)
            min_minutiae: Minimum minutiae required for matching
            max_distance: Maximum distance for minutiae pairing
        """
        self.match_threshold = match_threshold
        self.min_minutiae = min_minutiae
        self.max_distance = max_distance
        
        # Initialize minutiae extractor
        self.minutiae_extractor = MinutiaeExtractor()
        
        # Initialize ORB detector for backup features
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # Initialize SIFT detector (if available)
        try:
            self.sift = cv2.SIFT_create(nfeatures=500)
            self.use_sift = True
        except AttributeError:
            self.sift = None
            self.use_sift = False
    
    def extract_template(self, image: np.ndarray) -> Dict:
        """
        Extract complete fingerprint template including minutiae and backup features
        
        Args:
            image: Input grayscale fingerprint image
            
        Returns:
            Dictionary containing all extracted features
        """
        template = {}
        
        # Extract minutiae
        minutiae_data = self.minutiae_extractor.extract_minutiae(image)
        template['minutiae'] = minutiae_data['minutiae']
        template['minutiae_count'] = len(minutiae_data['minutiae'])
        
        # Extract ORB features as backup
        orb_keypoints, orb_descriptors = self.orb.detectAndCompute(image, None)
        template['orb_keypoints'] = []
        for kp in orb_keypoints:
            template['orb_keypoints'].append({
                'x': float(kp.pt[0]),
                'y': float(kp.pt[1]),
                'angle': float(kp.angle),
                'response': float(kp.response)
            })
        template['orb_descriptors'] = orb_descriptors.tolist() if orb_descriptors is not None else []
        
        # Extract SIFT features if available
        if self.use_sift:
            sift_keypoints, sift_descriptors = self.sift.detectAndCompute(image, None)
            template['sift_keypoints'] = []
            for kp in sift_keypoints:
                template['sift_keypoints'].append({
                    'x': float(kp.pt[0]),
                    'y': float(kp.pt[1]),
                    'angle': float(kp.angle),
                    'response': float(kp.response)
                })
            template['sift_descriptors'] = sift_descriptors.tolist() if sift_descriptors is not None else []
        
        # Add image metadata
        template['image_shape'] = image.shape
        template['template_quality'] = self._assess_template_quality(template)
        
        return template
    
    def _assess_template_quality(self, template: Dict) -> float:
        """Assess overall template quality"""
        minutiae_count = template['minutiae_count']
        orb_count = len(template['orb_keypoints'])
        
        # Quality based on feature counts
        minutiae_quality = min(minutiae_count / 20.0, 1.0)  # Ideal: 20+ minutiae
        orb_quality = min(orb_count / 100.0, 1.0)  # Ideal: 100+ ORB features
        
        # Combined quality score
        overall_quality = (minutiae_quality * 0.7 + orb_quality * 0.3)
        return overall_quality
    
    def match_minutiae(self, template1: Dict, template2: Dict) -> Dict:
        """
        Match minutiae between two templates using geometric alignment
        
        Args:
            template1: First fingerprint template
            template2: Second fingerprint template
            
        Returns:
            Dictionary containing matching results
        """
        minutiae1 = template1['minutiae']
        minutiae2 = template2['minutiae']
        
        if len(minutiae1) < self.min_minutiae or len(minutiae2) < self.min_minutiae:
            return {
                'matched_pairs': [],
                'similarity_score': 0.0,
                'alignment': None,
                'match_ratio': 0.0
            }
        
        # Find best alignment using RANSAC-like approach
        best_alignment = None
        best_matches = []
        best_score = 0.0
        
        # Try multiple alignment hypotheses
        for i in range(min(50, len(minutiae1))):  # Sample reference points
            for j in range(min(50, len(minutiae2))):
                ref1 = minutiae1[i]
                ref2 = minutiae2[j]
                
                # Compute alignment transformation
                alignment = self._compute_alignment(ref1, ref2)
                
                # Find matches under this alignment
                matches = self._find_matches_with_alignment(minutiae1, minutiae2, alignment)
                
                # Score this alignment
                score = len(matches) / max(len(minutiae1), len(minutiae2))
                
                if score > best_score:
                    best_score = score
                    best_matches = matches
                    best_alignment = alignment
        
        # Convert to similarity percentage
        similarity_score = best_score * 100.0
        
        return {
            'matched_pairs': best_matches,
            'similarity_score': similarity_score,
            'alignment': best_alignment,
            'match_ratio': best_score
        }
    
    def _compute_alignment(self, ref1: Dict, ref2: Dict) -> Dict:
        """
        Compute alignment transformation between two reference minutiae
        
        Args:
            ref1: Reference minutia from template 1
            ref2: Reference minutia from template 2
            
        Returns:
            Alignment transformation parameters
        """
        # Translation
        dx = ref2['x'] - ref1['x']
        dy = ref2['y'] - ref1['y']
        
        # Rotation (difference in ridge directions)
        angle_diff = ref2['angle'] - ref1['angle']
        
        return {
            'translation': (dx, dy),
            'rotation': angle_diff,
            'scale': 1.0  # Assume no scaling for now
        }
    
    def _apply_alignment(self, minutia: Dict, alignment: Dict) -> Tuple[float, float, float]:
        """Apply alignment transformation to a minutia"""
        x, y = minutia['x'], minutia['y']
        angle = minutia['angle']
        
        # Apply rotation
        cos_theta = math.cos(alignment['rotation'])
        sin_theta = math.sin(alignment['rotation'])
        
        x_rot = x * cos_theta - y * sin_theta
        y_rot = x * sin_theta + y * cos_theta
        
        # Apply translation
        x_final = x_rot + alignment['translation'][0]
        y_final = y_rot + alignment['translation'][1]
        
        # Apply rotation to angle
        angle_final = angle + alignment['rotation']
        
        return x_final, y_final, angle_final
    
    def _find_matches_with_alignment(self, minutiae1: List[Dict], minutiae2: List[Dict], 
                                   alignment: Dict) -> List[Tuple[int, int]]:
        """Find minutiae matches under given alignment"""
        matches = []
        
        for i, m1 in enumerate(minutiae1):
            # Transform m1 according to alignment
            x1_aligned, y1_aligned, angle1_aligned = self._apply_alignment(m1, alignment)
            
            best_match = None
            best_distance = float('inf')
            
            for j, m2 in enumerate(minutiae2):
                # Check type compatibility
                if m1['type'] != m2['type']:
                    continue
                
                # Compute distance after alignment
                pos_distance = euclidean([x1_aligned, y1_aligned], [m2['x'], m2['y']])
                angle_distance = abs(angle1_aligned - m2['angle'])
                
                # Normalize angle distance
                angle_distance = min(angle_distance, 2 * math.pi - angle_distance)
                
                # Combined distance
                total_distance = pos_distance + angle_distance * 10  # Weight angle more
                
                if total_distance < self.max_distance and total_distance < best_distance:
                    best_distance = total_distance
                    best_match = j
            
            if best_match is not None:
                matches.append((i, best_match))
        
        return matches
    
    def match_orb_features(self, template1: Dict, template2: Dict) -> Dict:
        """
        Match ORB features between templates as backup method
        
        Args:
            template1: First template
            template2: Second template
            
        Returns:
            ORB matching results
        """
        desc1 = np.array(template1['orb_descriptors'], dtype=np.uint8)
        desc2 = np.array(template2['orb_descriptors'], dtype=np.uint8)
        
        if desc1.size == 0 or desc2.size == 0:
            return {'matches': [], 'similarity_score': 0.0}
        
        # Use BFMatcher for ORB features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Calculate similarity score
        good_matches = [m for m in matches if m.distance < 50]  # Distance threshold
        similarity_score = (len(good_matches) / max(len(desc1), len(desc2))) * 100.0
        
        return {
            'matches': len(good_matches),
            'total_keypoints1': len(desc1),
            'total_keypoints2': len(desc2),
            'similarity_score': similarity_score
        }
    
    def match_sift_features(self, template1: Dict, template2: Dict) -> Dict:
        """
        Match SIFT features between templates (if available)
        
        Args:
            template1: First template
            template2: Second template
            
        Returns:
            SIFT matching results
        """
        if not self.use_sift or 'sift_descriptors' not in template1:
            return {'matches': 0, 'similarity_score': 0.0}
        
        desc1 = np.array(template1['sift_descriptors'], dtype=np.float32)
        desc2 = np.array(template2['sift_descriptors'], dtype=np.float32)
        
        if desc1.size == 0 or desc2.size == 0:
            return {'matches': 0, 'similarity_score': 0.0}
        
        # Use FLANN matcher for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        similarity_score = (len(good_matches) / max(len(desc1), len(desc2))) * 100.0
        
        return {
            'matches': len(good_matches),
            'total_keypoints1': len(desc1),
            'total_keypoints2': len(desc2),
            'similarity_score': similarity_score
        }
    
    def match_fingerprints(self, image1: np.ndarray, image2: np.ndarray) -> Dict:
        """
        Complete fingerprint matching pipeline
        
        Args:
            image1: First fingerprint image
            image2: Second fingerprint image
            
        Returns:
            Comprehensive matching results
        """
        print("🔍 Extracting templates...")
        
        # Extract templates
        template1 = self.extract_template(image1)
        template2 = self.extract_template(image2)
        
        print(f"Template 1: {template1['minutiae_count']} minutiae, "
              f"{len(template1['orb_keypoints'])} ORB features")
        print(f"Template 2: {template2['minutiae_count']} minutiae, "
              f"{len(template2['orb_keypoints'])} ORB features")
        
        # Minutiae matching
        print("🎯 Matching minutiae...")
        minutiae_result = self.match_minutiae(template1, template2)
        
        # ORB feature matching
        print("🔧 Matching ORB features...")
        orb_result = self.match_orb_features(template1, template2)
        
        # SIFT feature matching (if available)
        sift_result = {'similarity_score': 0.0}
        if self.use_sift:
            print("🎨 Matching SIFT features...")
            sift_result = self.match_sift_features(template1, template2)
        
        # Combine scores with weights
        minutiae_weight = 0.6
        orb_weight = 0.3
        sift_weight = 0.1
        
        combined_score = (minutiae_result['similarity_score'] * minutiae_weight +
                         orb_result['similarity_score'] * orb_weight +
                         sift_result['similarity_score'] * sift_weight)
        
        # Make match decision
        is_match = combined_score >= self.match_threshold
        confidence = min(combined_score / 100.0, 1.0)
        
        return {
            'is_match': is_match,
            'combined_score': combined_score,
            'confidence': confidence,
            'match_threshold': self.match_threshold,
            'minutiae_matching': minutiae_result,
            'orb_matching': orb_result,
            'sift_matching': sift_result,
            'template1_quality': template1['template_quality'],
            'template2_quality': template2['template_quality']
        }
    
    def visualize_matching(self, image1: np.ndarray, image2: np.ndarray, 
                          match_result: Dict, save_path: Optional[str] = None) -> np.ndarray:
        """
        Create visualization of matching results
        
        Args:
            image1: First fingerprint image
            image2: Second fingerprint image
            match_result: Output from match_fingerprints()
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Fingerprint Matching Results\n'
                    f'Match: {"YES" if match_result["is_match"] else "NO"} '
                    f'(Score: {match_result["combined_score"]:.1f}%)', 
                    fontsize=16, fontweight='bold')
        
        # Image 1
        ax = axes[0, 0]
        ax.imshow(image1, cmap='gray')
        ax.set_title('Fingerprint 1')
        ax.axis('off')
        
        # Image 2
        ax = axes[0, 1]
        ax.imshow(image2, cmap='gray')
        ax.set_title('Fingerprint 2')
        ax.axis('off')
        
        # Matching scores
        ax = axes[1, 0]
        methods = ['Minutiae', 'ORB', 'SIFT', 'Combined']
        scores = [
            match_result['minutiae_matching']['similarity_score'],
            match_result['orb_matching']['similarity_score'],
            match_result['sift_matching']['similarity_score'],
            match_result['combined_score']
        ]
        colors = ['skyblue', 'lightgreen', 'orange', 'red' if match_result['is_match'] else 'gray']
        
        bars = ax.bar(methods, scores, color=colors, alpha=0.7)
        ax.axhline(y=self.match_threshold, color='red', linestyle='--', 
                  label=f'Threshold ({self.match_threshold}%)')
        ax.set_ylabel('Similarity Score (%)')
        ax.set_title('Matching Scores by Method')
        ax.legend()
        ax.set_ylim(0, 100)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{score:.1f}%', ha='center', va='bottom')
        
        # Template quality comparison
        ax = axes[1, 1]
        qualities = [match_result['template1_quality'], match_result['template2_quality']]
        quality_labels = ['Template 1', 'Template 2']
        
        bars = ax.bar(quality_labels, qualities, color=['lightblue', 'lightcoral'], alpha=0.7)
        ax.set_ylabel('Template Quality')
        ax.set_title('Template Quality Assessment')
        ax.set_ylim(0, 1)
        
        # Add quality labels
        for bar, quality in zip(bars, qualities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{quality:.3f}', ha='center', va='bottom')
        
        # Add detailed statistics
        stats_text = f"""Matching Statistics:
Minutiae Matches: {len(match_result['minutiae_matching']['matched_pairs'])}
ORB Matches: {match_result['orb_matching']['matches']}
SIFT Matches: {match_result['sift_matching']['matches']}
Decision: {"MATCH" if match_result['is_match'] else "NO MATCH"}
Confidence: {match_result['confidence']:.3f}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Matching visualization saved: {save_path}")
        
        # Convert to image array
        fig.canvas.draw()
        try:
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img_array = img_array[:, :, :3]
        
        plt.close(fig)
        return img_array

def test_fingerprint_matching():
    """Test fingerprint matching system"""
    print("🔍 Testing Fingerprint Identity Matching System")
    print("=" * 60)
    
    # Test with two images from same finger (should match)
    data_root = r"C:\Users\alifa\Downloads\fingerprint_model-20250811T194218Z-1-001\fingerprint_model\data\archive\fingerprints\DB1_B"
    
    image1_path = f"{data_root}/101_1.tif"  # Same finger
    image2_path = f"{data_root}/101_2.tif"  # Same finger
    image3_path = f"{data_root}/102_1.tif"  # Different finger
    
    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    image3 = cv2.imread(image3_path, cv2.IMREAD_GRAYSCALE)
    
    if any(img is None for img in [image1, image2, image3]):
        print("❌ Failed to load test images")
        return
    
    # Initialize matcher
    matcher = FingerprintMatcher(match_threshold=30.0, min_minutiae=5)
    
    # Test 1: Same finger (should match)
    print("\n🧪 Test 1: Same finger comparison")
    print("-" * 40)
    result1 = matcher.match_fingerprints(image1, image2)
    
    print(f"Match Result: {'MATCH' if result1['is_match'] else 'NO MATCH'}")
    print(f"Combined Score: {result1['combined_score']:.1f}%")
    print(f"Confidence: {result1['confidence']:.3f}")
    
    # Save visualization
    matcher.visualize_matching(image1, image2, result1, "match_test_same_finger.png")
    
    # Test 2: Different finger (should not match)
    print("\n🧪 Test 2: Different finger comparison")
    print("-" * 40)
    result2 = matcher.match_fingerprints(image1, image3)
    
    print(f"Match Result: {'MATCH' if result2['is_match'] else 'NO MATCH'}")
    print(f"Combined Score: {result2['combined_score']:.1f}%")
    print(f"Confidence: {result2['confidence']:.3f}")
    
    # Save visualization
    matcher.visualize_matching(image1, image3, result2, "match_test_different_finger.png")
    
    # Save detailed results
    results_json = {
        'same_finger_test': {
            'images': ['101_1.tif', '101_2.tif'],
            'is_match': bool(result1['is_match']),
            'combined_score': float(result1['combined_score']),
            'confidence': float(result1['confidence']),
            'minutiae_score': float(result1['minutiae_matching']['similarity_score']),
            'orb_score': float(result1['orb_matching']['similarity_score'])
        },
        'different_finger_test': {
            'images': ['101_1.tif', '102_1.tif'],
            'is_match': bool(result2['is_match']),
            'combined_score': float(result2['combined_score']),
            'confidence': float(result2['confidence']),
            'minutiae_score': float(result2['minutiae_matching']['similarity_score']),
            'orb_score': float(result2['orb_matching']['similarity_score'])
        }
    }
    
    with open('fingerprint_matching_test.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Fingerprint matching test complete!")
    print(f"📊 Results saved: fingerprint_matching_test.json")
    print(f"🎨 Visualizations saved: match_test_*.png")

if __name__ == "__main__":
    test_fingerprint_matching()
