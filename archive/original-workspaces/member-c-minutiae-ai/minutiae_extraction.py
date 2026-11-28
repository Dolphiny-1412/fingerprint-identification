#!/usr/bin/env python3
"""
Minutiae Extraction Module
=========================

Implements geometric minutiae detection using:
1. Morphological thinning
2. Crossing Number (CN) algorithm for minutiae classification
3. Ridge endings and bifurcations detection
4. Quality-based filtering

Based on forensic standards for fingerprint analysis.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import skeletonize, thin
from typing import List, Dict, Tuple, Optional
import json

class MinutiaeExtractor:
    """
    Professional-grade minutiae extraction using thinning + CN algorithm
    """
    
    def __init__(self, min_ridge_length=10, quality_threshold=0.3):
        """
        Initialize minutiae extractor
        
        Args:
            min_ridge_length: Minimum ridge length to consider valid
            quality_threshold: Quality threshold for minutiae filtering
        """
        self.min_ridge_length = min_ridge_length
        self.quality_threshold = quality_threshold
        
        # Minutiae type codes
        self.RIDGE_ENDING = 1
        self.BIFURCATION = 2
        self.UNKNOWN = 0
        
        # CN patterns for minutiae classification
        self.CN_PATTERNS = {
            1: self.RIDGE_ENDING,    # CN=1 → Ridge ending
            3: self.BIFURCATION,     # CN=3 → Bifurcation
        }
    
    def preprocess_for_minutiae(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for minutiae extraction
        
        Args:
            image: Input grayscale fingerprint image
            
        Returns:
            Tuple of (binary_image, quality_map)
        """
        # Normalize image
        image = cv2.equalizeHist(image)
        
        # Gaussian smoothing
        smoothed = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        # Improved adaptive thresholding
        binary = cv2.adaptiveThreshold(
            smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 8
        )
        
        # More aggressive morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Remove small components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_size = 50  # Minimum component size
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                binary[labels == i] = 0
        
        # Compute quality map using local variance
        quality_map = self._compute_quality_map(image)
        
        return binary.astype(bool), quality_map
    
    def _compute_quality_map(self, image: np.ndarray, block_size: int = 16) -> np.ndarray:
        """Compute local quality map based on variance"""
        h, w = image.shape
        quality_map = np.zeros_like(image, dtype=np.float32)
        
        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block = image[i:i+block_size, j:j+block_size]
                variance = np.var(block.astype(np.float32))
                
                # Normalize variance to 0-1 range
                quality = min(variance / 1000.0, 1.0)
                
                quality_map[i:i+block_size, j:j+block_size] = quality
        
        return quality_map
    
    def thin_ridges(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Apply morphological thinning to get ridge skeleton
        
        Args:
            binary_image: Binary fingerprint image
            
        Returns:
            Thinned skeleton image
        """
        print(f"🦴 Thinning binary image with {np.sum(binary_image)} white pixels")
        
        # Convert to proper format for skeletonization
        skeleton = skeletonize(binary_image)
        
        print(f"   After skeletonization: {np.sum(skeleton)} white pixels")
        
        # Convert to uint8 format
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        
        print(f"   Final skeleton: {np.sum(skeleton_uint8 > 0)} white pixels")
        
        return skeleton_uint8
    
    def crossing_number(self, skeleton: np.ndarray, x: int, y: int) -> int:
        """
        Compute Crossing Number (CN) at point (x, y)
        
        CN = 0.5 * Σ|P(i) - P(i+1)| for i = 0 to 7
        where P(i) are the 8-connected neighbors
        
        Args:
            skeleton: Thinned ridge skeleton
            x, y: Coordinates to compute CN
            
        Returns:
            Crossing number (0, 1, 2, 3, or higher)
        """
        # 8-connected neighborhood offsets (clockwise)
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),  # Top row
            (0, 1),                       # Right
            (1, 1), (1, 0), (1, -1),     # Bottom row  
            (0, -1)                       # Left
        ]
        
        h, w = skeleton.shape
        
        # Get neighbor values (with boundary checking)
        neighbor_values = []
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                neighbor_values.append(skeleton[nx, ny])
            else:
                neighbor_values.append(0)  # Assume background outside image
        
        # Convert to binary (0 or 1)
        neighbor_values = [1 if val > 0 else 0 for val in neighbor_values]
        
        # Compute crossing number
        cn = 0
        for i in range(8):
            cn += abs(neighbor_values[i] - neighbor_values[(i + 1) % 8])
        
        return cn // 2  # CN is half the sum of differences
    
    def detect_minutiae_candidates(self, skeleton: np.ndarray) -> List[Dict]:
        """
        Detect minutiae candidates using improved algorithm
        
        Args:
            skeleton: Thinned ridge skeleton
            
        Returns:
            List of minutiae candidates with positions and types
        """
        minutiae = []
        h, w = skeleton.shape
        
        print(f"🔍 Scanning skeleton of size {h}x{w}")
        print(f"Skeleton has {np.sum(skeleton > 0)} white pixels")
        
        # Scan skeleton for potential minutiae points
        candidates_found = 0
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if skeleton[x, y] > 0:  # Ridge pixel
                    # Count 8-connected neighbors (simpler approach)
                    neighbors = skeleton[x-1:x+2, y-1:y+2]
                    neighbor_count = np.sum(neighbors > 0) - 1  # Subtract center pixel
                    
                    # Classify based on neighbor count
                    if neighbor_count == 1:
                        minutiae_type = self.RIDGE_ENDING
                        candidates_found += 1
                    elif neighbor_count >= 3:
                        minutiae_type = self.BIFURCATION
                        candidates_found += 1
                    else:
                        continue  # Normal ridge pixel
                    
                    minutiae.append({
                        'x': x,
                        'y': y,
                        'type': minutiae_type,
                        'cn': neighbor_count,
                        'angle': self._estimate_ridge_direction(skeleton, x, y),
                        'quality': 1.0  # Will be updated later
                    })
        
        print(f"✅ Found {len(minutiae)} minutiae candidates")
        endings = sum(1 for m in minutiae if m['type'] == self.RIDGE_ENDING)
        bifurcations = sum(1 for m in minutiae if m['type'] == self.BIFURCATION)
        print(f"   - Ridge endings: {endings}")
        print(f"   - Bifurcations: {bifurcations}")
        
        return minutiae
    
    def _estimate_ridge_direction(self, skeleton: np.ndarray, x: int, y: int, radius: int = 5) -> float:
        """
        Estimate ridge direction at minutiae point
        
        Args:
            skeleton: Ridge skeleton
            x, y: Minutiae coordinates
            radius: Neighborhood radius for direction estimation
            
        Returns:
            Ridge direction in radians
        """
        h, w = skeleton.shape
        
        # Extract local neighborhood
        x1 = max(0, x - radius)
        x2 = min(h, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(w, y + radius + 1)
        
        local_region = skeleton[x1:x2, y1:y2]
        
        # Find ridge pixels in neighborhood
        ridge_pixels = np.where(local_region > 0)
        
        if len(ridge_pixels[0]) < 3:
            return 0.0  # Not enough pixels for reliable direction
        
        # Convert to centered coordinates
        ridge_x = ridge_pixels[0] - (x - x1)
        ridge_y = ridge_pixels[1] - (y - y1)
        
        # Compute principal direction using PCA
        if len(ridge_x) > 1:
            # Center the points
            mean_x = np.mean(ridge_x)
            mean_y = np.mean(ridge_y)
            centered_x = ridge_x - mean_x
            centered_y = ridge_y - mean_y
            
            # Compute covariance matrix
            cov_xx = np.mean(centered_x * centered_x)
            cov_xy = np.mean(centered_x * centered_y)
            cov_yy = np.mean(centered_y * centered_y)
            
            # Compute principal direction
            if cov_xx != cov_yy:
                angle = 0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)
            else:
                angle = np.pi / 4 if cov_xy > 0 else -np.pi / 4
            
            return angle
        
        return 0.0
    
    def filter_minutiae(self, minutiae: List[Dict], quality_map: np.ndarray, 
                       skeleton: np.ndarray) -> List[Dict]:
        """
        Filter minutiae based on quality and geometric constraints
        
        Args:
            minutiae: List of minutiae candidates
            quality_map: Quality map of the image
            skeleton: Ridge skeleton
            
        Returns:
            Filtered list of high-quality minutiae
        """
        print(f"🔍 Filtering {len(minutiae)} minutiae candidates...")
        filtered_minutiae = []
        
        for i, m in enumerate(minutiae):
            x, y = m['x'], m['y']
            
            # Check quality at minutiae location
            local_quality = quality_map[x, y]
            m['quality'] = local_quality
            
            # More lenient quality threshold for testing
            if local_quality < self.quality_threshold:
                if i < 5:  # Debug first few
                    print(f"  Rejected minutia {i}: quality {local_quality:.3f} < {self.quality_threshold}")
                continue
            
            # Check for spurious minutiae near image borders
            h, w = skeleton.shape
            border_distance = min(x, y, h - x - 1, w - y - 1)
            if border_distance < 5:  # Reduced from 10
                if i < 5:
                    print(f"  Rejected minutia {i}: too close to border (distance: {border_distance})")
                continue
            
            # Check ridge length for ridge endings (more lenient)
            if m['type'] == self.RIDGE_ENDING:
                ridge_length = self._measure_ridge_length(skeleton, x, y)
                if ridge_length < self.min_ridge_length:
                    if i < 5:
                        print(f"  Rejected ridge ending {i}: length {ridge_length} < {self.min_ridge_length}")
                    continue
            
            # Check for duplicate minutiae (reduced distance threshold)
            is_duplicate = False
            for existing in filtered_minutiae:
                distance = np.sqrt((x - existing['x'])**2 + (y - existing['y'])**2)
                if distance < 5:  # Reduced from 8
                    # Keep the higher quality one
                    if local_quality > existing['quality']:
                        filtered_minutiae.remove(existing)
                    else:
                        is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_minutiae.append(m)
        
        print(f"✅ Kept {len(filtered_minutiae)} high-quality minutiae")
        return filtered_minutiae
    
    def _measure_ridge_length(self, skeleton: np.ndarray, start_x: int, start_y: int) -> int:
        """
        Measure ridge length from a ridge ending
        
        Args:
            skeleton: Ridge skeleton
            start_x, start_y: Starting point (ridge ending)
            
        Returns:
            Ridge length in pixels
        """
        visited = np.zeros_like(skeleton, dtype=bool)
        length = 0
        current_x, current_y = start_x, start_y
        
        while True:
            visited[current_x, current_y] = True
            length += 1
            
            # Find next ridge pixel
            found_next = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = current_x + dx, current_y + dy
                    h, w = skeleton.shape
                    
                    if (0 <= nx < h and 0 <= ny < w and 
                        skeleton[nx, ny] > 0 and not visited[nx, ny]):
                        current_x, current_y = nx, ny
                        found_next = True
                        break
                
                if found_next:
                    break
            
            if not found_next or length > 100:  # Max length limit
                break
        
        return length
    
    def extract_minutiae(self, image: np.ndarray) -> Dict:
        """
        Complete minutiae extraction pipeline
        
        Args:
            image: Input grayscale fingerprint image
            
        Returns:
            Dictionary containing minutiae data and metadata
        """
        # Preprocessing
        binary_image, quality_map = self.preprocess_for_minutiae(image)
        
        # Ridge thinning
        skeleton = self.thin_ridges(binary_image)
        
        # Minutiae detection
        minutiae_candidates = self.detect_minutiae_candidates(skeleton)
        
        # Quality filtering
        final_minutiae = self.filter_minutiae(minutiae_candidates, quality_map, skeleton)
        
        # Separate by type
        ridge_endings = [m for m in final_minutiae if m['type'] == self.RIDGE_ENDING]
        bifurcations = [m for m in final_minutiae if m['type'] == self.BIFURCATION]
        
        return {
            'minutiae': final_minutiae,
            'ridge_endings': ridge_endings,
            'bifurcations': bifurcations,
            'total_count': len(final_minutiae),
            'quality_map': quality_map,
            'skeleton': skeleton,
            'binary_image': binary_image,
            'statistics': {
                'ridge_endings': len(ridge_endings),
                'bifurcations': len(bifurcations),
                'total_minutiae': len(final_minutiae),
                'mean_quality': np.mean([m['quality'] for m in final_minutiae]) if final_minutiae else 0
            }
        }
    
    def visualize_minutiae(self, image: np.ndarray, minutiae_data: Dict, 
                          save_path: Optional[str] = None) -> np.ndarray:
        """
        Create visualization of detected minutiae
        
        Args:
            image: Original fingerprint image
            minutiae_data: Output from extract_minutiae()
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Minutiae Extraction Results', fontsize=16, fontweight='bold')
        
        # Original image
        ax = axes[0, 0]
        ax.imshow(image, cmap='gray')
        ax.set_title('Original Image')
        ax.axis('off')
        
        # Binary image with skeleton
        ax = axes[0, 1]
        ax.imshow(minutiae_data['binary_image'], cmap='gray')
        skeleton_overlay = minutiae_data['skeleton']
        ax.imshow(skeleton_overlay, cmap='Reds', alpha=0.3)
        ax.set_title(f"Binary + Skeleton")
        ax.axis('off')
        
        # Minutiae overlay on original
        ax = axes[1, 0]
        ax.imshow(image, cmap='gray')
        
        # Plot minutiae points
        for minutia in minutiae_data['minutiae']:
            x, y = minutia['x'], minutia['y']
            if minutia['type'] == self.RIDGE_ENDING:
                ax.plot(y, x, 'ro', markersize=8, label='Ridge Ending' if 'Ridge Ending' not in ax.get_legend_handles_labels()[1] else "")
            elif minutia['type'] == self.BIFURCATION:
                ax.plot(y, x, 'bs', markersize=8, label='Bifurcation' if 'Bifurcation' not in ax.get_legend_handles_labels()[1] else "")
        
        ax.set_title(f"Detected Minutiae (Total: {minutiae_data['total_count']})")
        ax.legend()
        ax.axis('off')
        
        # Quality map
        ax = axes[1, 1]
        im = ax.imshow(minutiae_data['quality_map'], cmap='viridis')
        ax.set_title('Quality Map')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        
        # Add statistics text
        stats = minutiae_data['statistics']
        stats_text = f"""Statistics:
Ridge Endings: {stats['ridge_endings']}
Bifurcations: {stats['bifurcations']}
Total Minutiae: {stats['total_minutiae']}
Mean Quality: {stats['mean_quality']:.3f}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Minutiae visualization saved: {save_path}")
        
        # Convert to image array for return
        fig.canvas.draw()
        try:
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Fallback for different matplotlib backends
            img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img_array = img_array[:, :, :3]  # Remove alpha channel
        
        plt.close(fig)
        return img_array

def test_minutiae_extraction():
    """Test minutiae extraction on sample image"""
    print("🔬 Testing Minutiae Extraction System")
    print("=" * 50)
    
    # Load sample image
    sample_path = r"C:\Users\alifa\Downloads\fingerprint_model-20250811T194218Z-1-001\fingerprint_model\data\archive\fingerprints\DB1_B\101_1.tif"
    
    if not os.path.exists(sample_path):
        print("❌ Sample image not found")
        return
    
    # Load image
    image = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("❌ Failed to load image")
        return
    
    print(f"📷 Testing on: {os.path.basename(sample_path)}")
    print(f"Image size: {image.shape}")
    
    # Initialize extractor
    extractor = MinutiaeExtractor(min_ridge_length=8, quality_threshold=0.1)
    
    # Extract minutiae
    print("🔄 Extracting minutiae...")
    minutiae_data = extractor.extract_minutiae(image)
    
    # Print results
    stats = minutiae_data['statistics']
    print(f"\n📊 Extraction Results:")
    print(f"   Ridge Endings: {stats['ridge_endings']}")
    print(f"   Bifurcations: {stats['bifurcations']}")
    print(f"   Total Minutiae: {stats['total_minutiae']}")
    print(f"   Mean Quality: {stats['mean_quality']:.3f}")
    
    # Create visualization
    print("🎨 Creating visualization...")
    vis_path = "minutiae_extraction_test.png"
    extractor.visualize_minutiae(image, minutiae_data, vis_path)
    
    # Save minutiae data
    json_path = "minutiae_extraction_test.json"
    
    # Convert numpy types to native Python types for JSON serialization
    minutiae_for_json = []
    for m in minutiae_data['minutiae']:
        minutiae_for_json.append({
            'x': int(m['x']),
            'y': int(m['y']),
            'type': int(m['type']),
            'cn': int(m['cn']),
            'angle': float(m['angle']),
            'quality': float(m['quality'])
        })
    
    minutiae_json = {
        'filename': os.path.basename(sample_path),
        'image_shape': [int(x) for x in image.shape],
        'minutiae': minutiae_for_json,
        'statistics': {
            'ridge_endings': int(minutiae_data['statistics']['ridge_endings']),
            'bifurcations': int(minutiae_data['statistics']['bifurcations']),
            'total_minutiae': int(minutiae_data['statistics']['total_minutiae']),
            'mean_quality': float(minutiae_data['statistics']['mean_quality'])
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(minutiae_json, f, indent=2)
    
    print(f"💾 Minutiae data saved: {json_path}")
    print("✅ Minutiae extraction test complete!")

if __name__ == "__main__":
    import os
    test_minutiae_extraction()
