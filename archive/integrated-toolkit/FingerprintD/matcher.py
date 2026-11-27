import cv2
import numpy as np
from PIL import Image
import os
import json
import hashlib
from datetime import datetime
import random

# Simulated database for demonstration
SAMPLE_DATABASE = {
    "person_001": {
        "name": "John Smith",
        "id": "001",
        "fingerprint_hash": "abc123def456",
        "minutiae_features": [
            {"x": 50, "y": 30, "type": "ridge_ending", "angle": 45},
            {"x": 120, "y": 80, "type": "bifurcation", "angle": 90},
            {"x": 200, "y": 150, "type": "ridge_ending", "angle": 135}
        ],
        "pattern_type": "loop",
        "confidence": 0.92
    },
    "person_002": {
        "name": "Jane Doe",
        "id": "002", 
        "fingerprint_hash": "xyz789uvw012",
        "minutiae_features": [
            {"x": 60, "y": 40, "type": "bifurcation", "angle": 60},
            {"x": 140, "y": 100, "type": "ridge_ending", "angle": 120},
            {"x": 220, "y": 160, "type": "bifurcation", "angle": 30}
        ],
        "pattern_type": "whorl",
        "confidence": 0.88
    },
    "person_003": {
        "name": "Bob Johnson",
        "id": "003",
        "fingerprint_hash": "def456ghi789",
        "minutiae_features": [
            {"x": 70, "y": 50, "type": "ridge_ending", "angle": 75},
            {"x": 160, "y": 120, "type": "ridge_ending", "angle": 150},
            {"x": 240, "y": 180, "type": "bifurcation", "angle": 45}
        ],
        "pattern_type": "arch",
        "confidence": 0.85
    }
}

def match_fingerprint(image_file):
    """
    Match fingerprint against database
    
    Args:
        image_file: Uploaded image file from Streamlit
        
    Returns:
        dict: Matching results with confidence scores
    """
    try:
        # Convert Streamlit uploaded file to PIL Image
        image = Image.open(image_file)
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Extract features from uploaded fingerprint
        uploaded_features = extract_matching_features(gray)
        
        # Generate fingerprint hash
        fingerprint_hash = generate_fingerprint_hash(img_array)
        
        # Match against database
        match_results = perform_matching(uploaded_features, fingerprint_hash)
        
        # Add metadata
        match_results.update({
            'uploaded_features': uploaded_features,
            'fingerprint_hash': fingerprint_hash,
            'timestamp': datetime.now().isoformat(),
            'image_size': img_array.shape,
            'processing_steps': ['feature_extraction', 'hash_generation', 'database_matching']
        })
        
        return match_results
        
    except Exception as e:
        # Return fallback results if processing fails
        return {
            'match_found': False,
            'confidence': 0.0,
            'best_match': None,
            'error': str(e),
            'processing_steps': ['error_occurred']
        }

def extract_matching_features(gray_image):
    """Extract features for fingerprint matching"""
    # Apply preprocessing
    processed = preprocess_for_matching(gray_image)
    
    # Extract minutiae points
    minutiae = extract_minutiae_for_matching(processed)
    
    # Calculate additional features
    features = {
        'minutiae_count': len(minutiae),
        'ridge_density': calculate_ridge_density(gray_image),
        'orientation_map': calculate_orientation_map(gray_image),
        'frequency_spectrum': calculate_frequency_spectrum(gray_image),
        'minutiae_distribution': analyze_minutiae_distribution(minutiae)
    }
    
    return features

def preprocess_for_matching(gray_image):
    """Preprocess image for matching"""
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # Apply morphological operations
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def extract_minutiae_for_matching(processed_image):
    """Extract minutiae points for matching"""
    # Apply edge detection
    edges = cv2.Canny(processed_image, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    minutiae = []
    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        if area > 10:  # Filter small contours
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Determine minutiae type based on contour shape
                minutiae_type = classify_minutiae_type(contour)
                
                minutiae.append({
                    'x': cx,
                    'y': cy,
                    'type': minutiae_type,
                    'area': area
                })
    
    return minutiae

def classify_minutiae_type(contour):
    """Classify minutiae type based on contour shape"""
    # Simplified classification based on contour area and shape
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.7:
            return 'ridge_ending'
        else:
            return 'bifurcation'
    
    return 'unknown'

def calculate_ridge_density(gray_image):
    """Calculate ridge density"""
    # Apply edge detection
    edges = cv2.Canny(gray_image, 50, 150)
    
    # Count edge pixels
    ridge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    
    return ridge_pixels / total_pixels

def calculate_orientation_map(gray_image):
    """Calculate ridge orientation map"""
    # Calculate gradients
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate orientation
    orientation = np.arctan2(sobely, sobelx)
    
    # Calculate orientation histogram
    hist, _ = np.histogram(orientation, bins=36, range=(-np.pi, np.pi))
    hist = hist.astype(float) / hist.sum()
    
    return hist.tolist()

def calculate_frequency_spectrum(gray_image):
    """Calculate frequency domain features"""
    # Apply FFT
    fft = np.fft.fft2(gray_image)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)
    
    # Calculate frequency features
    features = {
        'energy': np.sum(magnitude),
        'std': np.std(magnitude),
        'mean': np.mean(magnitude)
    }
    
    return features

def analyze_minutiae_distribution(minutiae):
    """Analyze distribution of minutiae points"""
    if not minutiae:
        return {'density': 0, 'spread': 0}
    
    # Calculate spatial distribution
    x_coords = [m['x'] for m in minutiae]
    y_coords = [m['y'] for m in minutiae]
    
    # Calculate density and spread
    density = len(minutiae) / (max(x_coords) - min(x_coords) + 1) / (max(y_coords) - min(y_coords) + 1)
    spread = np.std(x_coords) + np.std(y_coords)
    
    return {
        'density': density,
        'spread': spread,
        'center_x': np.mean(x_coords),
        'center_y': np.mean(y_coords)
    }

def generate_fingerprint_hash(image_array):
    """Generate a hash for the fingerprint image"""
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    # Resize to standard size for consistent hashing
    resized = cv2.resize(gray, (64, 64))
    
    # Generate hash
    hash_object = hashlib.md5(resized.tobytes())
    return hash_object.hexdigest()

def perform_matching(uploaded_features, fingerprint_hash):
    """Perform matching against database"""
    best_match = None
    best_score = 0.0
    
    # Check for exact hash match first
    for person_id, person_data in SAMPLE_DATABASE.items():
        if person_data['fingerprint_hash'] == fingerprint_hash:
            return {
                'match_found': True,
                'confidence': 1.0,
                'best_match': person_data,
                'match_type': 'exact_hash_match'
            }
    
    # Perform feature-based matching
    for person_id, person_data in SAMPLE_DATABASE.items():
        # Calculate similarity score
        score = calculate_similarity_score(uploaded_features, person_data)
        
        if score > best_score:
            best_score = score
            best_match = person_data
    
    # Determine if match is found based on threshold
    match_threshold = 0.7
    match_found = best_score >= match_threshold
    
    return {
        'match_found': match_found,
        'confidence': best_score,
        'best_match': best_match if match_found else None,
        'match_type': 'feature_based_match',
        'all_scores': {person_id: calculate_similarity_score(uploaded_features, person_data) 
                      for person_id, person_data in SAMPLE_DATABASE.items()}
    }

def calculate_similarity_score(uploaded_features, database_person):
    """Calculate similarity score between uploaded and database fingerprint"""
    # Extract features for comparison
    uploaded_minutiae_count = uploaded_features['minutiae_count']
    uploaded_ridge_density = uploaded_features['ridge_density']
    
    # Get database person features (simplified)
    db_minutiae_count = len(database_person['minutiae_features'])
    
    # Calculate similarity scores for different features
    minutiae_similarity = 1.0 - abs(uploaded_minutiae_count - db_minutiae_count) / max(uploaded_minutiae_count, db_minutiae_count, 1)
    
    # Pattern type similarity (if available)
    pattern_similarity = 0.5  # Default neutral score
    
    # Ridge density similarity (simplified)
    ridge_similarity = 0.7 + random.uniform(0, 0.3)  # Simulated similarity
    
    # Combine scores with weights
    final_score = (0.4 * minutiae_similarity + 
                   0.3 * pattern_similarity + 
                   0.3 * ridge_similarity)
    
    return min(1.0, max(0.0, final_score))

# Database management functions
def add_to_database(person_info, fingerprint_image):
    """Add a new fingerprint to the database"""
    # This would be implemented to actually save to a real database
    pass

def remove_from_database(person_id):
    """Remove a fingerprint from the database"""
    # This would be implemented to actually remove from a real database
    pass

def get_database_stats():
    """Get database statistics"""
    return {
        'total_records': len(SAMPLE_DATABASE),
        'last_updated': datetime.now().isoformat(),
        'database_size': 'demo_mode'
    }
