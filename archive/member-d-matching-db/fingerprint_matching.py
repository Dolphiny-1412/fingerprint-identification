import os
import cv2
import numpy as np
import json
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import sys

# Add the minutiae folder to the path to import the advanced detector
sys.path.append(os.path.join(os.path.dirname(__file__), 'minutiae'))
from detector import AdvancedFingerprintAnalyzer


def extract_minutiae_features_advanced(image_path):
    """
    Extract minutiae features using the advanced detector.
    
    Args:
        image_path: Path to the fingerprint image
        
    Returns:
        dict: Dictionary containing advanced minutiae features
    """
    try:
        # Initialize the advanced analyzer
        analyzer = AdvancedFingerprintAnalyzer()
        
        # Analyze the fingerprint
        result = analyzer.analyze_fingerprint(image_path)
        
        if 'error' in result:
            print(f"❌ Advanced analysis failed: {result['error']}")
            return None
        
        # Extract minutiae with confidence scores
        minutiae_points = []
        for minutiae in result.get('minutiae', []):
            minutiae_points.append({
                'x': minutiae['x'],
                'y': minutiae['y'],
                'type': minutiae['type'],
                'angle': minutiae.get('angle', 0.0),
                'confidence': minutiae.get('confidence', 0.5)
            })
        
        return {
            'minutiae_count': len(minutiae_points),
            'endings': len([m for m in minutiae_points if m['type'] == 'ending']),
            'bifurcations': len([m for m in minutiae_points if m['type'] == 'bifurcation']),
            'minutiae_points': minutiae_points,
            'quality_score': result.get('quality_score', 0.0),
            'spoof_risk_score': result.get('spoof_risk_score', 0.0),
            'spoof_suspected': result.get('spoof_suspected', False)
        }
    except Exception as e:
        print(f"❌ Error in advanced minutiae extraction: {e}")
        return None


def extract_ridge_features(image):
    """
    Extract ridge pattern features from fingerprint image.
    
    Args:
        image: Grayscale fingerprint image
        
    Returns:
        dict: Dictionary containing ridge features
    """
    # Apply Sobel filter to detect edges
    sobel_x = sobel(image, axis=0)
    sobel_y = sobel(image, axis=1)
    
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Compute LBP features
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    
    # Compute histogram of LBP
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()  # Normalize
    
    return {
        'gradient_mean': float(np.mean(gradient_magnitude)),
        'gradient_std': float(np.std(gradient_magnitude)),
        'lbp_histogram': lbp_hist.tolist()
    }


def extract_fingerprint_features(image_path):
    """
    Extract comprehensive features from a fingerprint image using advanced analysis.
    
    Args:
        image_path: Path to the fingerprint image
        
    Returns:
        dict: Dictionary containing all extracted features
    """
    # Read image for basic processing
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    # Resize image to standard size (maintain aspect ratio)
    target_size = (256, 256)
    h, w = image.shape
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h))
    
    # Pad to target size
    padded_image = np.zeros(target_size, dtype=np.uint8)
    y_offset = (target_size[1] - new_h) // 2
    x_offset = (target_size[0] - new_w) // 2
    padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image
    
    # Extract advanced minutiae features
    minutiae_features = extract_minutiae_features_advanced(image_path)
    if minutiae_features is None:
        print(f"⚠️  Falling back to basic minutiae extraction for {image_path}")
        minutiae_features = extract_minutiae_features_basic(padded_image)
    
    # Extract ridge features
    ridge_features = extract_ridge_features(padded_image)
    
    # Combine all features
    features = {
        'image_path': image_path,
        'image_size': padded_image.shape,
        'minutiae': minutiae_features,
        'ridge': ridge_features,
        'processed_timestamp': str(np.datetime64('now'))
    }
    
    return features


def extract_minutiae_features_basic(image):
    """
    Basic minutiae extraction (fallback method).
    
    Args:
        image: Grayscale fingerprint image
        
    Returns:
        dict: Dictionary containing basic minutiae features
    """
    # Convert to binary image
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Skeletonize the image to get ridge structure
    skeleton = skeletonize(binary == 0)
    
    # Find ridge endings and bifurcations
    minutiae_points = []
    for i in range(1, skeleton.shape[0] - 1):
        for j in range(1, skeleton.shape[1] - 1):
            if skeleton[i, j]:
                # Count 8-neighbors
                neighbors = skeleton[i-1:i+2, j-1:j+2].sum() - 1
                
                if neighbors == 1:  # Ridge ending
                    minutiae_points.append({
                        'type': 'ending',
                        'x': j,
                        'y': i,
                        'confidence': 0.5  # Default confidence for basic method
                    })
                elif neighbors == 3:  # Bifurcation
                    minutiae_points.append({
                        'type': 'bifurcation',
                        'x': j,
                        'y': i,
                        'confidence': 0.5  # Default confidence for basic method
                    })
    
    return {
        'minutiae_count': len(minutiae_points),
        'endings': len([m for m in minutiae_points if m['type'] == 'ending']),
        'bifurcations': len([m for m in minutiae_points if m['type'] == 'bifurcation']),
        'minutiae_points': minutiae_points,
        'quality_score': 0.5,  # Default quality score
        'spoof_risk_score': 0.0,  # No spoof detection in basic method
        'spoof_suspected': False
    }


def detect_all_images_in_folder(folder_path, recursive=True):
    """
    Detect all image files in a folder by searching each file individually.
    
    Args:
        folder_path (str): Path to the folder to scan
        recursive (bool): Whether to search subdirectories recursively
        
    Returns:
        list: List of full paths to all detected image files
    """
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp")
    detected_images = []
    
    print(f"🔍 Scanning folder: {folder_path}")
    print(f"📁 Recursive search: {'Yes' if recursive else 'No'}")
    print("-" * 50)
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return detected_images
    
    if not os.path.isdir(folder_path):
        print(f"❌ Path is not a directory: {folder_path}")
        return detected_images
    
    # Get all files in the folder
    if recursive:
        # Walk through all subdirectories
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(image_extensions):
                    image_path = os.path.join(root, filename)
                    detected_images.append(image_path)
                    print(f"📸 Found: {os.path.relpath(image_path, folder_path)}")
    else:
        # Only check files in the immediate directory
        try:
            files = os.listdir(folder_path)
            for filename in files:
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
                    detected_images.append(file_path)
                    print(f"📸 Found: {filename}")
        except PermissionError:
            print(f"❌ Permission denied accessing folder: {folder_path}")
            return detected_images
    
    print(f"\n📊 Detection Summary:")
    print(f"   • Total images found: {len(detected_images)}")
    print(f"   • Folder: {folder_path}")
    
    return detected_images


def process_fingerprint_images(dataset_folder, log_filename, database_filename):
    """
    Recursively processes fingerprint images in the given dataset folder,
    extracts comprehensive features, and builds a fingerprint database.
    
    Args:
        dataset_folder (str): Path to the dataset folder containing fingerprint images
        log_filename (str): Path to the log file where processed image paths will be saved
        database_filename (str): Path to the JSON database file
    """
    processed_files = []
    fingerprint_database = {}
    
    # Walk through all subfolders and files
    for root, _, files in os.walk(dataset_folder):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                image_path = os.path.join(root, filename)
                
                print(f"🔍 Processing: {os.path.relpath(image_path, dataset_folder)}")
                
                # Extract features
                features = extract_fingerprint_features(image_path)
                if features is None:
                    print(f"❌ Could not process: {image_path}")
                    continue
                
                # Store the relative path for logging
                rel_path = os.path.relpath(image_path, dataset_folder)
                processed_files.append(rel_path)
                
                # Add to database with a unique ID
                fingerprint_id = f"{os.path.basename(root)}_{filename.split('.')[0]}"
                fingerprint_database[fingerprint_id] = features
                
                print(f"✅ Processed: {rel_path} (ID: {fingerprint_id})")

    # Write all processed file names to the log
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("Processed fingerprint images:\n")
        f.write("=" * 50 + "\n")
        for item in processed_files:
            f.write(item + "\n")

    # Save the fingerprint database
    with open(database_filename, "w", encoding="utf-8") as f:
        json.dump(fingerprint_database, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Summary:")
    print(f"   • Processed {len(processed_files)} files")
    print(f"   • Database entries: {len(fingerprint_database)}")
    print(f"   • Log saved to: {log_filename}")
    print(f"   • Database saved to: {database_filename}")
    
    return processed_files, fingerprint_database


def process_images_from_detected_list(image_paths, log_filename, database_filename, base_folder=None):
    """
    Process a list of detected image files and build a fingerprint database.
    
    Args:
        image_paths (list): List of full paths to image files
        log_filename (str): Path to the log file where processed image paths will be saved
        database_filename (str): Path to the JSON database file
        base_folder (str): Optional base folder for relative path calculation
        
    Returns:
        tuple: (processed_files, fingerprint_database)
    """
    processed_files = []
    fingerprint_database = {}
    
    print(f"🔄 Processing {len(image_paths)} detected images...")
    print("=" * 60)
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n🔍 Processing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Extract features
        features = extract_fingerprint_features(image_path)
        if features is None:
            print(f"❌ Could not process: {image_path}")
            continue
        
        # Create relative path for logging
        if base_folder:
            rel_path = os.path.relpath(image_path, base_folder)
        else:
            rel_path = image_path
        
        processed_files.append(rel_path)
        
        # Create unique ID from file path
        path_parts = os.path.normpath(image_path).split(os.sep)
        if len(path_parts) >= 2:
            # Use parent folder and filename
            fingerprint_id = f"{path_parts[-2]}_{os.path.splitext(path_parts[-1])[0]}"
        else:
            # Use just filename
            fingerprint_id = os.path.splitext(os.path.basename(image_path))[0]
        
        fingerprint_database[fingerprint_id] = features
        
        print(f"✅ Processed: {rel_path} (ID: {fingerprint_id})")
        print(f"   📊 Minutiae: {features['minutiae']['minutiae_count']}")
        print(f"   🎯 Quality: {features['minutiae'].get('quality_score', 0.0):.3f}")
        print(f"   🛡️  Spoof Risk: {features['minutiae'].get('spoof_risk_score', 0.0):.3f}")

    # Write all processed file names to the log
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("Processed fingerprint images:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total images processed: {len(processed_files)}\n")
        f.write(f"Processing timestamp: {str(np.datetime64('now'))}\n")
        f.write("-" * 50 + "\n")
        for item in processed_files:
            f.write(item + "\n")

    # Save the fingerprint database
    with open(database_filename, "w", encoding="utf-8") as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Convert the database to ensure JSON serialization
        serializable_database = convert_numpy_types(fingerprint_database)
        json.dump(serializable_database, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Processing Summary:")
    print(f"   • Total images processed: {len(processed_files)}")
    print(f"   • Database entries: {len(fingerprint_database)}")
    print(f"   • Log saved to: {log_filename}")
    print(f"   • Database saved to: {database_filename}")
    
    return processed_files, fingerprint_database


def calculate_minutiae_similarity(minutiae1, minutiae2, max_distance=20):
    """
    Calculate similarity between two sets of minutiae points with confidence weighting.
    
    Args:
        minutiae1: List of minutiae points from first fingerprint
        minutiae2: List of minutiae points from second fingerprint
        max_distance: Maximum distance for considering minutiae as matching
        
    Returns:
        dict: Similarity metrics
    """
    if not minutiae1 or not minutiae2:
        return {
            'similarity_score': 0.0,
            'matched_pairs': 0,
            'total_minutiae': len(minutiae1) + len(minutiae2),
            'match_ratio': 0.0
        }
    
    # Create distance matrix between all minutiae points
    distances = np.zeros((len(minutiae1), len(minutiae2)))
    type_matches = np.zeros((len(minutiae1), len(minutiae2)))
    confidence_weights = np.zeros((len(minutiae1), len(minutiae2)))
    
    for i, m1 in enumerate(minutiae1):
        for j, m2 in enumerate(minutiae2):
            # Calculate Euclidean distance
            dist = euclidean([m1['x'], m1['y']], [m2['x'], m2['y']])
            distances[i, j] = dist
            
            # Check if types match (ending vs ending, bifurcation vs bifurcation)
            if m1['type'] == m2['type']:
                type_matches[i, j] = 1
            
            # Calculate confidence weight (geometric mean of both confidences)
            conf1 = m1.get('confidence', 0.5)
            conf2 = m2.get('confidence', 0.5)
            confidence_weights[i, j] = np.sqrt(conf1 * conf2)
    
    # Use Hungarian algorithm to find optimal matching
    row_indices, col_indices = linear_sum_assignment(distances)
    
    # Count valid matches with confidence weighting
    matched_pairs = 0
    total_distance = 0
    total_confidence = 0
    
    for i, j in zip(row_indices, col_indices):
        if (distances[i, j] <= max_distance and 
            type_matches[i, j] == 1):
            matched_pairs += 1
            total_distance += distances[i, j]
            total_confidence += confidence_weights[i, j]
    
    # Calculate similarity metrics
    total_minutiae = len(minutiae1) + len(minutiae2)
    match_ratio = (2 * matched_pairs) / total_minutiae if total_minutiae > 0 else 0
    
    # Similarity score based on match ratio, average distance, and confidence
    avg_distance = total_distance / matched_pairs if matched_pairs > 0 else float('inf')
    avg_confidence = total_confidence / matched_pairs if matched_pairs > 0 else 0
    
    distance_score = max(0, 1 - (avg_distance / max_distance)) if avg_distance != float('inf') else 0
    
    # Weighted similarity score
    similarity_score = (match_ratio * 0.5) + (distance_score * 0.3) + (avg_confidence * 0.2)
    
    return {
        'similarity_score': similarity_score,
        'matched_pairs': matched_pairs,
        'total_minutiae': total_minutiae,
        'match_ratio': match_ratio,
        'average_distance': avg_distance if avg_distance != float('inf') else 0,
        'average_confidence': avg_confidence
    }


def calculate_ridge_similarity(ridge1, ridge2):
    """
    Calculate similarity between ridge pattern features.
    
    Args:
        ridge1: Ridge features from first fingerprint
        ridge2: Ridge features from second fingerprint
        
    Returns:
        float: Ridge similarity score
    """
    # Compare gradient features
    gradient_diff = abs(ridge1['gradient_mean'] - ridge2['gradient_mean']) / max(ridge1['gradient_mean'], ridge2['gradient_mean'])
    gradient_similarity = max(0, 1 - gradient_diff)
    
    # Compare LBP histograms using correlation
    lbp1 = np.array(ridge1['lbp_histogram'])
    lbp2 = np.array(ridge2['lbp_histogram'])
    
    # Normalize histograms
    lbp1_norm = lbp1 / (np.sum(lbp1) + 1e-8)
    lbp2_norm = lbp2 / (np.sum(lbp2) + 1e-8)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(lbp1_norm, lbp2_norm)[0, 1]
    lbp_similarity = max(0, correlation) if not np.isnan(correlation) else 0
    
    # Weighted combination
    ridge_similarity = (gradient_similarity * 0.4) + (lbp_similarity * 0.6)
    
    return ridge_similarity


def match_fingerprints(features1, features2, weights=None):
    """
    Match two fingerprints using comprehensive feature comparison with spoof detection.
    
    Args:
        features1: Features from first fingerprint
        features2: Features from second fingerprint
        weights: Dictionary of weights for different similarity measures
        
    Returns:
        dict: Comprehensive matching results
    """
    if weights is None:
        weights = {
            'minutiae': 0.6,
            'ridge': 0.3,
            'overall': 0.1
        }
    
    # Check for spoof detection
    spoof_risk1 = features1['minutiae'].get('spoof_risk_score', 0.0)
    spoof_risk2 = features2['minutiae'].get('spoof_risk_score', 0.0)
    spoof_suspected1 = features1['minutiae'].get('spoof_suspected', False)
    spoof_suspected2 = features2['minutiae'].get('spoof_suspected', False)
    
    # Calculate combined spoof risk
    combined_spoof_risk = max(spoof_risk1, spoof_risk2)
    spoof_detected = spoof_suspected1 or spoof_suspected2
    
    # Minutiae similarity
    minutiae_sim = calculate_minutiae_similarity(
        features1['minutiae']['minutiae_points'],
        features2['minutiae']['minutiae_points']
    )
    
    # Ridge pattern similarity
    ridge_sim = calculate_ridge_similarity(
        features1['ridge'],
        features2['ridge']
    )
    
    # Overall similarity (based on minutiae counts, etc.)
    minutiae_count_diff = abs(features1['minutiae']['minutiae_count'] - features2['minutiae']['minutiae_count'])
    max_count = max(features1['minutiae']['minutiae_count'], features2['minutiae']['minutiae_count'])
    overall_sim = max(0, 1 - (minutiae_count_diff / max_count)) if max_count > 0 else 0
    
    # Calculate weighted final score
    final_score = (
        minutiae_sim['similarity_score'] * weights['minutiae'] +
        ridge_sim * weights['ridge'] +
        overall_sim * weights['overall']
    )
    
    # Apply spoof penalty
    if spoof_detected:
        final_score *= (1 - combined_spoof_risk * 0.5)  # Reduce score based on spoof risk
    
    # Determine confidence level
    if spoof_detected:
        confidence = 'spoof_detected'
    elif final_score >= 0.8:
        confidence = 'high'
    elif final_score >= 0.6:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return {
        'final_score': final_score,
        'minutiae_similarity': minutiae_sim,
        'ridge_similarity': ridge_sim,
        'overall_similarity': overall_sim,
        'spoof_risk': combined_spoof_risk,
        'spoof_detected': spoof_detected,
        'is_match': final_score >= 0.6 and not spoof_detected,  # No match if spoof detected
        'confidence': confidence
    }


def find_matches_in_database(query_features, database, threshold=0.6, top_k=5):
    """
    Find matches for a query fingerprint in the database with spoof detection.
    
    Args:
        query_features: Features of the query fingerprint
        database: Dictionary containing fingerprint database
        threshold: Minimum similarity threshold for matches
        top_k: Number of top matches to return
        
    Returns:
        list: List of matches sorted by similarity score
    """
    matches = []
    
    for fingerprint_id, db_features in database.items():
        match_result = match_fingerprints(query_features, db_features)
        
        if match_result['final_score'] >= threshold and not match_result['spoof_detected']:
            matches.append({
                'fingerprint_id': fingerprint_id,
                'similarity_score': match_result['final_score'],
                'confidence': match_result['confidence'],
                'minutiae_matched': match_result['minutiae_similarity']['matched_pairs'],
                'spoof_risk': match_result['spoof_risk'],
                'image_path': db_features['image_path']
            })
    
    # Sort by similarity score (descending)
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return matches[:top_k]


def verify_fingerprint(query_image_path, database_path, threshold=0.6):
    """
    Verify a fingerprint against the database.
    
    Args:
        query_image_path: Path to the query fingerprint image
        database_path: Path to the fingerprint database JSON file
        threshold: Minimum similarity threshold for verification
        
    Returns:
        dict: Verification results
    """
    # Load database
    with open(database_path, 'r') as f:
        database = json.load(f)
    
    # Extract features from query image
    query_features = extract_fingerprint_features(query_image_path)
    if query_features is None:
        return {
            'success': False,
            'error': 'Could not extract features from query image'
        }
    
    # Find matches
    matches = find_matches_in_database(query_features, database, threshold)
    
    # Determine verification result
    is_verified = len(matches) > 0
    best_match = matches[0] if matches else None
    
    return {
        'success': True,
        'is_verified': is_verified,
        'best_match': best_match,
        'all_matches': matches,
        'query_features': {
            'minutiae_count': query_features['minutiae']['minutiae_count'],
            'image_size': query_features['image_size']
        }
    }


def test_matching_system(database_path, num_tests=5):
    """
    Test the fingerprint matching system with sample fingerprints.
    
    Args:
        database_path: Path to the fingerprint database JSON file
        num_tests: Number of test cases to run
        
    Returns:
        dict: Test results
    """
    # Load database
    with open(database_path, 'r') as f:
        database = json.load(f)
    
    print("🧪 Testing Advanced Fingerprint Matching System")
    print("=" * 60)
    
    test_results = []
    database_ids = list(database.keys())
    
    for i in range(min(num_tests, len(database_ids))):
        # Use a fingerprint as query
        query_id = database_ids[i]
        query_features = database[query_id]
        
        print(f"\n🔍 Test {i+1}: Querying with {query_id}")
        print(f"   Minutiae count: {query_features['minutiae']['minutiae_count']}")
        
        # Show quality and spoof information
        quality_score = query_features['minutiae'].get('quality_score', 0.0)
        spoof_risk = query_features['minutiae'].get('spoof_risk_score', 0.0)
        spoof_suspected = query_features['minutiae'].get('spoof_suspected', False)
        
        print(f"   Quality score: {quality_score:.3f}")
        print(f"   Spoof risk: {spoof_risk:.3f}")
        print(f"   Spoof suspected: {'⚠️ YES' if spoof_suspected else '✅ NO'}")
        
        # Find matches
        matches = find_matches_in_database(query_features, database, threshold=0.5, top_k=3)
        
        # Check if the query fingerprint itself is found (should be first match)
        self_match_found = len(matches) > 0 and matches[0]['fingerprint_id'] == query_id
        
        test_result = {
            'query_id': query_id,
            'self_match_found': self_match_found,
            'self_match_score': matches[0]['similarity_score'] if self_match_found else 0,
            'total_matches': len(matches),
            'matches': matches,
            'quality_score': quality_score,
            'spoof_risk': spoof_risk,
            'spoof_suspected': spoof_suspected
        }
        
        test_results.append(test_result)
        
        print(f"   ✅ Self-match found: {self_match_found}")
        print(f"   📊 Self-match score: {test_result['self_match_score']:.3f}")
        print(f"   🎯 Total matches: {len(matches)}")
        
        if len(matches) > 1:
            print(f"   🔗 Other matches:")
            for j, match in enumerate(matches[1:3]):  # Show top 2 other matches
                confidence_emoji = "🟢" if match['confidence'] == 'high' else "🟡" if match['confidence'] == 'medium' else "🔴"
                print(f"      {j+1}. {match['fingerprint_id']} (Score: {match['similarity_score']:.3f}, Confidence: {confidence_emoji} {match['confidence']})")
    
    # Calculate overall statistics
    successful_self_matches = sum(1 for result in test_results if result['self_match_found'])
    avg_self_match_score = np.mean([r['self_match_score'] for r in test_results if r['self_match_found']])
    avg_quality_score = np.mean([r['quality_score'] for r in test_results])
    avg_spoof_risk = np.mean([r['spoof_risk'] for r in test_results])
    spoof_detections = sum(1 for r in test_results if r['spoof_suspected'])
    
    print(f"\n📈 Overall Test Results:")
    print(f"   • Successful self-matches: {successful_self_matches}/{len(test_results)}")
    print(f"   • Average self-match score: {avg_self_match_score:.3f}")
    print(f"   • System accuracy: {(successful_self_matches/len(test_results)*100):.1f}%")
    print(f"   • Average quality score: {avg_quality_score:.3f}")
    print(f"   • Average spoof risk: {avg_spoof_risk:.3f}")
    print(f"   • Spoof detections: {spoof_detections}/{len(test_results)}")
    
    return {
        'test_results': test_results,
        'successful_self_matches': successful_self_matches,
        'total_tests': len(test_results),
        'accuracy': successful_self_matches / len(test_results),
        'avg_self_match_score': avg_self_match_score,
        'avg_quality_score': avg_quality_score,
        'avg_spoof_risk': avg_spoof_risk,
        'spoof_detections': spoof_detections
    }


def generate_security_enhanced_matching_data(features):
    """
    Generate security-enhanced matching data in the exact format required for Person D integration.
    
    Args:
        features: Extracted fingerprint features
        
    Returns:
        dict: Security-enhanced matching data in the specified format
    """
    minutiae_data = features['minutiae']
    
    # Format minutiae points for integration
    formatted_minutiae = []
    for minutiae in minutiae_data.get('minutiae_points', []):
        formatted_minutiae.append({
            'x': minutiae['x'],
            'y': minutiae['y'],
            'type': minutiae['type'],
            'confidence': minutiae.get('confidence', 0.5),
            'angle': minutiae.get('angle', 0.0)
        })
    
    # Return the exact format specified for Person D
    return {
        "minutiae_count": minutiae_data.get('minutiae_count', 0),
        "spoof_risk_score": minutiae_data.get('spoof_risk_score', 0.0),
        "quality_score": minutiae_data.get('quality_score', 0.0),
        "minutiae": formatted_minutiae
    }


def match_fingerprints_with_integration_format(features1, features2, weights=None):
    """
    Match two fingerprints and return results in integration-ready format.
    
    Args:
        features1: Features from first fingerprint
        features2: Features from second fingerprint
        weights: Dictionary of weights for different similarity measures
        
    Returns:
        dict: Matching results with integration-ready data structures
    """
    # Perform standard matching
    match_result = match_fingerprints(features1, features2, weights)
    
    # Generate security-enhanced data for both fingerprints
    security_data_1 = generate_security_enhanced_matching_data(features1)
    security_data_2 = generate_security_enhanced_matching_data(features2)
    
    # Return enhanced result with integration data
    return {
        'matching_result': match_result,
        'query_fingerprint_data': security_data_1,
        'database_fingerprint_data': security_data_2,
        'integration_ready': True
    }


def verify_fingerprint_with_integration_data(query_image_path, database_path, threshold=0.6):
    """
    Verify a fingerprint with integration-ready data output.
    
    Args:
        query_image_path: Path to the query fingerprint image
        database_path: Path to the fingerprint database JSON file
        threshold: Minimum similarity threshold for verification
        
    Returns:
        dict: Verification results with integration data
    """
    # Load database
    with open(database_path, 'r') as f:
        database = json.load(f)
    
    # Extract features from query image
    query_features = extract_fingerprint_features(query_image_path)
    if query_features is None:
        return {
            'success': False,
            'error': 'Could not extract features from query image',
            'integration_data': None
        }
    
    # Generate security-enhanced data for query
    query_integration_data = generate_security_enhanced_matching_data(query_features)
    
    # Find matches
    matches = find_matches_in_database(query_features, database, threshold)
    
    # Generate integration data for best match if available
    best_match_integration_data = None
    if matches:
        best_match_id = matches[0]['fingerprint_id']
        best_match_features = database[best_match_id]
        best_match_integration_data = generate_security_enhanced_matching_data(best_match_features)
    
    # Determine verification result
    is_verified = len(matches) > 0
    best_match = matches[0] if matches else None
    
    return {
        'success': True,
        'is_verified': is_verified,
        'best_match': best_match,
        'all_matches': matches,
        'query_integration_data': query_integration_data,
        'best_match_integration_data': best_match_integration_data,
        'query_features': {
            'minutiae_count': query_features['minutiae']['minutiae_count'],
            'image_size': query_features['image_size']
        }
    }


def process_folder_with_task_d_output(folder_path, recursive=True, output_format="task_d"):
    """
    Process all images in a folder and output results in Task D (Matcher) format.
    
    Args:
        folder_path: Path to the folder containing images
        recursive: Whether to search subdirectories recursively
        output_format: Output format ("task_d" for Task D format, "detailed" for full details)
        
    Returns:
        dict: Results in the specified format
    """
    print(f"🔍 Processing folder: {folder_path}")
    print(f"📁 Recursive search: {'Yes' if recursive else 'No'}")
    print("=" * 60)
    
    # Detect all images in the folder
    detected_images = detect_all_images_in_folder(folder_path, recursive=recursive)
    
    if not detected_images:
        return {
            'success': False,
            'error': 'No images found in the specified folder',
            'results': []
        }
    
    print(f"\n🔄 Processing {len(detected_images)} detected images...")
    print("=" * 60)
    
    results = []
    
    for i, image_path in enumerate(detected_images, 1):
        print(f"\n🔍 Processing {i}/{len(detected_images)}: {os.path.basename(image_path)}")
        
        # Extract features from the image
        features = extract_fingerprint_features(image_path)
        if features is None:
            print(f"❌ Could not process: {image_path}")
            continue
        
        # Generate security-enhanced matching data in Task D format
        security_enhanced_data = generate_security_enhanced_matching_data(features)
        
        # Create result entry
        result_entry = {
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'processing_success': True,
            'security_enhanced_matching_data': security_enhanced_data
        }
        
        results.append(result_entry)
        
        # Display the Task D format output
        print(f"✅ Processed: {os.path.basename(image_path)}")
        print(f"📊 Task D (Matcher) Output:")
        print(f"   minutiae_count: {security_enhanced_data['minutiae_count']}")
        print(f"   spoof_risk_score: {security_enhanced_data['spoof_risk_score']:.3f}")
        print(f"   quality_score: {security_enhanced_data['quality_score']:.3f}")
        print(f"   minutiae: [...] # {len(security_enhanced_data['minutiae'])} minutiae points")
    
    print(f"\n📊 Processing Summary:")
    print(f"   • Total images processed: {len(results)}")
    print(f"   • Success rate: {len(results)/len(detected_images)*100:.1f}%")
    
    return {
        'success': True,
        'total_images_found': len(detected_images),
        'total_images_processed': len(results),
        'folder_path': folder_path,
        'recursive_search': recursive,
        'results': results
    }


def main():
    """Main function to run the fingerprint processing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fingerprint Processing and Matching System')
    parser.add_argument('--folder', '-f', help='Custom folder path to scan for images')
    parser.add_argument('--recursive', '-r', action='store_true', default=True, 
                       help='Search subdirectories recursively (default: True)')
    parser.add_argument('--no-recursive', action='store_true', 
                       help='Disable recursive search (only immediate directory)')
    parser.add_argument('--database', '-d', help='Custom database filename')
    parser.add_argument('--log', '-l', help='Custom log filename')
    parser.add_argument('--test-only', action='store_true', 
                       help='Only test existing database without processing new images')
    parser.add_argument('--detect-only', action='store_true', 
                       help='Only detect images without processing them')
    parser.add_argument('--task-d-output', action='store_true', 
                       help='Output results in Task D (Matcher) format')
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine folder to scan
    if args.folder:
        dataset_dir = args.folder
        print(f"🎯 Using custom folder: {dataset_dir}")
    else:
        dataset_dir = os.path.join(script_dir, "archive", "fingerprints")
        print(f"📁 Using default folder: {dataset_dir}")
    
    # Determine recursive search
    recursive = args.recursive and not args.no_recursive
    
    # Output files
    log_file = args.log if args.log else os.path.join(script_dir, "processed_images.txt")
    database_file = args.database if args.database else os.path.join(script_dir, "fingerprint_database.json")
    
    print(f"🔍 Processing fingerprint images from: {dataset_dir}")
    print(f"📁 Recursive search: {'Yes' if recursive else 'No'}")
    print(f"📄 Log file will be saved to: {log_file}")
    print(f"🗄️  Database will be saved to: {database_file}")
    print("=" * 60)
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Dataset directory not found: {dataset_dir}")
        return
    
    # If test-only mode, just test existing database
    if args.test_only:
        if os.path.exists(database_file):
            print("📋 Testing existing database...")
            test_results = test_matching_system(database_file)
        else:
            print(f"❌ Database not found: {database_file}")
        return
    
    # Detect images in the folder
    print("\n🔍 Step 1: Detecting all images in folder...")
    detected_images = detect_all_images_in_folder(dataset_dir, recursive=recursive)
    
    if not detected_images:
        print("❌ No images found in the specified folder!")
        return
    
    # If detect-only mode, just show what was found
    if args.detect_only:
        print(f"\n📊 Detection Complete!")
        print(f"   • Found {len(detected_images)} images")
        print(f"   • Folder: {dataset_dir}")
        print(f"   • Recursive: {'Yes' if recursive else 'No'}")
        return
    
    # If Task D output mode, process images and show Task D format
    if args.task_d_output:
        print(f"\n🎯 Task D (Matcher) Output Mode")
        print("=" * 60)
        result = process_folder_with_task_d_output(dataset_dir, recursive=recursive)
        
        if result['success']:
            print(f"\n🎉 Task D Processing Complete!")
            print(f"📊 Summary:")
            print(f"   • Images found: {result['total_images_found']}")
            print(f"   • Images processed: {result['total_images_processed']}")
            print(f"   • Folder: {result['folder_path']}")
            
            # Show sample Task D output format
            if result['results']:
                print(f"\n📋 Sample Task D (Matcher) Output Format:")
                print("-" * 50)
                sample_result = result['results'][0]
                sample_data = sample_result['security_enhanced_matching_data']
                print(f"# Security-enhanced matching data:")
                print(f"{{")
                print(f'  "minutiae_count": {sample_data["minutiae_count"]},')
                print(f'  "spoof_risk_score": {sample_data["spoof_risk_score"]:.3f},')
                print(f'  "quality_score": {sample_data["quality_score"]:.3f},')
                print(f'  "minutiae": [...] # {len(sample_data["minutiae"])} minutiae points')
                print(f"}}")
        return
    
    # Check if database already exists
    if os.path.exists(database_file):
        print(f"\n📋 Database already exists! Testing matching system...")
        test_results = test_matching_system(database_file)
        
        # Demonstrate verification with a sample fingerprint
        print(f"\n🔐 Demonstration: Fingerprint Verification")
        print("-" * 40)
        
        # Use first fingerprint as example
        with open(database_file, 'r') as f:
            database = json.load(f)
        
        if database:
            sample_id = list(database.keys())[0]
            sample_path = database[sample_id]['image_path']
            
            print(f"Testing verification with: {sample_id}")
            verification_result = verify_fingerprint(sample_path, database_file)
            
            if verification_result['success']:
                print(f"✅ Verification successful!")
                print(f"📊 Is verified: {verification_result['is_verified']}")
                if verification_result['best_match']:
                    print(f"🎯 Best match: {verification_result['best_match']['fingerprint_id']}")
                    print(f"📈 Similarity score: {verification_result['best_match']['similarity_score']:.3f}")
                    print(f"🛡️  Confidence: {verification_result['best_match']['confidence']}")
            else:
                print(f"❌ Verification failed: {verification_result['error']}")
    else:
        # Process the detected images and build database
        print(f"\n🔄 Step 2: Processing detected images...")
        processed_files, database = process_images_from_detected_list(
            detected_images, log_file, database_file, base_folder=dataset_dir
        )
        
        print("=" * 60)
        print("🎉 Processing complete!")
        print(f"📈 Database contains {len(database)} fingerprint entries")
        print("🚀 Ready for matching and verification!")
        
        # Test the matching system
        print(f"\n🧪 Testing matching system...")
        test_results = test_matching_system(database_file)


if __name__ == "__main__":
    main()
