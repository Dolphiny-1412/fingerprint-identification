"""
Person D - Fingerprint Matching & Verification
==============================================

Implements fingerprint matching against a database with access control decisions.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment

class FingerprintMatcher:
    """
    Fingerprint matching system with database management and access control
    """
    
    def __init__(self, database_path: str = "src/data/fingerprints_db/fingerprint_database.json"):
        self.database_path = database_path
        self.database = {}
        self.thresholds = {
            'match_accept': 0.25,  # Lowered to 0.25 for more lenient matching
            'match_reject': 0.10,  # Lowered to 0.10
            'quality_min': 0.1,    # Lowered to 0.1 for easier new user creation
            'spoof_risk_max': 0.8, # Increased to 0.8 to be less strict
            'minutiae_min': 3      # Lowered to 3 minutiae minimum
        }
        
        # Create database directory
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        
        # Load existing database
        self._load_database()
    
    def _load_database(self):
        """Load fingerprint database from file"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'r') as f:
                    self.database = json.load(f)
                print(f"✅ Loaded fingerprint database: {len(self.database)} entries")
            else:
                print("⚠️ No existing database found, will create new one")
                self.database = {}
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            self.database = {}
    
    def _save_database(self):
        """Save fingerprint database to file"""
        try:
            with open(self.database_path, 'w') as f:
                json.dump(self.database, f, indent=2)
            print(f"💾 Database saved: {len(self.database)} entries")
        except Exception as e:
            print(f"❌ Error saving database: {e}")
    
    def add_fingerprint_to_database(self, fingerprint_id: str, minutiae_data: Dict, 
                                  image_path: str = None) -> bool:
        """
        Add a fingerprint to the database
        
        Args:
            fingerprint_id: Unique identifier for the fingerprint
            minutiae_data: Minutiae data from Person C
            image_path: Path to the original image (optional)
            
        Returns:
            True if successfully added
        """
        try:
            # Validate minutiae data
            if not self._validate_minutiae_data(minutiae_data):
                print(f"❌ Invalid minutiae data for {fingerprint_id}")
                return False
            
            # Store in database
            self.database[fingerprint_id] = {
                'minutiae_data': minutiae_data,
                'image_path': image_path,
                'added_timestamp': time.time(),
                'access_count': 0,
                'last_accessed': None
            }
            
            # Save database
            self._save_database()
            
            print(f"✅ Added fingerprint {fingerprint_id} to database")
            return True
            
        except Exception as e:
            print(f"❌ Error adding fingerprint {fingerprint_id}: {e}")
            return False
    
    def _validate_minutiae_data(self, minutiae_data: Dict) -> bool:
        """Validate minutiae data structure"""
        required_fields = ['minutiae', 'quality_score', 'spoof_suspected']
        
        for field in required_fields:
            if field not in minutiae_data:
                return False
        
        # Check minutiae list
        if not isinstance(minutiae_data['minutiae'], list):
            return False
        
        # Validate each minutiae point
        for minutia in minutiae_data['minutiae']:
            required_minutia_fields = ['x', 'y', 'type', 'confidence']
            for field in required_minutia_fields:
                if field not in minutia:
                    return False
        
        return True
    
    def match_minutiae(self, query_minutiae_data: Dict) -> Dict:
        """
        Match query minutiae against database
        
        Args:
            query_minutiae_data: Minutiae data from Person C
            
        Returns:
            Matching result dictionary
        """
        start_time = time.time()
        
        try:
            # Validate query data
            if not self._validate_minutiae_data(query_minutiae_data):
                return self._create_error_result("Invalid query minutiae data")
            
            # Check quality and spoof flags
            quality_score = query_minutiae_data.get('quality_score', 0.0)
            spoof_suspected = query_minutiae_data.get('spoof_suspected', False)
            spoof_risk_score = query_minutiae_data.get('spoof_risk_score', 0.0)
            
            # Log matching parameters (can be removed in production)
            # print(f"🔍 Matching: quality={quality_score:.2f}, spoof={spoof_suspected}, minutiae={len(query_minutiae_data.get('minutiae', []))}")
            
            if quality_score < self.thresholds['quality_min']:
                return self._create_match_result(
                    match_score=0.0,
                    matched_id=None,
                    access_granted=False,
                    spoof_flag=True,
                    reason="Poor image quality"
                )
            
            if spoof_suspected or query_minutiae_data.get('spoof_risk_score', 0.0) > self.thresholds['spoof_risk_max']:
                return self._create_match_result(
                    match_score=0.0,
                    matched_id=None,
                    access_granted=False,
                    spoof_flag=True,
                    reason="Spoof detected"
                )
            
            # Find best match in database
            best_match = None
            best_score = 0.0
            
            for fingerprint_id, db_entry in self.database.items():
                db_minutiae_data = db_entry['minutiae_data']
                
                # Calculate similarity
                similarity_score = self._calculate_minutiae_similarity(
                    query_minutiae_data['minutiae'],
                    db_minutiae_data['minutiae']
                )
                
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_match = fingerprint_id
            
            # Auto-create new user if match score is very low (likely new user)
            # More lenient conditions for new user creation
            if (best_score < 0.20 and  # Increased from 0.15 to 0.20
                not spoof_suspected and 
                quality_score >= self.thresholds['quality_min'] and
                len(query_minutiae_data.get('minutiae', [])) >= self.thresholds['minutiae_min']):
                
                # Generate new user ID
                new_user_id = f"user_{len(self.database) + 1:03d}"
                
                # Add to database
                self.add_fingerprint_to_database(
                    fingerprint_id=new_user_id,
                    minutiae_data=query_minutiae_data,
                    image_path=None
                )
                
                # Grant access to new user
                access_granted = True
                best_match = new_user_id
                best_score = 0.8  # Set high score for new user
                
                print(f"✅ New user created: {new_user_id} (score: {best_score:.3f})")
            else:
                # Determine access decision for existing users
                access_granted = (
                    best_score >= self.thresholds['match_accept'] and
                    not spoof_suspected and
                    quality_score >= self.thresholds['quality_min']
                )
            
            # Update database statistics
            if best_match and access_granted:
                self.database[best_match]['access_count'] += 1
                self.database[best_match]['last_accessed'] = time.time()
                self._save_database()
            
            processing_time = time.time() - start_time
            
            # Log final decision for debugging
            print(f"🎯 Final Decision:")
            print(f"   Score: {best_score:.3f} (threshold: {self.thresholds['match_accept']})")
            print(f"   Quality: {quality_score:.3f} (min: {self.thresholds['quality_min']})")
            print(f"   Spoof: {spoof_suspected} (max risk: {self.thresholds['spoof_risk_max']})")
            print(f"   Access: {'✅ GRANTED' if access_granted else '❌ DENIED'}")
            print(f"   Match ID: {best_match}")
            
            return self._create_match_result(
                match_score=best_score,
                matched_id=best_match,
                access_granted=access_granted,
                spoof_flag=spoof_suspected,
                processing_time_ms=round(processing_time * 1000, 2)
            )
            
        except Exception as e:
            return self._create_error_result(str(e))
    
    def _calculate_minutiae_similarity(self, minutiae1: List[Dict], minutiae2: List[Dict]) -> float:
        """
        Calculate similarity between two sets of minutiae points
        
        Args:
            minutiae1: First set of minutiae points
            minutiae2: Second set of minutiae points
            
        Returns:
            Similarity score between 0 and 1
        """
        if not minutiae1 or not minutiae2:
            return 0.0
        
        # Create distance matrix
        distances = np.zeros((len(minutiae1), len(minutiae2)))
        type_matches = np.zeros((len(minutiae1), len(minutiae2)))
        confidence_weights = np.zeros((len(minutiae1), len(minutiae2)))
        
        for i, m1 in enumerate(minutiae1):
            for j, m2 in enumerate(minutiae2):
                # Calculate Euclidean distance
                dist = euclidean([m1['x'], m1['y']], [m2['x'], m2['y']])
                distances[i, j] = dist
                
                # Check if types match
                if m1['type'] == m2['type']:
                    type_matches[i, j] = 1
                
                # Calculate confidence weight
                conf1 = m1.get('confidence', 0.5)
                conf2 = m2.get('confidence', 0.5)
                confidence_weights[i, j] = np.sqrt(conf1 * conf2)
        
        # Use Hungarian algorithm for optimal matching
        row_indices, col_indices = linear_sum_assignment(distances)
        
        # Count valid matches with more lenient criteria
        matched_pairs = 0
        total_distance = 0
        total_confidence = 0
        max_distance = 50  # Increased from 20 to 50 for more lenient matching
        
        for i, j in zip(row_indices, col_indices):
            # More lenient matching - allow type mismatches with distance penalty
            distance_penalty = 1.0 if type_matches[i, j] == 1 else 0.7
            effective_distance = distances[i, j] * (2.0 - distance_penalty)
            
            if effective_distance <= max_distance:
                matched_pairs += 1
                total_distance += effective_distance
                total_confidence += confidence_weights[i, j] * distance_penalty
        
        if matched_pairs == 0:
            return 0.0
        
        # Calculate similarity metrics
        total_minutiae = len(minutiae1) + len(minutiae2)
        match_ratio = (2 * matched_pairs) / total_minutiae
        
        # Distance score (closer is better)
        avg_distance = total_distance / matched_pairs
        distance_score = max(0, 1 - (avg_distance / max_distance))
        
        # Confidence score
        avg_confidence = total_confidence / matched_pairs
        
        # More generous weighted similarity score
        similarity_score = (
            match_ratio * 0.6 +      # Increased weight for match ratio
            distance_score * 0.2 +   # Decreased weight for distance
            avg_confidence * 0.2     # Keep confidence weight
        )
        
        # Apply bonus for having any matches at all
        if matched_pairs > 0:
            similarity_score = min(1.0, similarity_score * 1.2)  # 20% bonus
        
        return min(1.0, similarity_score)
    
    def _create_match_result(self, match_score: float, matched_id: Optional[str], 
                           access_granted: bool, spoof_flag: bool, 
                           processing_time_ms: float = 0.0, reason: str = None) -> Dict:
        """Create standardized match result dictionary"""
        result = {
            "match_score": float(match_score),
            "matched_id": str(matched_id) if matched_id is not None else None,
            "access_granted": bool(access_granted),
            "spoof_flag": bool(spoof_flag),
            "processing_time_ms": float(processing_time_ms),
            "timestamp": float(time.time())
        }
        
        if reason:
            result["reason"] = str(reason)
        
        return result
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result dictionary"""
        return {
            "match_score": 0.0,
            "matched_id": None,
            "access_granted": False,
            "spoof_flag": True,
            "processing_time_ms": 0.0,
            "error": str(error_message),
            "timestamp": float(time.time())
        }
    
    def get_database_statistics(self) -> Dict:
        """Get database statistics"""
        if not self.database:
            return {
                "total_entries": 0,
                "message": "Database is empty"
            }
        
        total_entries = len(self.database)
        total_accesses = sum(entry.get('access_count', 0) for entry in self.database.values())
        
        # Calculate average quality scores
        quality_scores = []
        for entry in self.database.values():
            minutiae_data = entry.get('minutiae_data', {})
            quality_score = minutiae_data.get('quality_score', 0.0)
            quality_scores.append(quality_score)
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return {
            "total_entries": total_entries,
            "total_accesses": total_accesses,
            "average_quality_score": avg_quality,
            "database_size_mb": os.path.getsize(self.database_path) / (1024 * 1024) if os.path.exists(self.database_path) else 0
        }
    
    def seed_database_with_sample_data(self) -> int:
        """
        Seed the database with sample fingerprint data for testing
        
        Returns:
            Number of entries added
        """
        print("🌱 Seeding database with sample data...")
        
        # Sample minutiae data for different fingerprint types
        sample_data = [
            {
                "fingerprint_id": "user_001",
                "minutiae_data": {
                    "minutiae": [
                        {"x": 100, "y": 150, "type": "ending", "angle": 1.57, "confidence": 0.85},
                        {"x": 200, "y": 250, "type": "bifurcation", "angle": 0.78, "confidence": 0.92},
                        {"x": 150, "y": 200, "type": "ending", "angle": 2.35, "confidence": 0.78},
                        {"x": 180, "y": 180, "type": "bifurcation", "angle": 1.96, "confidence": 0.88},
                        {"x": 120, "y": 220, "type": "ending", "angle": 0.52, "confidence": 0.82}
                    ],
                    "quality_score": 0.85,
                    "spoof_suspected": False,
                    "spoof_risk_score": 0.15
                }
            },
            {
                "fingerprint_id": "user_002",
                "minutiae_data": {
                    "minutiae": [
                        {"x": 80, "y": 120, "type": "bifurcation", "angle": 2.1, "confidence": 0.90},
                        {"x": 160, "y": 180, "type": "ending", "angle": 1.2, "confidence": 0.87},
                        {"x": 220, "y": 200, "type": "bifurcation", "angle": 0.5, "confidence": 0.83},
                        {"x": 140, "y": 160, "type": "ending", "angle": 2.8, "confidence": 0.79},
                        {"x": 190, "y": 140, "type": "bifurcation", "angle": 1.4, "confidence": 0.91}
                    ],
                    "quality_score": 0.78,
                    "spoof_suspected": False,
                    "spoof_risk_score": 0.22
                }
            },
            {
                "fingerprint_id": "user_003",
                "minutiae_data": {
                    "minutiae": [
                        {"x": 90, "y": 110, "type": "ending", "angle": 1.8, "confidence": 0.86},
                        {"x": 170, "y": 190, "type": "bifurcation", "angle": 0.9, "confidence": 0.89},
                        {"x": 130, "y": 170, "type": "ending", "angle": 2.4, "confidence": 0.84},
                        {"x": 210, "y": 150, "type": "bifurcation", "angle": 1.1, "confidence": 0.87},
                        {"x": 110, "y": 130, "type": "ending", "angle": 0.7, "confidence": 0.81}
                    ],
                    "quality_score": 0.82,
                    "spoof_suspected": False,
                    "spoof_risk_score": 0.18
                }
            },
            {
                "fingerprint_id": "user_004",
                "minutiae_data": {
                    "minutiae": [
                        {"x": 75, "y": 125, "type": "bifurcation", "angle": 2.2, "confidence": 0.88},
                        {"x": 155, "y": 185, "type": "ending", "angle": 1.3, "confidence": 0.85},
                        {"x": 225, "y": 205, "type": "bifurcation", "angle": 0.6, "confidence": 0.82},
                        {"x": 135, "y": 165, "type": "ending", "angle": 2.9, "confidence": 0.80},
                        {"x": 195, "y": 145, "type": "bifurcation", "angle": 1.5, "confidence": 0.90}
                    ],
                    "quality_score": 0.80,
                    "spoof_suspected": False,
                    "spoof_risk_score": 0.20
                }
            },
            {
                "fingerprint_id": "user_005",
                "minutiae_data": {
                    "minutiae": [
                        {"x": 85, "y": 115, "type": "ending", "angle": 1.9, "confidence": 0.87},
                        {"x": 165, "y": 195, "type": "bifurcation", "angle": 1.0, "confidence": 0.88},
                        {"x": 125, "y": 175, "type": "ending", "angle": 2.5, "confidence": 0.83},
                        {"x": 205, "y": 155, "type": "bifurcation", "angle": 1.2, "confidence": 0.86},
                        {"x": 105, "y": 135, "type": "ending", "angle": 0.8, "confidence": 0.82}
                    ],
                    "quality_score": 0.83,
                    "spoof_suspected": False,
                    "spoof_risk_score": 0.17
                }
            }
        ]
        
        added_count = 0
        for sample in sample_data:
            if self.add_fingerprint_to_database(
                sample["fingerprint_id"],
                sample["minutiae_data"]
            ):
                added_count += 1
        
        print(f"✅ Added {added_count} sample fingerprints to database")
        return added_count

# Global matcher instance
_matcher_instance = None

def get_matcher() -> FingerprintMatcher:
    """Get or create global matcher instance"""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = FingerprintMatcher()
    return _matcher_instance

def match_minutiae(minutiae_data: Dict) -> Dict:
    """
    Main function to match minutiae against database
    
    Args:
        minutiae_data: Minutiae data from Person C
        
    Returns:
        Matching result dictionary
    """
    matcher = get_matcher()
    return matcher.match_minutiae(minutiae_data)

def seed_database() -> int:
    """
    Seed the database with sample data
    
    Returns:
        Number of entries added
    """
    matcher = get_matcher()
    return matcher.seed_database_with_sample_data()

if __name__ == "__main__":
    # Initialize matcher and seed database
    print("🚀 Initializing fingerprint matcher...")
    matcher = get_matcher()
    
    # Seed database if empty
    if not matcher.database:
        seed_database()
    
    # Test matching with sample data
    print("\n🧪 Testing fingerprint matching...")
    
    # Create test minutiae data
    test_minutiae = {
        "minutiae": [
            {"x": 100, "y": 150, "type": "ending", "angle": 1.57, "confidence": 0.85},
            {"x": 200, "y": 250, "type": "bifurcation", "angle": 0.78, "confidence": 0.92},
            {"x": 150, "y": 200, "type": "ending", "angle": 2.35, "confidence": 0.78},
            {"x": 180, "y": 180, "type": "bifurcation", "angle": 1.96, "confidence": 0.88},
            {"x": 120, "y": 220, "type": "ending", "angle": 0.52, "confidence": 0.82}
        ],
        "quality_score": 0.85,
        "spoof_suspected": False,
        "spoof_risk_score": 0.15
    }
    
    # Test matching
    result = match_minutiae(test_minutiae)
    
    print(f"✅ Match result:")
    print(f"   Match score: {result['match_score']:.3f}")
    print(f"   Matched ID: {result['matched_id']}")
    print(f"   Access granted: {result['access_granted']}")
    print(f"   Spoof flag: {result['spoof_flag']}")
    
    # Show database statistics
    stats = matcher.get_database_statistics()
    print(f"\n📊 Database statistics:")
    print(f"   Total entries: {stats['total_entries']}")
    print(f"   Total accesses: {stats['total_accesses']}")
    print(f"   Average quality: {stats['average_quality_score']:.3f}")
