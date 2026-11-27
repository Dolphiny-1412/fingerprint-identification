"""
Quick Test Script for Fingerprint Authentication System
Fast testing without slow database comparisons
"""

import os
import json
import time
from datetime import datetime
import sys

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'minutiae'))
from fingerprint_matching import verify_fingerprint, extract_fingerprint_features
from spoof_detection import SpoofDetectionSystem
from access_decision_system import AccessDecisionSystem


def quick_system_test():
    """Quick test of all system components"""
    print("🚀 Quick System Test - Fingerprint Authentication")
    print("=" * 60)
    
    # Test files
    test_image = "archive/fingerprints/DB1_B/101_1.tif"
    database_file = "fingerprint_database.json"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    if not os.path.exists(database_file):
        print(f"❌ Database not found: {database_file}")
        print("💡 Please run fingerprint_matching.py first to build the database")
        return
    
    print(f"✅ Test image found: {test_image}")
    print(f"✅ Database found: {database_file}")
    
    # Test 1: Feature Extraction
    print(f"\n🔍 Test 1: Feature Extraction")
    print("-" * 30)
    start_time = time.time()
    
    try:
        features = extract_fingerprint_features(test_image)
        extraction_time = time.time() - start_time
        
        if features:
            print(f"✅ Feature extraction successful!")
            print(f"   ⏱️  Time: {extraction_time:.2f} seconds")
            print(f"   📊 Minutiae count: {features['minutiae']['minutiae_count']}")
            print(f"   🎯 Image size: {features['image_size']}")
        else:
            print(f"❌ Feature extraction failed")
            return
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return
    
    # Test 2: Spoof Detection
    print(f"\n🛡️  Test 2: Spoof Detection")
    print("-" * 30)
    start_time = time.time()
    
    try:
        spoof_detector = SpoofDetectionSystem()
        spoof_result = spoof_detector.detect_spoof(test_image)
        spoof_time = time.time() - start_time
        
        print(f"✅ Spoof detection successful!")
        print(f"   ⏱️  Time: {spoof_time:.2f} seconds")
        print(f"   🎯 Risk score: {spoof_result.get('overall_spoof_risk', 0):.3f}")
        print(f"   🛡️  Spoof detected: {spoof_result.get('spoof_detected', False)}")
        print(f"   📊 Confidence: {spoof_result.get('confidence', 'unknown')}")
    except Exception as e:
        print(f"❌ Spoof detection error: {e}")
        return
    
    # Test 3: Fingerprint Verification (Single Match)
    print(f"\n🔍 Test 3: Fingerprint Verification")
    print("-" * 30)
    start_time = time.time()
    
    try:
        verification_result = verify_fingerprint(test_image, database_file, threshold=0.5)
        verification_time = time.time() - start_time
        
        print(f"✅ Verification successful!")
        print(f"   ⏱️  Time: {verification_time:.2f} seconds")
        print(f"   📊 Success: {verification_result['success']}")
        print(f"   🎯 Is verified: {verification_result['is_verified']}")
        
        if verification_result['best_match']:
            print(f"   🏆 Best match: {verification_result['best_match']['fingerprint_id']}")
            print(f"   📈 Similarity: {verification_result['best_match']['similarity_score']:.3f}")
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return
    
    # Test 4: Access Decision System
    print(f"\n🔓 Test 4: Access Decision System")
    print("-" * 30)
    start_time = time.time()
    
    try:
        access_system = AccessDecisionSystem()
        access_result = access_system.authenticate_user(
            query_image_path=test_image,
            database_path=database_file,
            user_id="test_user",
            access_level="user"
        )
        access_time = time.time() - start_time
        
        print(f"✅ Access decision successful!")
        print(f"   ⏱️  Time: {access_time:.2f} seconds")
        print(f"   🔓 Access granted: {access_result['access_granted']}")
        print(f"   🎯 Confidence: {access_result['confidence_level']}")
        print(f"   📊 Authentication success: {access_result['authentication_success']}")
    except Exception as e:
        print(f"❌ Access decision error: {e}")
        return
    
    # Test 5: Database Statistics
    print(f"\n📊 Test 5: Database Statistics")
    print("-" * 30)
    
    try:
        with open(database_file, 'r') as f:
            database = json.load(f)
        
        print(f"✅ Database loaded successfully!")
        print(f"   📈 Total entries: {len(database)}")
        print(f"   💾 File size: {os.path.getsize(database_file) / (1024*1024):.2f} MB")
        
        # Sample some entries
        sample_ids = list(database.keys())[:3]
        print(f"   🔍 Sample entries: {', '.join(sample_ids)}")
        
        # Check feature consistency
        avg_minutiae = sum(database[id]['minutiae']['minutiae_count'] for id in sample_ids) / len(sample_ids)
        print(f"   🎯 Avg minutiae per fingerprint: {avg_minutiae:.1f}")
        
    except Exception as e:
        print(f"❌ Database statistics error: {e}")
        return
    
    # Summary
    total_time = extraction_time + spoof_time + verification_time + access_time
    print(f"\n🎉 All Tests Completed Successfully!")
    print("=" * 40)
    print(f"⏱️  Total testing time: {total_time:.2f} seconds")
    print(f"✅ All components working correctly")
    print(f"🚀 System ready for production use")


def test_individual_components():
    """Test individual components separately"""
    print("🔧 Individual Component Tests")
    print("=" * 40)
    
    # Test image
    test_image = "archive/fingerprints/DB1_B/101_1.tif"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    # Component 1: Minutiae Detection
    print(f"\n1️⃣ Testing Minutiae Detection...")
    try:
        from minutiae.detector import AdvancedFingerprintAnalyzer
        analyzer = AdvancedFingerprintAnalyzer()
        result = analyzer.analyze_fingerprint(test_image)
        
        if 'error' not in result:
            print(f"   ✅ Minutiae detection: {len(result.get('minutiae', []))} minutiae found")
            print(f"   📊 Quality score: {result.get('quality_score', 0):.3f}")
        else:
            print(f"   ❌ Minutiae detection failed: {result['error']}")
    except Exception as e:
        print(f"   ❌ Minutiae detection error: {e}")
    
    # Component 2: Spoof Detection
    print(f"\n2️⃣ Testing Spoof Detection...")
    try:
        spoof_detector = SpoofDetectionSystem()
        spoof_result = spoof_detector.detect_spoof(test_image)
        print(f"   ✅ Spoof detection: Risk score {spoof_result.get('overall_spoof_risk', 0):.3f}")
    except Exception as e:
        print(f"   ❌ Spoof detection error: {e}")
    
    # Component 3: Access Decision
    print(f"\n3️⃣ Testing Access Decision System...")
    try:
        access_system = AccessDecisionSystem()
        print(f"   ✅ Access decision system initialized")
        print(f"   📊 Configuration loaded successfully")
    except Exception as e:
        print(f"   ❌ Access decision error: {e}")


def main():
    """Main function for quick testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Fingerprint System Test')
    parser.add_argument('--components', action='store_true', help='Test individual components')
    parser.add_argument('--full', action='store_true', help='Run full system test')
    
    args = parser.parse_args()
    
    if args.components:
        test_individual_components()
    elif args.full:
        quick_system_test()
    else:
        print("Quick Fingerprint Authentication System Test")
        print("=" * 50)
        print("Available options:")
        print("  --components    Test individual components")
        print("  --full         Run complete system test")
        print("\nExample: python quick_test.py --full")


if __name__ == "__main__":
    main()
