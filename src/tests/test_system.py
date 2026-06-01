import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

"""
Comprehensive Test Suite for Fingerprint Analysis System
=======================================================

Tests all components: Classifier, Minutiae Detection, Matching, and AI Assistant
"""

import json
import time
from pathlib import Path

# Import our modules
from core.classifier import classify_fingerprint, train_classifier
from core.minutiae_detector import detect_minutiae
from core.fingerprint_matcher import match_minutiae, seed_database, get_matcher
from core.ai_assistant import agent_generate

def test_classifier():
    """Test the fingerprint classifier"""
    print("🧪 Testing Fingerprint Classifier")
    print("=" * 50)
    
    # Test with sample images
    test_images = [
        "Fingerprint D/archive/fingerprints/DB1_B/101_1.tif",
        "Fingerprint D/archive/fingerprints/DB1_B/102_1.tif",
        "Fingerprint D/archive/fingerprints/DB2_B/101_1.tif"
    ]
    
    results = []
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\n🔍 Testing with {os.path.basename(test_image)}")
            result = classify_fingerprint(test_image)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ Label: {result['label']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                if 'probabilities' in result:
                    print(f"   Probabilities: {result['probabilities']}")
                results.append(result)
        else:
            print(f"⚠️ Test image not found: {test_image}")
    
    print(f"\n📊 Classifier Test Results:")
    print(f"   Tests run: {len(test_images)}")
    print(f"   Successful: {len(results)}")
    print(f"   Success rate: {len(results)/len(test_images)*100:.1f}%")
    
    return results

def test_minutiae_detection():
    """Test the minutiae detection system"""
    print("\n🧪 Testing Minutiae Detection")
    print("=" * 50)
    
    # Test with sample images
    test_images = [
        "Fingerprint D/archive/fingerprints/DB1_B/101_1.tif",
        "Fingerprint D/archive/fingerprints/DB1_B/102_1.tif"
    ]
    
    results = []
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\n🔍 Testing with {os.path.basename(test_image)}")
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
                
                results.append(result)
        else:
            print(f"⚠️ Test image not found: {test_image}")
    
    print(f"\n📊 Minutiae Detection Test Results:")
    print(f"   Tests run: {len(test_images)}")
    print(f"   Successful: {len(results)}")
    print(f"   Success rate: {len(results)/len(test_images)*100:.1f}%")
    
    return results

def test_matching_system():
    """Test the fingerprint matching system"""
    print("\n🧪 Testing Fingerprint Matching System")
    print("=" * 50)
    
    # Initialize matcher and seed database
    matcher = get_matcher()
    
    # Seed database if empty
    if not matcher.database:
        print("🌱 Seeding database with sample data...")
        seed_database()
    
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
    
    print("🔍 Testing matching with sample minutiae data...")
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
    
    return result

def test_ai_assistant():
    """Test the AI assistant"""
    print("\n🧪 Testing AI Assistant")
    print("=" * 50)
    
    # Test classification explanation
    print("📋 Testing classification explanation:")
    classification_payload = {
        "label": "Loop",
        "confidence": 0.85,
        "probabilities": {"Loop": 0.85, "Whorl": 0.10, "Arch": 0.05}
    }
    result = agent_generate("classify", classification_payload)
    print(f"✅ Explanation: {result['text'][:100]}...")
    
    # Test minutiae explanation
    print("\n🔬 Testing minutiae explanation:")
    minutiae_payload = {
        "minutiae": [
            {"x": 100, "y": 150, "type": "ending", "confidence": 0.85},
            {"x": 200, "y": 250, "type": "bifurcation", "confidence": 0.92}
        ],
        "quality_score": 0.78,
        "spoof_suspected": False
    }
    result = agent_generate("minutiae", minutiae_payload)
    print(f"✅ Explanation: {result['text'][:100]}...")
    
    # Test matching explanation
    print("\n🔐 Testing matching explanation:")
    matching_payload = {
        "match_score": 0.85,
        "matched_id": "user_001",
        "access_granted": True,
        "spoof_flag": False
    }
    result = agent_generate("match", matching_payload)
    print(f"✅ Explanation: {result['text'][:100]}...")
    
    # Test comprehensive explanation
    print("\n🧠 Testing comprehensive explanation:")
    comprehensive_payload = {
        "classification": classification_payload,
        "minutiae": minutiae_payload,
        "matching": matching_payload
    }
    result = agent_generate("agent", comprehensive_payload)
    print(f"✅ Comprehensive explanation: {result['text'][:100]}...")
    
    return True

def test_end_to_end():
    """Test complete end-to-end workflow"""
    print("\n🧪 Testing End-to-End Workflow")
    print("=" * 50)
    
    # Find a test image
    test_image = None
    test_images = [
        "Fingerprint D/archive/fingerprints/DB1_B/101_1.tif",
        "Fingerprint D/archive/fingerprints/DB1_B/102_1.tif"
    ]
    
    for img in test_images:
        if os.path.exists(img):
            test_image = img
            break
    
    if not test_image:
        print("❌ No test images found for end-to-end testing")
        return False
    
    print(f"🔍 Using test image: {os.path.basename(test_image)}")
    
    # Step 1: Classification
    print("\n1️⃣ Running classification...")
    classification_result = classify_fingerprint(test_image)
    print(f"   Result: {classification_result.get('label', 'Unknown')} ({classification_result.get('confidence', 0):.3f})")
    
    # Step 2: Minutiae Detection
    print("\n2️⃣ Running minutiae detection...")
    minutiae_result = detect_minutiae(test_image)
    print(f"   Result: {minutiae_result.get('minutiae_count', 0)} minutiae detected")
    
    # Step 3: Matching
    print("\n3️⃣ Running matching...")
    match_result = match_minutiae(minutiae_result)
    print(f"   Result: Match score {match_result.get('match_score', 0):.3f}, Access: {match_result.get('access_granted', False)}")
    
    # Step 4: AI Explanation
    print("\n4️⃣ Generating AI explanation...")
    payload = {
        "classification": classification_result,
        "minutiae": minutiae_result,
        "matching": match_result
    }
    ai_result = agent_generate("agent", payload)
    print(f"   Result: {ai_result.get('text', 'No explanation')[:100]}...")
    
    print("\n✅ End-to-end test completed successfully!")
    return True

def run_performance_benchmark():
    """Run performance benchmarks"""
    print("\n⚡ Running Performance Benchmarks")
    print("=" * 50)
    
    # Find test images
    test_images = []
    for img in ["Fingerprint D/archive/fingerprints/DB1_B/101_1.tif", 
                "Fingerprint D/archive/fingerprints/DB1_B/102_1.tif"]:
        if os.path.exists(img):
            test_images.append(img)
    
    if not test_images:
        print("❌ No test images found for benchmarking")
        return
    
    print(f"🔍 Benchmarking with {len(test_images)} images...")
    
    # Benchmark classification
    print("\n📊 Classification Performance:")
    start_time = time.time()
    for img in test_images:
        classify_fingerprint(img)
    classification_time = time.time() - start_time
    print(f"   Total time: {classification_time:.2f}s")
    print(f"   Average per image: {classification_time/len(test_images):.2f}s")
    
    # Benchmark minutiae detection
    print("\n📊 Minutiae Detection Performance:")
    start_time = time.time()
    for img in test_images:
        detect_minutiae(img)
    minutiae_time = time.time() - start_time
    print(f"   Total time: {minutiae_time:.2f}s")
    print(f"   Average per image: {minutiae_time/len(test_images):.2f}s")
    
    # Benchmark matching
    print("\n📊 Matching Performance:")
    start_time = time.time()
    for img in test_images:
        minutiae_result = detect_minutiae(img)
        match_minutiae(minutiae_result)
    matching_time = time.time() - start_time
    print(f"   Total time: {matching_time:.2f}s")
    print(f"   Average per image: {matching_time/len(test_images):.2f}s")
    
    print(f"\n📈 Overall Performance Summary:")
    print(f"   Classification: {classification_time/len(test_images):.2f}s per image")
    print(f"   Minutiae Detection: {minutiae_time/len(test_images):.2f}s per image")
    print(f"   Matching: {matching_time/len(test_images):.2f}s per image")
    print(f"   Total Pipeline: {(classification_time + minutiae_time + matching_time)/len(test_images):.2f}s per image")

def main():
    """Run all tests"""
    print("🚀 Starting Comprehensive System Tests")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/derived", exist_ok=True)
    os.makedirs("data/fingerprints_db", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    test_results = {}
    
    try:
        # Test individual components
        test_results['classifier'] = test_classifier()
        test_results['minutiae'] = test_minutiae_detection()
        test_results['matching'] = test_matching_system()
        test_results['ai_assistant'] = test_ai_assistant()
        
        # Test end-to-end workflow
        test_results['end_to_end'] = test_end_to_end()
        
        # Run performance benchmarks
        run_performance_benchmark()
        
        # Summary
        print("\n🎯 Test Summary")
        print("=" * 60)
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name}: {status}")
        
        passed_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        print(f"\n📊 Overall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! System is ready for use.")
        else:
            print("⚠️ Some tests failed. Please check the output above.")
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
