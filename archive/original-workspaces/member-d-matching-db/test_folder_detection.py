#!/usr/bin/env python3
"""
Test script to demonstrate the new folder detection functionality
"""

import os
import sys
from fingerprint_matching import detect_all_images_in_folder, process_images_from_detected_list

def test_folder_detection():
    """Test the new folder detection functionality"""
    print("🧪 Testing Folder Detection Functionality")
    print("=" * 60)
    
    # Test with the existing archive folder
    test_folder = "archive/fingerprints"
    
    if not os.path.exists(test_folder):
        print(f"❌ Test folder not found: {test_folder}")
        print("💡 Please ensure the fingerprint dataset is available")
        return
    
    print(f"📁 Testing with folder: {test_folder}")
    print("-" * 40)
    
    # Test 1: Detect images recursively
    print("\n🔍 Test 1: Recursive Detection")
    detected_images = detect_all_images_in_folder(test_folder, recursive=True)
    
    if detected_images:
        print(f"✅ Found {len(detected_images)} images recursively")
        print(f"📸 First few images:")
        for i, img in enumerate(detected_images[:3]):
            print(f"   {i+1}. {os.path.basename(img)}")
    else:
        print("❌ No images found recursively")
    
    # Test 2: Detect images non-recursively
    print("\n🔍 Test 2: Non-Recursive Detection")
    detected_images_non_recursive = detect_all_images_in_folder(test_folder, recursive=False)
    
    if detected_images_non_recursive:
        print(f"✅ Found {len(detected_images_non_recursive)} images in immediate directory")
    else:
        print("❌ No images found in immediate directory")
    
    # Test 3: Process a few detected images
    if detected_images:
        print(f"\n🔄 Test 3: Processing Sample Images")
        print("-" * 40)
        
        # Process first 3 images
        sample_images = detected_images[:3]
        log_file = "test_processed_images.txt"
        database_file = "test_fingerprint_database.json"
        
        print(f"Processing {len(sample_images)} sample images...")
        
        try:
            processed_files, database = process_images_from_detected_list(
                sample_images, log_file, database_file, base_folder=test_folder
            )
            
            print(f"✅ Successfully processed {len(processed_files)} images")
            print(f"📊 Database contains {len(database)} entries")
            
            # Show some details about processed images
            for fingerprint_id, features in list(database.items())[:2]:
                print(f"\n📋 Sample Entry: {fingerprint_id}")
                print(f"   📊 Minutiae Count: {features['minutiae']['minutiae_count']}")
                print(f"   🎯 Quality Score: {features['minutiae'].get('quality_score', 0.0):.3f}")
                print(f"   🛡️  Spoof Risk: {features['minutiae'].get('spoof_risk_score', 0.0):.3f}")
            
            # Clean up test files
            if os.path.exists(log_file):
                os.remove(log_file)
            if os.path.exists(database_file):
                os.remove(database_file)
            print(f"\n🧹 Cleaned up test files")
            
        except Exception as e:
            print(f"❌ Error processing images: {e}")
    
    print(f"\n🎉 Folder Detection Test Complete!")
    print("=" * 60)

def test_custom_folder():
    """Test with a custom folder if provided"""
    if len(sys.argv) > 1:
        custom_folder = sys.argv[1]
        print(f"\n🎯 Testing Custom Folder: {custom_folder}")
        print("-" * 40)
        
        if os.path.exists(custom_folder):
            detected_images = detect_all_images_in_folder(custom_folder, recursive=True)
            print(f"📊 Found {len(detected_images)} images in custom folder")
            
            if detected_images:
                print("📸 Sample images found:")
                for i, img in enumerate(detected_images[:5]):
                    print(f"   {i+1}. {os.path.basename(img)}")
        else:
            print(f"❌ Custom folder not found: {custom_folder}")
    else:
        print("\n💡 To test with a custom folder, run:")
        print("   python test_folder_detection.py /path/to/your/folder")

if __name__ == "__main__":
    test_folder_detection()
    test_custom_folder()
