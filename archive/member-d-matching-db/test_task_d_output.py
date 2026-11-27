#!/usr/bin/env python3
"""
Test script to demonstrate Task D (Matcher) output format
"""

import os
import sys
from fingerprint_matching import process_folder_with_task_d_output, detect_all_images_in_folder

def test_task_d_output():
    """Test the Task D output format with a few sample images"""
    print("🎯 Testing Task D (Matcher) Output Format")
    print("=" * 60)
    
    # Test with the existing archive folder
    test_folder = "archive/fingerprints"
    
    if not os.path.exists(test_folder):
        print(f"❌ Test folder not found: {test_folder}")
        print("💡 Please ensure the fingerprint dataset is available")
        return
    
    print(f"📁 Testing with folder: {test_folder}")
    print("-" * 40)
    
    # First, detect images to see what we have
    print("\n🔍 Step 1: Detecting images...")
    detected_images = detect_all_images_in_folder(test_folder, recursive=True)
    
    if not detected_images:
        print("❌ No images found!")
        return
    
    print(f"✅ Found {len(detected_images)} images")
    
    # Process just the first 3 images for demonstration
    print(f"\n🔄 Step 2: Processing first 3 images for Task D output...")
    print("=" * 60)
    
    # Create a temporary folder with just 3 images for testing
    import shutil
    temp_folder = "temp_test_images"
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)
    
    # Copy first 3 images to temp folder
    for i, image_path in enumerate(detected_images[:3]):
        dest_path = os.path.join(temp_folder, os.path.basename(image_path))
        shutil.copy2(image_path, dest_path)
        print(f"📋 Copied: {os.path.basename(image_path)}")
    
    # Process the temp folder
    result = process_folder_with_task_d_output(temp_folder, recursive=False)
    
    if result['success']:
        print(f"\n🎉 Task D Processing Complete!")
        print(f"📊 Summary:")
        print(f"   • Images found: {result['total_images_found']}")
        print(f"   • Images processed: {result['total_images_processed']}")
        
        # Show detailed Task D output for each image
        print(f"\n📋 Detailed Task D (Matcher) Output:")
        print("=" * 60)
        
        for i, result_entry in enumerate(result['results'], 1):
            print(f"\n🔍 Image {i}: {result_entry['image_name']}")
            print("-" * 40)
            
            data = result_entry['security_enhanced_matching_data']
            print(f"# Security-enhanced matching data:")
            print(f"{{")
            print(f'  "minutiae_count": {data["minutiae_count"]},')
            print(f'  "spoof_risk_score": {data["spoof_risk_score"]:.3f},')
            print(f'  "quality_score": {data["quality_score"]:.3f},')
            print(f'  "minutiae": [')
            
            # Show first few minutiae points
            for j, minutiae in enumerate(data['minutiae'][:3]):
                print(f'    {{"x": {minutiae["x"]}, "y": {minutiae["y"]}, "type": "{minutiae["type"]}", "confidence": {minutiae["confidence"]:.3f}, "angle": {minutiae["angle"]:.3f}}},')
            
            if len(data['minutiae']) > 3:
                print(f'    ... # {len(data["minutiae"]) - 3} more minutiae points')
            
            print(f'  ]')
            print(f"}}")
    
    # Clean up temp folder
    shutil.rmtree(temp_folder)
    print(f"\n🧹 Cleaned up temporary files")
    
    print(f"\n🎉 Task D Output Test Complete!")
    print("=" * 60)

def test_single_image_task_d():
    """Test Task D output for a single image"""
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\n🎯 Testing Single Image Task D Output: {image_path}")
        print("-" * 40)
        
        if os.path.exists(image_path):
            from fingerprint_matching import extract_fingerprint_features, generate_security_enhanced_matching_data
            
            # Extract features
            features = extract_fingerprint_features(image_path)
            if features:
                # Generate Task D format data
                task_d_data = generate_security_enhanced_matching_data(features)
                
                print(f"✅ Successfully processed: {os.path.basename(image_path)}")
                print(f"\n📊 Task D (Matcher) Output:")
                print(f"{{")
                print(f'  "minutiae_count": {task_d_data["minutiae_count"]},')
                print(f'  "spoof_risk_score": {task_d_data["spoof_risk_score"]:.3f},')
                print(f'  "quality_score": {task_d_data["quality_score"]:.3f},')
                print(f'  "minutiae": [...] # {len(task_d_data["minutiae"])} minutiae points')
                print(f"}}")
            else:
                print(f"❌ Could not process image: {image_path}")
        else:
            print(f"❌ Image not found: {image_path}")
    else:
        print("\n💡 To test with a single image, run:")
        print("   python test_task_d_output.py /path/to/your/image.tif")

if __name__ == "__main__":
    test_task_d_output()
    test_single_image_task_d()
