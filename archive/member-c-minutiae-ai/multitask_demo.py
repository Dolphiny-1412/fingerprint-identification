#!/usr/bin/env python3
"""
Multi-Task Fingerprint Classification Demo
==========================================

Interactive demonstration of the dual classification system:
1. Pattern Classification (Arch/Loop/Whorl)
2. Finger Identity Classification (10 finger positions)
"""

import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from multitask_predict import MultiTaskPredictor

def demo_single_prediction():
    """Demonstrate single image prediction"""
    print("🔍 Single Image Prediction Demo")
    print("=" * 40)
    
    # Find a sample image
    data_root = r"C:\Users\alifa\Downloads\fingerprint_model-20250811T194218Z-1-001\fingerprint_model\data"
    sample_path = os.path.join(data_root, "archive", "fingerprints", "DB1_B", "101_1.tif")
    
    if not os.path.exists(sample_path):
        print("❌ Sample image not found")
        return
    
    # Initialize predictor
    predictor = MultiTaskPredictor("multitask_output/models")
    
    # Make prediction
    result = predictor.predict_single_image(sample_path, include_confidence=True)
    
    # Display results
    print(predictor.format_prediction_output(result, verbose=True))
    
    return result

def demo_batch_analysis():
    """Demonstrate batch analysis with statistics"""
    print("\n📊 Batch Analysis Demo")
    print("=" * 40)
    
    # Load existing batch results if available
    batch_file = "batch_test_results.json"
    if not os.path.exists(batch_file):
        print("❌ Batch results not found. Run batch prediction first.")
        return
    
    # Load results
    with open(batch_file, 'r') as f:
        results = json.load(f)
    
    # Filter successful predictions
    successful = [r for r in results if r['success']]
    
    print(f"📈 Analysis of {len(successful)} successful predictions:")
    print()
    
    # Pattern distribution
    patterns = [r['pattern_prediction'] for r in successful if r['pattern_prediction']]
    pattern_counts = {}
    for pattern in patterns:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    print("🎯 Pattern Classification Results:")
    for pattern, count in sorted(pattern_counts.items()):
        percentage = count / len(patterns) * 100
        print(f"  {pattern}: {count} images ({percentage:.1f}%)")
    
    # Finger distribution
    fingers = [r['finger_prediction'] for r in successful if r['finger_prediction']]
    finger_counts = {}
    for finger in fingers:
        finger_counts[finger] = finger_counts.get(finger, 0) + 1
    
    print("\n👆 Finger Identity Results:")
    for finger, count in sorted(finger_counts.items()):
        percentage = count / len(fingers) * 100
        print(f"  {finger}: {count} images ({percentage:.1f}%)")
    
    # Confidence analysis
    pattern_confidences = [r['pattern_confidence'] for r in successful if r['pattern_confidence']]
    finger_confidences = [r['finger_confidence'] for r in successful if r['finger_confidence']]
    
    print(f"\n📊 Confidence Analysis:")
    if pattern_confidences:
        print(f"  Pattern Confidence: {sum(pattern_confidences)/len(pattern_confidences):.1%} ± {(max(pattern_confidences)-min(pattern_confidences))/4:.1%}")
    if finger_confidences:
        print(f"  Finger Confidence: {sum(finger_confidences)/len(finger_confidences):.1%} ± {(max(finger_confidences)-min(finger_confidences))/4:.1%}")
    
    # High confidence predictions
    high_conf_both = [r for r in successful 
                     if r.get('pattern_confidence', 0) > 0.9 
                     and r.get('finger_confidence', 0) > 0.8]
    
    print(f"\n🎯 High Confidence Predictions:")
    print(f"  Both tasks >90%/80%: {len(high_conf_both)} images ({len(high_conf_both)/len(successful)*100:.1f}%)")
    
    return successful

def demo_accuracy_comparison():
    """Compare parsed vs predicted finger identities"""
    print("\n🔍 Accuracy Comparison Demo")
    print("=" * 40)
    
    # Load batch results
    batch_file = "batch_test_results.json"
    if not os.path.exists(batch_file):
        print("❌ Batch results not found")
        return
    
    with open(batch_file, 'r') as f:
        results = json.load(f)
    
    # Filter results with both parsed and predicted finger identities
    valid_comparisons = [r for r in results 
                        if r['success'] 
                        and r.get('parsed_finger') != 'Unknown'
                        and r.get('finger_prediction')]
    
    if len(valid_comparisons) == 0:
        print("❌ No valid comparisons found")
        return
    
    print(f"🔍 Comparing {len(valid_comparisons)} predictions with parsed finger identities:")
    print()
    
    # Calculate accuracy
    correct = 0
    for result in valid_comparisons:
        if result['parsed_finger'] == result['finger_prediction']:
            correct += 1
    
    accuracy = correct / len(valid_comparisons)
    print(f"📊 Finger Identity Accuracy: {accuracy:.1%} ({correct}/{len(valid_comparisons)})")
    
    # Show some examples
    print(f"\n📷 Sample Comparisons:")
    print("-" * 60)
    for i, result in enumerate(valid_comparisons[:5]):
        filename = result['filename']
        parsed = result['parsed_finger']
        predicted = result['finger_prediction']
        confidence = result.get('finger_confidence', 0)
        match = "✅" if parsed == predicted else "❌"
        
        print(f"{match} {filename}")
        print(f"   Parsed: {parsed}")
        print(f"   Predicted: {predicted} ({confidence:.1%})")
        print()
    
    if len(valid_comparisons) > 5:
        print(f"... and {len(valid_comparisons)-5} more comparisons")
    
    # Misclassification analysis
    errors = [r for r in valid_comparisons 
             if r['parsed_finger'] != r['finger_prediction']]
    
    if errors:
        print(f"\n❌ Common Misclassifications:")
        error_patterns = {}
        for error in errors:
            pattern = f"{error['parsed_finger']} → {error['finger_prediction']}"
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
        
        for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"   {pattern}: {count} cases")

def create_demo_visualization():
    """Create visualization of demo results"""
    print("\n📊 Creating Demo Visualization")
    print("=" * 40)
    
    # Load batch results
    batch_file = "batch_test_results.json"
    if not os.path.exists(batch_file):
        print("❌ Batch results not found")
        return
    
    with open(batch_file, 'r') as f:
        results = json.load(f)
    
    successful = [r for r in results if r['success']]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Task Fingerprint Classification Demo Results', fontsize=16, fontweight='bold')
    
    # Pattern distribution
    patterns = [r['pattern_prediction'] for r in successful if r['pattern_prediction']]
    pattern_counts = {}
    for pattern in patterns:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    ax = axes[0, 0]
    bars = ax.bar(pattern_counts.keys(), pattern_counts.values(), color=['lightcoral', 'lightblue', 'lightgreen'])
    ax.set_title('Pattern Classification Distribution')
    ax.set_ylabel('Count')
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    # Finger distribution
    fingers = [r['finger_prediction'] for r in successful if r['finger_prediction']]
    finger_counts = {}
    for finger in fingers:
        finger_counts[finger] = finger_counts.get(finger, 0) + 1
    
    ax = axes[0, 1]
    finger_names = list(finger_counts.keys())
    finger_values = list(finger_counts.values())
    
    bars = ax.barh(finger_names, finger_values, color='lightcyan')
    ax.set_title('Finger Identity Distribution')
    ax.set_xlabel('Count')
    
    # Confidence distributions
    pattern_confidences = [r['pattern_confidence'] for r in successful if r.get('pattern_confidence')]
    finger_confidences = [r['finger_confidence'] for r in successful if r.get('finger_confidence')]
    
    ax = axes[1, 0]
    if pattern_confidences:
        ax.hist(pattern_confidences, bins=20, alpha=0.7, color='skyblue', label='Pattern')
        ax.axvline(sum(pattern_confidences)/len(pattern_confidences), color='red', linestyle='--', label='Mean')
    ax.set_title('Pattern Confidence Distribution')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.legend()
    
    ax = axes[1, 1]
    if finger_confidences:
        ax.hist(finger_confidences, bins=20, alpha=0.7, color='lightgreen', label='Finger')
        ax.axvline(sum(finger_confidences)/len(finger_confidences), color='red', linestyle='--', label='Mean')
    ax.set_title('Finger Confidence Distribution')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('multitask_demo_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Demo visualization saved: multitask_demo_results.png")

def main():
    """Run complete demo"""
    print("🚀 Multi-Task Fingerprint Classification Demo")
    print("=" * 60)
    print("This demo showcases the dual classification capabilities:")
    print("1. Pattern Classification (Arch/Loop/Whorl)")
    print("2. Finger Identity (10 finger positions)")
    print()
    
    # Check if models exist
    if not os.path.exists("multitask_output/models"):
        print("❌ Models not found. Please run the training pipeline first:")
        print("   python multitask_fingerprint_pipeline.py")
        return
    
    try:
        # Demo single prediction
        single_result = demo_single_prediction()
        
        # Demo batch analysis
        batch_results = demo_batch_analysis()
        
        # Demo accuracy comparison
        demo_accuracy_comparison()
        
        # Create visualization
        create_demo_visualization()
        
        print("\n🎉 Multi-Task Demo Complete!")
        print("=" * 60)
        print("✅ Single image prediction demonstrated")
        print("✅ Batch analysis completed")
        print("✅ Accuracy comparison shown")
        print("✅ Visualization generated")
        print("\n📂 Check 'multitask_demo_results.png' for visual summary")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
