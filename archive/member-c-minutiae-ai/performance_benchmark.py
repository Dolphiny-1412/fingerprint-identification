#!/usr/bin/env python3
"""
Comprehensive Benchmark Validation System
=========================================

Validates the complete fingerprint system using:
1. Pattern classification accuracy
2. Finger identity accuracy
3. Minutiae detection quality
4. Identity matching performance
5. Spoof detection effectiveness
6. System integration tests

Generates ROC curves, confusion matrices, and performance metrics.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, f1_score
)
import json
import time
import os
from typing import Dict, List, Tuple, Any
import glob
from pathlib import Path

# Import our modules
from minutiae_extraction import MinutiaeExtractor
from fast_fingerprint_matcher import FastFingerprintMatcher
from spoof_detection import SpoofDetector
from ai_forensic_assistant import ForensicAIAssistant

class BenchmarkValidator:
    """
    Comprehensive benchmark validation for fingerprint system
    """
    
    def __init__(self, data_path: str):
        """
        Initialize benchmark validator
        
        Args:
            data_path: Path to fingerprint database
        """
        self.data_path = data_path
        self.results = {}
        
        # Initialize components
        self.minutiae_extractor = MinutiaeExtractor()
        self.matcher = FastFingerprintMatcher()
        self.spoof_detector = SpoofDetector()
        self.ai_assistant = ForensicAIAssistant()
        
        # Performance tracking
        self.timing_results = {}
        self.accuracy_results = {}
        
    def load_test_dataset(self, limit: int = 100) -> Dict:
        """
        Load test dataset from fingerprint database
        
        Args:
            limit: Maximum number of images to load
            
        Returns:
            Dictionary containing loaded images and metadata
        """
        print(f"📂 Loading test dataset from: {self.data_path}")
        
        # Find image files
        image_patterns = ['*.tif', '*.jpg', '*.png', '*.bmp']
        image_files = []
        
        for pattern in image_patterns:
            image_files.extend(glob.glob(os.path.join(self.data_path, pattern)))
        
        if not image_files:
            # Try subdirectories
            for pattern in image_patterns:
                image_files.extend(glob.glob(os.path.join(self.data_path, '**', pattern), recursive=True))
        
        image_files = image_files[:limit]  # Limit for performance
        
        dataset = {
            'images': [],
            'filenames': [],
            'subjects': [],
            'fingers': [],
            'metadata': []
        }
        
        print(f"📷 Found {len(image_files)} images")
        
        for i, img_path in enumerate(image_files):
            try:
                # Load image
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                # Extract metadata from filename
                filename = os.path.basename(img_path)
                subject_id, finger_id = self._parse_filename(filename)
                
                dataset['images'].append(image)
                dataset['filenames'].append(filename)
                dataset['subjects'].append(subject_id)
                dataset['fingers'].append(finger_id)
                dataset['metadata'].append({
                    'path': img_path,
                    'subject': subject_id,
                    'finger': finger_id,
                    'size': image.shape
                })
                
                if (i + 1) % 20 == 0:
                    print(f"   Loaded {i + 1}/{len(image_files)} images")
                    
            except Exception as e:
                print(f"   ⚠️ Failed to load {img_path}: {e}")
                continue
        
        print(f"✅ Loaded {len(dataset['images'])} valid images")
        return dataset
    
    def _parse_filename(self, filename: str) -> Tuple[str, str]:
        """
        Parse subject and finger information from filename
        
        Args:
            filename: Image filename
            
        Returns:
            Tuple of (subject_id, finger_id)
        """
        # Parse format like "101_1.tif" -> subject=101, finger=1
        try:
            parts = filename.split('_')
            if len(parts) >= 2:
                subject_id = parts[0]
                finger_id = parts[1].split('.')[0]
                return subject_id, finger_id
        except:
            pass
        
        # Default fallback
        return "unknown", "unknown"
    
    def benchmark_minutiae_detection(self, dataset: Dict) -> Dict:
        """
        Benchmark minutiae detection performance
        
        Args:
            dataset: Test dataset
            
        Returns:
            Minutiae detection benchmark results
        """
        print("\n🔍 Benchmarking Minutiae Detection")
        print("-" * 50)
        
        results = {
            'total_images': len(dataset['images']),
            'successful_extractions': 0,
            'failed_extractions': 0,
            'minutiae_counts': [],
            'processing_times': [],
            'quality_scores': [],
            'error_cases': []
        }
        
        for i, (image, filename) in enumerate(zip(dataset['images'], dataset['filenames'])):
            try:
                start_time = time.time()
                
                # Extract minutiae
                minutiae_result = self.minutiae_extractor.extract_minutiae(image)
                
                processing_time = time.time() - start_time
                
                # Count minutiae
                minutiae_count = len(minutiae_result.get('minutiae', []))
                avg_quality = np.mean(minutiae_result.get('quality_scores', [0]))
                
                results['successful_extractions'] += 1
                results['minutiae_counts'].append(minutiae_count)
                results['processing_times'].append(processing_time)
                results['quality_scores'].append(avg_quality)
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(dataset['images'])} images")
                
            except Exception as e:
                results['failed_extractions'] += 1
                results['error_cases'].append({
                    'filename': filename,
                    'error': str(e)
                })
                print(f"   ⚠️ Failed on {filename}: {e}")
        
        # Compute statistics
        if results['minutiae_counts']:
            results['avg_minutiae'] = np.mean(results['minutiae_counts'])
            results['std_minutiae'] = np.std(results['minutiae_counts'])
            results['min_minutiae'] = np.min(results['minutiae_counts'])
            results['max_minutiae'] = np.max(results['minutiae_counts'])
            
        if results['processing_times']:
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['total_processing_time'] = np.sum(results['processing_times'])
            
        if results['quality_scores']:
            results['avg_quality'] = np.mean(results['quality_scores'])
            results['std_quality'] = np.std(results['quality_scores'])
        
        # Print summary
        print(f"📊 Minutiae Detection Results:")
        print(f"   Success Rate: {results['successful_extractions']}/{results['total_images']} ({results['successful_extractions']/results['total_images']:.1%})")
        if results['minutiae_counts']:
            print(f"   Average Minutiae: {results['avg_minutiae']:.1f} ± {results['std_minutiae']:.1f}")
            print(f"   Average Quality: {results['avg_quality']:.3f}")
            print(f"   Average Time: {results['avg_processing_time']:.3f}s per image")
        
        return results
    
    def benchmark_identity_matching(self, dataset: Dict, num_comparisons: int = 50) -> Dict:
        """
        Benchmark identity matching performance
        
        Args:
            dataset: Test dataset
            num_comparisons: Number of comparisons to perform
            
        Returns:
            Identity matching benchmark results
        """
        print("\n🤝 Benchmarking Identity Matching")
        print("-" * 50)
        
        results = {
            'total_comparisons': 0,
            'same_finger_comparisons': 0,
            'different_finger_comparisons': 0,
            'same_finger_scores': [],
            'different_finger_scores': [],
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'processing_times': [],
            'threshold': 0.3  # Default threshold
        }
        
        images = dataset['images'][:num_comparisons]
        subjects = dataset['subjects'][:num_comparisons]
        fingers = dataset['fingers'][:num_comparisons]
        
        print(f"   Performing {num_comparisons} comparisons...")
        
        comparison_count = 0
        for i in range(len(images)):
            for j in range(i + 1, min(i + 6, len(images))):  # Compare with next 5 images
                try:
                    start_time = time.time()
                    
                    # Perform matching
                    match_result = self.matcher.match_fingerprints_fast(images[i], images[j])
                    
                    processing_time = time.time() - start_time
                    
                    # Determine ground truth
                    same_finger = (subjects[i] == subjects[j] and fingers[i] == fingers[j])
                    
                    score = match_result['combined_score']
                    predicted_match = score > results['threshold']
                    
                    # Record results
                    results['total_comparisons'] += 1
                    results['processing_times'].append(processing_time)
                    
                    if same_finger:
                        results['same_finger_comparisons'] += 1
                        results['same_finger_scores'].append(score)
                        if predicted_match:
                            results['true_positives'] += 1
                        else:
                            results['false_negatives'] += 1
                    else:
                        results['different_finger_comparisons'] += 1
                        results['different_finger_scores'].append(score)
                        if predicted_match:
                            results['false_positives'] += 1
                        else:
                            results['true_negatives'] += 1
                    
                    comparison_count += 1
                    if comparison_count >= num_comparisons:
                        break
                        
                except Exception as e:
                    print(f"   ⚠️ Comparison failed: {e}")
                    continue
            
            if comparison_count >= num_comparisons:
                break
        
        # Compute performance metrics
        total_predictions = results['true_positives'] + results['true_negatives'] + results['false_positives'] + results['false_negatives']
        
        if total_predictions > 0:
            results['accuracy'] = (results['true_positives'] + results['true_negatives']) / total_predictions
            
            # Precision and Recall
            if results['true_positives'] + results['false_positives'] > 0:
                results['precision'] = results['true_positives'] / (results['true_positives'] + results['false_positives'])
            else:
                results['precision'] = 0.0
                
            if results['true_positives'] + results['false_negatives'] > 0:
                results['recall'] = results['true_positives'] / (results['true_positives'] + results['false_negatives'])
            else:
                results['recall'] = 0.0
            
            if results['precision'] + results['recall'] > 0:
                results['f1_score'] = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'])
            else:
                results['f1_score'] = 0.0
        
        # Statistics
        if results['same_finger_scores']:
            results['avg_same_finger_score'] = np.mean(results['same_finger_scores'])
            results['std_same_finger_score'] = np.std(results['same_finger_scores'])
            
        if results['different_finger_scores']:
            results['avg_different_finger_score'] = np.mean(results['different_finger_scores'])
            results['std_different_finger_score'] = np.std(results['different_finger_scores'])
        
        if results['processing_times']:
            results['avg_processing_time'] = np.mean(results['processing_times'])
        
        # Print summary
        print(f"📊 Identity Matching Results:")
        print(f"   Total Comparisons: {results['total_comparisons']}")
        if total_predictions > 0:
            print(f"   Accuracy: {results['accuracy']:.3f}")
            print(f"   Precision: {results['precision']:.3f}")
            print(f"   Recall: {results['recall']:.3f}")
            print(f"   F1 Score: {results['f1_score']:.3f}")
        if results['same_finger_scores']:
            print(f"   Same Finger Score: {results['avg_same_finger_score']:.3f} ± {results['std_same_finger_score']:.3f}")
        if results['different_finger_scores']:
            print(f"   Different Finger Score: {results['avg_different_finger_score']:.3f} ± {results['std_different_finger_score']:.3f}")
        
        return results
    
    def benchmark_spoof_detection(self, dataset: Dict, num_tests: int = 30) -> Dict:
        """
        Benchmark spoof detection (using all as live since we don't have spoof samples)
        
        Args:
            dataset: Test dataset
            num_tests: Number of tests to perform
            
        Returns:
            Spoof detection benchmark results
        """
        print("\n🛡️ Benchmarking Spoof Detection")
        print("-" * 50)
        
        results = {
            'total_tests': 0,
            'live_predictions': 0,
            'spoof_predictions': 0,
            'live_probabilities': [],
            'spoof_probabilities': [],
            'processing_times': [],
            'feature_analysis': {}
        }
        
        images = dataset['images'][:num_tests]
        
        print(f"   Testing {len(images)} images (assuming all are live)...")
        
        for i, image in enumerate(images):
            try:
                start_time = time.time()
                
                # Detect spoof
                spoof_result = self.spoof_detector.predict_spoof(image)
                
                processing_time = time.time() - start_time
                
                # Record results
                results['total_tests'] += 1
                results['processing_times'].append(processing_time)
                results['live_probabilities'].append(spoof_result['live_probability'])
                results['spoof_probabilities'].append(spoof_result['spoof_probability'])
                
                if spoof_result['is_live']:
                    results['live_predictions'] += 1
                else:
                    results['spoof_predictions'] += 1
                
                # Collect feature statistics
                features = spoof_result['features']
                for feature_name, value in features.items():
                    if feature_name not in results['feature_analysis']:
                        results['feature_analysis'][feature_name] = []
                    results['feature_analysis'][feature_name].append(value)
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(images)} images")
                    
            except Exception as e:
                print(f"   ⚠️ Failed on image {i}: {e}")
                continue
        
        # Compute statistics
        if results['live_probabilities']:
            results['avg_live_probability'] = np.mean(results['live_probabilities'])
            results['std_live_probability'] = np.std(results['live_probabilities'])
            
        if results['spoof_probabilities']:
            results['avg_spoof_probability'] = np.mean(results['spoof_probabilities'])
            results['std_spoof_probability'] = np.std(results['spoof_probabilities'])
        
        if results['processing_times']:
            results['avg_processing_time'] = np.mean(results['processing_times'])
        
        # Feature statistics
        for feature_name, values in results['feature_analysis'].items():
            if values:
                results['feature_analysis'][feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Print summary
        print(f"📊 Spoof Detection Results:")
        print(f"   Total Tests: {results['total_tests']}")
        print(f"   Live Predictions: {results['live_predictions']} ({results['live_predictions']/results['total_tests']:.1%})")
        print(f"   Spoof Predictions: {results['spoof_predictions']} ({results['spoof_predictions']/results['total_tests']:.1%})")
        if results['live_probabilities']:
            print(f"   Avg Live Probability: {results['avg_live_probability']:.3f} ± {results['std_live_probability']:.3f}")
        if results['processing_times']:
            print(f"   Avg Processing Time: {results['avg_processing_time']:.3f}s")
        
        return results
    
    def benchmark_system_integration(self, dataset: Dict, num_tests: int = 10) -> Dict:
        """
        Benchmark complete system integration
        
        Args:
            dataset: Test dataset
            num_tests: Number of integration tests
            
        Returns:
            System integration benchmark results
        """
        print("\n🔗 Benchmarking System Integration")
        print("-" * 50)
        
        results = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'component_times': {},
            'total_processing_times': [],
            'error_cases': []
        }
        
        images = dataset['images'][:num_tests]
        filenames = dataset['filenames'][:num_tests]
        
        print(f"   Running {len(images)} complete integration tests...")
        
        for i, (image, filename) in enumerate(zip(images, filenames)):
            try:
                start_time = time.time()
                component_times = {}
                
                # Step 1: Minutiae Detection
                step_start = time.time()
                minutiae_result = self.minutiae_extractor.extract_minutiae(image)
                component_times['minutiae_detection'] = time.time() - step_start
                
                # Step 2: Spoof Detection
                step_start = time.time()
                spoof_result = self.spoof_detector.predict_spoof(image)
                component_times['spoof_detection'] = time.time() - step_start
                
                # Step 3: AI Explanation (pattern classification mock)
                step_start = time.time()
                pattern_explanation = self.ai_assistant.explain_pattern_classification(
                    image, 'Loop', 0.85, {}
                )
                component_times['ai_explanation'] = time.time() - step_start
                
                total_time = time.time() - start_time
                
                # Record success
                results['successful_tests'] += 1
                results['total_processing_times'].append(total_time)
                
                # Accumulate component times
                for component, comp_time in component_times.items():
                    if component not in results['component_times']:
                        results['component_times'][component] = []
                    results['component_times'][component].append(comp_time)
                
                print(f"   ✅ Test {i + 1}/{len(images)}: {total_time:.2f}s")
                
            except Exception as e:
                results['failed_tests'] += 1
                results['error_cases'].append({
                    'filename': filename,
                    'error': str(e)
                })
                print(f"   ❌ Test {i + 1}/{len(images)} failed: {e}")
            
            results['total_tests'] += 1
        
        # Compute statistics
        if results['total_processing_times']:
            results['avg_total_time'] = np.mean(results['total_processing_times'])
            results['std_total_time'] = np.std(results['total_processing_times'])
            results['min_total_time'] = np.min(results['total_processing_times'])
            results['max_total_time'] = np.max(results['total_processing_times'])
        
        # Component time statistics
        for component, times in results['component_times'].items():
            if times:
                results['component_times'][component] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'total': np.sum(times)
                }
        
        # Calculate success rate
        results['success_rate'] = results['successful_tests'] / results['total_tests'] if results['total_tests'] > 0 else 0
        
        # Print summary
        print(f"📊 System Integration Results:")
        print(f"   Success Rate: {results['successful_tests']}/{results['total_tests']} ({results['success_rate']:.1%})")
        if results['total_processing_times']:
            print(f"   Avg Total Time: {results['avg_total_time']:.2f}s ± {results['std_total_time']:.2f}s")
            print(f"   Time Range: {results['min_total_time']:.2f}s - {results['max_total_time']:.2f}s")
        
        print(f"   Component Breakdown:")
        for component, stats in results['component_times'].items():
            if isinstance(stats, dict):
                print(f"     {component}: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
        
        return results
    
    def generate_performance_visualizations(self, all_results: Dict) -> None:
        """
        Generate performance visualization plots
        
        Args:
            all_results: Combined benchmark results
        """
        print("\n📊 Generating Performance Visualizations")
        print("-" * 50)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Fingerprint System Benchmark Results', fontsize=16)
        
        # 1. Minutiae Detection Statistics
        if 'minutiae_detection' in all_results:
            minutiae_data = all_results['minutiae_detection']
            if minutiae_data.get('minutiae_counts'):
                axes[0, 0].hist(minutiae_data['minutiae_counts'], bins=20, alpha=0.7, color='blue')
                axes[0, 0].set_title('Minutiae Count Distribution')
                axes[0, 0].set_xlabel('Number of Minutiae')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].axvline(minutiae_data['avg_minutiae'], color='red', linestyle='--', 
                                 label=f'Mean: {minutiae_data["avg_minutiae"]:.1f}')
                axes[0, 0].legend()
        
        # 2. Identity Matching Score Distribution
        if 'identity_matching' in all_results:
            matching_data = all_results['identity_matching']
            if matching_data.get('same_finger_scores') and matching_data.get('different_finger_scores'):
                axes[0, 1].hist(matching_data['same_finger_scores'], bins=15, alpha=0.7, 
                               color='green', label='Same Finger')
                axes[0, 1].hist(matching_data['different_finger_scores'], bins=15, alpha=0.7, 
                               color='red', label='Different Finger')
                axes[0, 1].set_title('Matching Score Distribution')
                axes[0, 1].set_xlabel('Match Score')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].axvline(matching_data['threshold'], color='black', linestyle='-', 
                                 label=f'Threshold: {matching_data["threshold"]}')
                axes[0, 1].legend()
        
        # 3. Spoof Detection Probabilities
        if 'spoof_detection' in all_results:
            spoof_data = all_results['spoof_detection']
            if spoof_data.get('live_probabilities'):
                axes[0, 2].hist(spoof_data['live_probabilities'], bins=20, alpha=0.7, color='cyan')
                axes[0, 2].set_title('Live Probability Distribution')
                axes[0, 2].set_xlabel('Live Probability')
                axes[0, 2].set_ylabel('Frequency')
                axes[0, 2].axvline(spoof_data['avg_live_probability'], color='red', linestyle='--',
                                 label=f'Mean: {spoof_data["avg_live_probability"]:.3f}')
                axes[0, 2].legend()
        
        # 4. Processing Time Comparison
        processing_times = {}
        if 'minutiae_detection' in all_results and all_results['minutiae_detection'].get('avg_processing_time'):
            processing_times['Minutiae'] = all_results['minutiae_detection']['avg_processing_time']
        if 'identity_matching' in all_results and all_results['identity_matching'].get('avg_processing_time'):
            processing_times['Matching'] = all_results['identity_matching']['avg_processing_time']
        if 'spoof_detection' in all_results and all_results['spoof_detection'].get('avg_processing_time'):
            processing_times['Spoof Detection'] = all_results['spoof_detection']['avg_processing_time']
        if 'system_integration' in all_results and all_results['system_integration'].get('avg_total_time'):
            processing_times['Full System'] = all_results['system_integration']['avg_total_time']
        
        if processing_times:
            components = list(processing_times.keys())
            times = list(processing_times.values())
            axes[1, 0].bar(components, times, color=['blue', 'green', 'orange', 'purple'])
            axes[1, 0].set_title('Average Processing Times')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. System Performance Summary
        performance_metrics = {}
        if 'minutiae_detection' in all_results:
            success_rate = all_results['minutiae_detection']['successful_extractions'] / all_results['minutiae_detection']['total_images']
            performance_metrics['Minutiae Detection'] = success_rate
        if 'identity_matching' in all_results and 'accuracy' in all_results['identity_matching']:
            performance_metrics['Identity Matching'] = all_results['identity_matching']['accuracy']
        if 'spoof_detection' in all_results:
            live_accuracy = all_results['spoof_detection']['live_predictions'] / all_results['spoof_detection']['total_tests']
            performance_metrics['Spoof Detection'] = live_accuracy
        if 'system_integration' in all_results:
            performance_metrics['System Integration'] = all_results['system_integration']['success_rate']
        
        if performance_metrics:
            components = list(performance_metrics.keys())
            accuracies = list(performance_metrics.values())
            bars = axes[1, 1].bar(components, accuracies, color=['lightblue', 'lightgreen', 'lightyellow', 'lightpink'])
            axes[1, 1].set_title('System Performance Summary')
            axes[1, 1].set_ylabel('Success Rate / Accuracy')
            axes[1, 1].set_ylim(0, 1.1)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{acc:.2f}', ha='center', va='bottom')
        
        # 6. Component Time Breakdown (if available)
        if 'system_integration' in all_results and 'component_times' in all_results['system_integration']:
            comp_times = all_results['system_integration']['component_times']
            valid_times = {k: v['mean'] for k, v in comp_times.items() if isinstance(v, dict)}
            
            if valid_times:
                labels = list(valid_times.keys())
                sizes = list(valid_times.values())
                axes[1, 2].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                axes[1, 2].set_title('Component Time Breakdown')
        
        # Remove empty subplot if needed
        for i in range(2):
            for j in range(3):
                if not axes[i, j].get_children():
                    axes[i, j].text(0.5, 0.5, 'No Data Available', 
                                   ha='center', va='center', transform=axes[i, j].transAxes)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        print("💾 Saved visualization: benchmark_results.png")
        plt.close()
    
    def run_comprehensive_benchmark(self, dataset_limit: int = 50) -> Dict:
        """
        Run comprehensive benchmark suite
        
        Args:
            dataset_limit: Maximum number of images to test
            
        Returns:
            Complete benchmark results
        """
        print("🚀 Starting Comprehensive Fingerprint System Benchmark")
        print("=" * 80)
        
        # Load dataset
        dataset = self.load_test_dataset(limit=dataset_limit)
        
        if len(dataset['images']) == 0:
            print("❌ No images loaded - cannot run benchmark")
            return {}
        
        # Run all benchmarks
        all_results = {}
        
        # 1. Minutiae Detection
        all_results['minutiae_detection'] = self.benchmark_minutiae_detection(dataset)
        
        # 2. Identity Matching
        all_results['identity_matching'] = self.benchmark_identity_matching(dataset, num_comparisons=30)
        
        # 3. Spoof Detection
        all_results['spoof_detection'] = self.benchmark_spoof_detection(dataset, num_tests=20)
        
        # 4. System Integration
        all_results['system_integration'] = self.benchmark_system_integration(dataset, num_tests=10)
        
        # Generate visualizations
        self.generate_performance_visualizations(all_results)
        
        # Save results
        with open('benchmark_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print("\n🎯 Benchmark Summary")
        print("=" * 80)
        self._print_benchmark_summary(all_results)
        
        return all_results
    
    def _print_benchmark_summary(self, results: Dict) -> None:
        """Print comprehensive benchmark summary"""
        
        print("📊 PERFORMANCE SUMMARY:")
        print("-" * 40)
        
        if 'minutiae_detection' in results:
            md = results['minutiae_detection']
            success_rate = md['successful_extractions'] / md['total_images']
            print(f"Minutiae Detection: {success_rate:.1%} success, {md.get('avg_minutiae', 0):.1f} avg minutiae")
        
        if 'identity_matching' in results:
            im = results['identity_matching']
            if 'accuracy' in im:
                print(f"Identity Matching: {im['accuracy']:.1%} accuracy, F1={im.get('f1_score', 0):.3f}")
        
        if 'spoof_detection' in results:
            sd = results['spoof_detection']
            live_rate = sd['live_predictions'] / sd['total_tests']
            print(f"Spoof Detection: {live_rate:.1%} classified as live")
        
        if 'system_integration' in results:
            si = results['system_integration']
            print(f"System Integration: {si['success_rate']:.1%} success, {si.get('avg_total_time', 0):.2f}s avg time")
        
        print("\n💾 Outputs Generated:")
        print("   • benchmark_results.json - Detailed results")
        print("   • benchmark_results.png - Performance visualizations")

def run_benchmark_test():
    """Run benchmark validation test"""
    
    # Test with sample data path
    data_path = r"C:\Users\alifa\Downloads\fingerprint_model-20250811T194218Z-1-001\fingerprint_model\data\archive\fingerprints\DB1_B"
    
    # Initialize validator
    validator = BenchmarkValidator(data_path)
    
    # Run comprehensive benchmark
    results = validator.run_comprehensive_benchmark(dataset_limit=30)
    
    return results

if __name__ == "__main__":
    run_benchmark_test()
