#!/usr/bin/env python3
"""
Comprehensive Benchmark Test Suite for Forensic Fingerprint System
================================================================

This script performs extensive benchmarking of all system components:
- Pattern Classification Performance
- Minutiae Detection Accuracy 
- Identity Matching Precision
- Spoof Detection Effectiveness
- Processing Speed Analysis
- Memory Usage Profiling
- Cross-Dataset Validation
- Scalability Testing

Author: Forensic AI Assistant
Date: August 19, 2025
"""

import os
import sys
import time
import json
import psutil
import threading
import statistics
from typing import Dict, List, Tuple, Any
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path

# Import all system modules
from complete_fingerprint_system import CompleteFingerprintSystem
from socofing_dataset_integration import SOCOFingDatasetManager
from minutiae_extraction import MinutiaeExtractor
from fast_fingerprint_matcher import FastFingerprintMatcher
from spoof_detection import SpoofDetector
from ai_forensic_assistant import ForensicAIAssistant

class ComprehensiveBenchmark:
    """Complete benchmarking suite for the forensic fingerprint system"""
    
    def __init__(self):
        print("🚀 Comprehensive Fingerprint System Benchmark")
        print("=" * 80)
        self.start_time = time.time()
        self.benchmark_id = f"benchmark_{int(time.time())}"
        
        # Initialize system components
        print("🔧 Initializing system components...")
        self.system = CompleteFingerprintSystem()
        self.dataset_manager = SOCOFingDatasetManager(r"C:\Users\alifa\Downloads\fingerprint_model-20250811T194218Z-1-001\fingerprint_model\data\archive\fingerprints")
        self.minutiae_extractor = MinutiaeExtractor()
        self.matcher = FastFingerprintMatcher()
        self.spoof_detector = SpoofDetector()
        self.forensic_ai = ForensicAIAssistant()
        
        # Benchmark results storage
        self.results = {
            'benchmark_id': self.benchmark_id,
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'dataset_info': {},
            'performance_metrics': {},
            'accuracy_metrics': {},
            'scalability_metrics': {},
            'memory_metrics': {},
            'timing_metrics': {},
            'error_analysis': {},
            'recommendations': []
        }
        
        print("✅ Benchmark suite initialized")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        return {
            'platform': sys.platform,
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'opencv_version': cv2.__version__
        }
    
    def _measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage during function execution"""
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        memory_after = process.memory_info().rss
        memory_used = memory_after - memory_before
        
        return result, execution_time, memory_used
    
    def benchmark_dataset_loading(self) -> Dict[str, Any]:
        """Benchmark dataset loading performance"""
        print("\n📂 Benchmarking Dataset Loading")
        print("-" * 50)
        
        results = {}
        
        # Benchmark dataset initialization
        start_time = time.time()
        self.dataset_manager._scan_all_datasets()
        init_time = time.time() - start_time
        
        dataset_info = self.dataset_manager.get_dataset_summary()
        results['initialization_time'] = init_time
        results['total_images'] = dataset_info['total_images']
        results['total_datasets'] = dataset_info['total_datasets']
        results['images_per_second'] = dataset_info['total_images'] / init_time if init_time > 0 else 0
        
        print(f"   📊 Total Images: {dataset_info['total_images']}")
        print(f"   ⏱️ Loading Time: {init_time:.2f}s")
        print(f"   🚀 Speed: {results['images_per_second']:.1f} images/sec")
        
        # Benchmark individual dataset loading
        dataset_timings = {}
        for dataset_name in ['DB1_B', 'DB2_B', 'data_ready', 'labeled_patterns']:
            start_time = time.time()
            images = self.dataset_manager.load_images_from_dataset(dataset_name, limit=10)
            load_time = time.time() - start_time
            
            dataset_timings[dataset_name] = {
                'load_time': load_time,
                'image_count': len(images),
                'avg_time_per_image': load_time / len(images) if images else 0
            }
            
            print(f"   📁 {dataset_name}: {len(images)} images in {load_time:.2f}s")
        
        results['dataset_timings'] = dataset_timings
        self.results['dataset_info'] = results
        return results
    
    def benchmark_minutiae_detection(self) -> Dict[str, Any]:
        """Benchmark minutiae detection across different image types"""
        print("\n🔍 Benchmarking Minutiae Detection")
        print("-" * 50)
        
        results = {
            'total_images_tested': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'average_minutiae_count': 0,
            'average_processing_time': 0,
            'average_memory_usage': 0,
            'dataset_performance': {},
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        all_minutiae_counts = []
        all_processing_times = []
        all_memory_usage = []
        
        # Test across different datasets
        for dataset_name in ['DB1_B', 'DB2_B', 'data_ready', 'labeled_patterns']:
            print(f"   📁 Testing {dataset_name}...")
            
            images = self.dataset_manager.load_images_from_dataset(dataset_name, limit=10)
            dataset_results = {
                'images_tested': len(images),
                'successful_extractions': 0,
                'minutiae_counts': [],
                'processing_times': [],
                'memory_usage': []
            }
            
            for img_info in images:
                img_path = "unknown"
                try:
                    img_path = img_info['metadata']['path']
                    img_data = img_info['image']
                    
                    # Measure minutiae extraction performance
                    extraction_result, proc_time, memory_used = self._measure_memory_usage(
                        self.minutiae_extractor.extract_minutiae, img_data
                    )
                    
                    if extraction_result and 'minutiae' in extraction_result:
                        minutiae_count = len(extraction_result['minutiae'])
                        dataset_results['successful_extractions'] += 1
                        dataset_results['minutiae_counts'].append(minutiae_count)
                        dataset_results['processing_times'].append(proc_time)
                        dataset_results['memory_usage'].append(memory_used)
                        
                        all_minutiae_counts.append(minutiae_count)
                        all_processing_times.append(proc_time)
                        all_memory_usage.append(memory_used)
                        
                        # Quality assessment
                        if minutiae_count > 100:
                            results['quality_distribution']['high'] += 1
                        elif minutiae_count > 50:
                            results['quality_distribution']['medium'] += 1
                        else:
                            results['quality_distribution']['low'] += 1
                    
                except Exception as e:
                    results['failed_extractions'] += 1
                    print(f"      ❌ Failed: {os.path.basename(img_path)} - {str(e)}")
            
            # Calculate dataset statistics
            if dataset_results['minutiae_counts']:
                dataset_results['avg_minutiae'] = statistics.mean(dataset_results['minutiae_counts'])
                dataset_results['avg_processing_time'] = statistics.mean(dataset_results['processing_times'])
                dataset_results['avg_memory_usage'] = statistics.mean(dataset_results['memory_usage'])
                dataset_results['success_rate'] = dataset_results['successful_extractions'] / dataset_results['images_tested']
                
                print(f"      ✅ Success Rate: {dataset_results['success_rate']:.1%}")
                print(f"      🔢 Avg Minutiae: {dataset_results['avg_minutiae']:.1f}")
                print(f"      ⏱️ Avg Time: {dataset_results['avg_processing_time']:.3f}s")
            
            results['dataset_performance'][dataset_name] = dataset_results
            results['total_images_tested'] += dataset_results['images_tested']
            results['successful_extractions'] += dataset_results['successful_extractions']
        
        # Calculate overall statistics
        if all_minutiae_counts:
            results['average_minutiae_count'] = statistics.mean(all_minutiae_counts)
            results['average_processing_time'] = statistics.mean(all_processing_times)
            results['average_memory_usage'] = statistics.mean(all_memory_usage)
            results['minutiae_std_dev'] = statistics.stdev(all_minutiae_counts) if len(all_minutiae_counts) > 1 else 0
            results['overall_success_rate'] = results['successful_extractions'] / results['total_images_tested']
        
        print(f"\n   📊 Overall Results:")
        print(f"      🎯 Success Rate: {results.get('overall_success_rate', 0):.1%}")
        print(f"      🔢 Avg Minutiae: {results.get('average_minutiae_count', 0):.1f}")
        print(f"      ⏱️ Avg Time: {results.get('average_processing_time', 0):.3f}s")
        
        self.results['performance_metrics']['minutiae_detection'] = results
        return results
    
    def benchmark_identity_matching(self) -> Dict[str, Any]:
        """Benchmark identity matching performance and accuracy"""
        print("\n🤝 Benchmarking Identity Matching")
        print("-" * 50)
        
        results = {
            'total_comparisons': 0,
            'successful_matches': 0,
            'average_matching_time': 0,
            'average_confidence_score': 0,
            'match_accuracy': 0,
            'false_positive_rate': 0,
            'false_negative_rate': 0,
            'dataset_performance': {}
        }
        
        all_matching_times = []
        all_confidence_scores = []
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        # Test matching within same subject (should match)
        for dataset_name in ['DB1_B', 'DB2_B']:
            print(f"   📁 Testing {dataset_name}...")
            
            images = self.dataset_manager.load_images_from_dataset(dataset_name, limit=8)
            dataset_results = {
                'comparisons': 0,
                'matching_times': [],
                'confidence_scores': [],
                'accuracy_metrics': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
            }
            
            # Test same-subject matching (positive cases)
            for i in range(0, min(len(images), 4), 2):
                if i + 1 < len(images):
                    img1_info = images[i]
                    img2_info = images[i + 1]
                    img1_path = img1_info['metadata']['path']
                    img1_data = img1_info['image']
                    img2_path = img2_info['metadata']['path']
                    img2_data = img2_info['image']
                    
                    # Extract subject IDs from filenames
                    subject1 = self._extract_subject_id(img1_path)
                    subject2 = self._extract_subject_id(img2_path)
                    is_same_subject = subject1 == subject2
                    
                    try:
                        start_time = time.time()
                        match_result = self.matcher.match_fingerprints(img1_data, img2_data)
                        matching_time = time.time() - start_time
                        
                        confidence = match_result.get('confidence', 0)
                        is_match = confidence > 0.3  # Threshold for match decision
                        
                        all_matching_times.append(matching_time)
                        all_confidence_scores.append(confidence)
                        dataset_results['matching_times'].append(matching_time)
                        dataset_results['confidence_scores'].append(confidence)
                        dataset_results['comparisons'] += 1
                        
                        # Update confusion matrix
                        if is_same_subject and is_match:
                            true_positives += 1
                            dataset_results['accuracy_metrics']['tp'] += 1
                        elif is_same_subject and not is_match:
                            false_negatives += 1
                            dataset_results['accuracy_metrics']['fn'] += 1
                        elif not is_same_subject and is_match:
                            false_positives += 1
                            dataset_results['accuracy_metrics']['fp'] += 1
                        else:
                            true_negatives += 1
                            dataset_results['accuracy_metrics']['tn'] += 1
                        
                    except Exception as e:
                        print(f"      ❌ Matching failed: {str(e)}")
            
            # Calculate dataset-specific metrics
            if dataset_results['matching_times']:
                dataset_results['avg_matching_time'] = statistics.mean(dataset_results['matching_times'])
                dataset_results['avg_confidence'] = statistics.mean(dataset_results['confidence_scores'])
                
                print(f"      ⏱️ Avg Matching Time: {dataset_results['avg_matching_time']:.3f}s")
                print(f"      🎯 Avg Confidence: {dataset_results['avg_confidence']:.3f}")
            
            results['dataset_performance'][dataset_name] = dataset_results
            results['total_comparisons'] += dataset_results['comparisons']
        
        # Calculate overall metrics
        if all_matching_times:
            results['average_matching_time'] = statistics.mean(all_matching_times)
            results['average_confidence_score'] = statistics.mean(all_confidence_scores)
        
        # Calculate accuracy metrics
        total_predictions = true_positives + false_positives + true_negatives + false_negatives
        if total_predictions > 0:
            results['match_accuracy'] = (true_positives + true_negatives) / total_predictions
            results['false_positive_rate'] = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
            results['false_negative_rate'] = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
            results['precision'] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            results['recall'] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        print(f"\n   📊 Overall Matching Results:")
        print(f"      🎯 Accuracy: {results.get('match_accuracy', 0):.1%}")
        print(f"      ⏱️ Avg Time: {results.get('average_matching_time', 0):.3f}s")
        print(f"      🔍 Precision: {results.get('precision', 0):.1%}")
        print(f"      📈 Recall: {results.get('recall', 0):.1%}")
        
        self.results['accuracy_metrics']['identity_matching'] = results
        return results
    
    def _extract_subject_id(self, image_path: str) -> str:
        """Extract subject ID from image filename"""
        filename = os.path.basename(image_path)
        # Extract subject ID (e.g., "101" from "101_1.tif")
        if '_' in filename:
            return filename.split('_')[0]
        return filename.split('.')[0]
    
    def benchmark_spoof_detection(self) -> Dict[str, Any]:
        """Benchmark spoof detection performance"""
        print("\n🛡️ Benchmarking Spoof Detection")
        print("-" * 50)
        
        results = {
            'total_images_tested': 0,
            'successful_detections': 0,
            'average_processing_time': 0,
            'average_confidence': 0,
            'live_detection_accuracy': 0,
            'dataset_performance': {}
        }
        
        all_processing_times = []
        all_confidences = []
        
        # Assume all SOCOFing images are live (real fingerprints)
        for dataset_name in ['DB1_B', 'DB2_B', 'data_ready', 'labeled_patterns']:
            print(f"   📁 Testing {dataset_name}...")
            
            images = self.dataset_manager.load_images_from_dataset(dataset_name, limit=5)
            dataset_results = {
                'images_tested': len(images),
                'successful_detections': 0,
                'processing_times': [],
                'confidences': [],
                'live_classifications': 0
            }
            
            for img_info in images:
                img_path = "unknown"
                try:
                    img_path = img_info['metadata']['path']
                    img_data = img_info['image']
                    
                    start_time = time.time()
                    spoof_result = self.spoof_detector.detect_spoof(img_data)
                    processing_time = time.time() - start_time
                    
                    if spoof_result:
                        confidence = spoof_result.get('live_probability', 0)
                        is_live = confidence > 0.5
                        
                        dataset_results['successful_detections'] += 1
                        dataset_results['processing_times'].append(processing_time)
                        dataset_results['confidences'].append(confidence)
                        
                        all_processing_times.append(processing_time)
                        all_confidences.append(confidence)
                        
                        if is_live:
                            dataset_results['live_classifications'] += 1
                
                except Exception as e:
                    print(f"      ❌ Spoof detection failed: {os.path.basename(img_path)} - {str(e)}")
            
            # Calculate dataset metrics
            if dataset_results['processing_times']:
                dataset_results['avg_processing_time'] = statistics.mean(dataset_results['processing_times'])
                dataset_results['avg_confidence'] = statistics.mean(dataset_results['confidences'])
                dataset_results['live_accuracy'] = dataset_results['live_classifications'] / dataset_results['successful_detections']
                
                print(f"      ⏱️ Avg Time: {dataset_results['avg_processing_time']:.3f}s")
                print(f"      🎯 Live Accuracy: {dataset_results['live_accuracy']:.1%}")
            
            results['dataset_performance'][dataset_name] = dataset_results
            results['total_images_tested'] += dataset_results['images_tested']
            results['successful_detections'] += dataset_results['successful_detections']
        
        # Calculate overall metrics
        if all_processing_times:
            results['average_processing_time'] = statistics.mean(all_processing_times)
            results['average_confidence'] = statistics.mean(all_confidences)
            results['success_rate'] = results['successful_detections'] / results['total_images_tested']
        
        print(f"\n   📊 Overall Spoof Detection Results:")
        print(f"      ✅ Success Rate: {results.get('success_rate', 0):.1%}")
        print(f"      ⏱️ Avg Time: {results.get('average_processing_time', 0):.3f}s")
        
        self.results['performance_metrics']['spoof_detection'] = results
        return results
    
    def benchmark_system_scalability(self) -> Dict[str, Any]:
        """Benchmark system scalability with increasing load"""
        print("\n📈 Benchmarking System Scalability")
        print("-" * 50)
        
        results = {
            'batch_sizes': [],
            'processing_times': [],
            'memory_usage': [],
            'throughput': [],
            'scalability_score': 0
        }
        
        batch_sizes = [1, 5, 10, 20, 50]
        
        for batch_size in batch_sizes:
            print(f"   📦 Testing batch size: {batch_size}")
            
            # Load test images
            test_images = []
            for dataset_name in ['DB1_B', 'data_ready']:
                images = self.dataset_manager.load_images_from_dataset(dataset_name, limit=batch_size//2)
                test_images.extend(images)
                if len(test_images) >= batch_size:
                    break
            
            test_images = test_images[:batch_size]
            
            # Measure batch processing performance
            start_time = time.time()
            memory_before = psutil.virtual_memory().used
            
            successful_processes = 0
            for img_info in test_images:
                try:
                    img_data = img_info['image']
                    # Simulate complete analysis
                    minutiae_result = self.minutiae_extractor.extract_minutiae(img_data)
                    if minutiae_result:
                        successful_processes += 1
                except:
                    pass
            
            processing_time = time.time() - start_time
            memory_after = psutil.virtual_memory().used
            memory_used = memory_after - memory_before
            
            throughput = successful_processes / processing_time if processing_time > 0 else 0
            
            results['batch_sizes'].append(batch_size)
            results['processing_times'].append(processing_time)
            results['memory_usage'].append(memory_used)
            results['throughput'].append(throughput)
            
            print(f"      ⏱️ Time: {processing_time:.2f}s")
            print(f"      🚀 Throughput: {throughput:.1f} images/sec")
            print(f"      💾 Memory: {memory_used / 1024 / 1024:.1f} MB")
        
        # Calculate scalability score (efficiency retention)
        if len(results['throughput']) >= 2:
            efficiency_retention = results['throughput'][-1] / results['throughput'][0]
            results['scalability_score'] = min(efficiency_retention, 1.0)
        
        print(f"\n   📊 Scalability Score: {results.get('scalability_score', 0):.2f}")
        
        self.results['scalability_metrics'] = results
        return results
    
    def benchmark_complete_system(self) -> Dict[str, Any]:
        """Benchmark the complete integrated system"""
        print("\n🎯 Benchmarking Complete System Integration")
        print("-" * 50)
        
        results = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'average_total_time': 0,
            'component_timings': {
                'pattern_classification': [],
                'minutiae_extraction': [],
                'identity_matching': [],
                'spoof_detection': [],
                'ai_explanation': []
            },
            'end_to_end_performance': {}
        }
        
        # Test complete workflow on different image types
        test_images = []
        for dataset_name in ['DB1_B', 'DB2_B', 'data_ready']:
            images = self.dataset_manager.load_images_from_dataset(dataset_name, limit=3)
            test_images.extend(images)
        
        print(f"   🧪 Testing {len(test_images)} complete workflows...")
        
        all_total_times = []
        
        for i, img_info in enumerate(test_images):
            img_path = img_info['metadata']['path']
            img_data = img_info['image']
            print(f"      📸 Processing {i+1}/{len(test_images)}: {os.path.basename(img_path)}")
            
            try:
                # Measure complete analysis time
                start_time = time.time()
                
                # Component timing tracking
                component_times = {}
                
                # 1. Pattern Classification (part of complete analysis)
                comp_start = time.time()
                analysis_result = self.system.analyze_fingerprint(img_data, f"test_analysis_{i}")
                component_times['pattern_classification'] = time.time() - comp_start
                
                # 2. Minutiae Extraction (if not already done)
                comp_start = time.time()
                minutiae_result = self.minutiae_extractor.extract_minutiae(img_data)
                component_times['minutiae_extraction'] = time.time() - comp_start
                
                # 3. Identity Matching (against first image)
                if i > 0:
                    comp_start = time.time()
                    match_result = self.matcher.match_fingerprints(img_data, test_images[0]['image'])
                    component_times['identity_matching'] = time.time() - comp_start
                else:
                    component_times['identity_matching'] = 0
                
                # 4. Spoof Detection
                comp_start = time.time()
                spoof_result = self.spoof_detector.detect_spoof(img_data)
                component_times['spoof_detection'] = time.time() - comp_start
                
                # 5. AI Explanation (if available)
                comp_start = time.time()
                try:
                    ai_result = self.forensic_ai.generate_forensic_report([{
                        'analysis': analysis_result,
                        'minutiae': minutiae_result,
                        'spoof': spoof_result
                    }])
                    component_times['ai_explanation'] = time.time() - comp_start
                except:
                    component_times['ai_explanation'] = 0
                
                total_time = time.time() - start_time
                all_total_times.append(total_time)
                
                # Store component timings
                for component, timing in component_times.items():
                    results['component_timings'][component].append(timing)
                
                results['successful_analyses'] += 1
                print(f"         ✅ Completed in {total_time:.2f}s")
                
            except Exception as e:
                print(f"         ❌ Failed: {str(e)}")
            
            results['total_analyses'] += 1
        
        # Calculate average timings
        if all_total_times:
            results['average_total_time'] = statistics.mean(all_total_times)
            results['success_rate'] = results['successful_analyses'] / results['total_analyses']
            
            # Calculate component averages
            for component in results['component_timings']:
                timings = results['component_timings'][component]
                if timings:
                    results['component_timings'][component] = {
                        'average': statistics.mean(timings),
                        'min': min(timings),
                        'max': max(timings),
                        'count': len(timings)
                    }
        
        print(f"\n   📊 Complete System Results:")
        print(f"      ✅ Success Rate: {results.get('success_rate', 0):.1%}")
        print(f"      ⏱️ Avg Total Time: {results.get('average_total_time', 0):.2f}s")
        
        # Print component breakdown
        for component, timing_data in results['component_timings'].items():
            if isinstance(timing_data, dict) and 'average' in timing_data:
                print(f"      🔧 {component.replace('_', ' ').title()}: {timing_data['average']:.3f}s")
        
        self.results['performance_metrics']['complete_system'] = results
        return results
    
    def generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on benchmark results"""
        recommendations = []
        
        # Analyze minutiae detection performance
        minutiae_metrics = self.results.get('performance_metrics', {}).get('minutiae_detection', {})
        if minutiae_metrics.get('overall_success_rate', 0) < 0.9:
            recommendations.append("Consider improving image preprocessing for better minutiae detection")
        
        if minutiae_metrics.get('average_processing_time', 0) > 1.0:
            recommendations.append("Optimize minutiae extraction algorithm for faster processing")
        
        # Analyze matching performance
        matching_metrics = self.results.get('accuracy_metrics', {}).get('identity_matching', {})
        if matching_metrics.get('match_accuracy', 0) < 0.8:
            recommendations.append("Fine-tune matching algorithms for better accuracy")
        
        if matching_metrics.get('false_positive_rate', 0) > 0.1:
            recommendations.append("Adjust matching thresholds to reduce false positives")
        
        # Analyze scalability
        scalability_metrics = self.results.get('scalability_metrics', {})
        if scalability_metrics.get('scalability_score', 0) < 0.7:
            recommendations.append("Implement parallel processing for better scalability")
        
        # General performance recommendations
        system_metrics = self.results.get('performance_metrics', {}).get('complete_system', {})
        if system_metrics.get('average_total_time', 0) > 5.0:
            recommendations.append("Consider system optimization for faster end-to-end processing")
        
        return recommendations
    
    def save_benchmark_results(self) -> str:
        """Save comprehensive benchmark results"""
        # Add final metrics
        self.results['total_benchmark_time'] = time.time() - self.start_time
        self.results['recommendations'] = self.generate_recommendations()
        
        # Save to JSON file
        filename = f"comprehensive_benchmark_results_{self.benchmark_id}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return filename
    
    def print_summary(self):
        """Print comprehensive benchmark summary"""
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Dataset Information
        dataset_info = self.results.get('dataset_info', {})
        print(f"\n📊 Dataset Performance:")
        print(f"   🖼️ Total Images: {dataset_info.get('total_images', 0)}")
        print(f"   ⏱️ Loading Speed: {dataset_info.get('images_per_second', 0):.1f} images/sec")
        
        # Minutiae Detection
        minutiae_metrics = self.results.get('performance_metrics', {}).get('minutiae_detection', {})
        print(f"\n🔍 Minutiae Detection:")
        print(f"   ✅ Success Rate: {minutiae_metrics.get('overall_success_rate', 0):.1%}")
        print(f"   🔢 Avg Minutiae: {minutiae_metrics.get('average_minutiae_count', 0):.1f}")
        print(f"   ⏱️ Avg Time: {minutiae_metrics.get('average_processing_time', 0):.3f}s")
        
        # Identity Matching
        matching_metrics = self.results.get('accuracy_metrics', {}).get('identity_matching', {})
        print(f"\n🤝 Identity Matching:")
        print(f"   🎯 Accuracy: {matching_metrics.get('match_accuracy', 0):.1%}")
        print(f"   🔍 Precision: {matching_metrics.get('precision', 0):.1%}")
        print(f"   📈 Recall: {matching_metrics.get('recall', 0):.1%}")
        print(f"   ⏱️ Avg Time: {matching_metrics.get('average_matching_time', 0):.3f}s")
        
        # Spoof Detection
        spoof_metrics = self.results.get('performance_metrics', {}).get('spoof_detection', {})
        print(f"\n🛡️ Spoof Detection:")
        print(f"   ✅ Success Rate: {spoof_metrics.get('success_rate', 0):.1%}")
        print(f"   ⏱️ Avg Time: {spoof_metrics.get('average_processing_time', 0):.3f}s")
        
        # Complete System
        system_metrics = self.results.get('performance_metrics', {}).get('complete_system', {})
        print(f"\n🎯 Complete System:")
        print(f"   ✅ Success Rate: {system_metrics.get('success_rate', 0):.1%}")
        print(f"   ⏱️ Avg Total Time: {system_metrics.get('average_total_time', 0):.2f}s")
        
        # Scalability
        scalability_metrics = self.results.get('scalability_metrics', {})
        print(f"\n📈 Scalability:")
        print(f"   📊 Scalability Score: {scalability_metrics.get('scalability_score', 0):.2f}")
        
        # Recommendations
        recommendations = self.results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Overall Assessment
        print(f"\n🏆 Overall Assessment:")
        total_time = self.results.get('total_benchmark_time', 0)
        print(f"   ⏱️ Total Benchmark Time: {total_time:.2f}s")
        
        # Determine system status
        excellent_criteria = all([
            minutiae_metrics.get('overall_success_rate', 0) > 0.9,
            matching_metrics.get('match_accuracy', 0) > 0.8,
            scalability_metrics.get('scalability_score', 0) > 0.7
        ])
        good_criteria = all([
            minutiae_metrics.get('overall_success_rate', 0) > 0.8,
            matching_metrics.get('match_accuracy', 0) > 0.7
        ])
        
        if excellent_criteria:
            status = "Excellent"
        elif good_criteria:
            status = "Good"
        else:
            status = "Needs Improvement"
            
        print(f"   🚀 System Status: {status}")
        
        print("=" * 80)
    
    def run_complete_benchmark(self) -> str:
        """Run the complete benchmark suite"""
        print("🚀 Starting Comprehensive Benchmark Suite...")
        print("This will test all system components with the SOCOFing dataset\n")
        
        try:
            # Run all benchmark components
            self.benchmark_dataset_loading()
            self.benchmark_minutiae_detection()
            self.benchmark_identity_matching()
            self.benchmark_spoof_detection()
            self.benchmark_system_scalability()
            self.benchmark_complete_system()
            
            # Save results and print summary
            results_file = self.save_benchmark_results()
            self.print_summary()
            
            print(f"\n💾 Detailed results saved to: {results_file}")
            return results_file
            
        except Exception as e:
            print(f"\n❌ Benchmark failed: {str(e)}")
            return ""

def main():
    """Main benchmark execution function"""
    benchmark = ComprehensiveBenchmark()
    results_file = benchmark.run_complete_benchmark()
    
    if results_file:
        print(f"\n✅ Comprehensive benchmark completed successfully!")
        print(f"📊 Results file: {results_file}")
    else:
        print("\n❌ Benchmark failed to complete")

if __name__ == "__main__":
    main()
