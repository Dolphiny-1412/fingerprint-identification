#!/usr/bin/env python3
"""
SOCOFing Dataset Integration
===========================

Comprehensive integration of all available SOCOFing fingerprint datasets:
- DB1_B, DB2_B, DB3_B, DB4_B (Raw fingerprint databases)
- data_ready (Preprocessed training/validation/test sets)
- labeled_patterns (Pattern-labeled fingerprints)

Provides unified access to the complete 6000+ fingerprint dataset.
"""

import os
import glob
import cv2
import numpy as np
import json
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

class SOCOFingDatasetManager:
    """
    Comprehensive manager for SOCOFing fingerprint datasets
    """
    
    def __init__(self, base_path: str):
        """
        Initialize dataset manager
        
        Args:
            base_path: Base path to fingerprint datasets
        """
        self.base_path = Path(base_path)
        self.datasets = {}
        self.metadata = {}
        
        print("🗂️ SOCOFing Dataset Manager Initialized")
        print("=" * 60)
        
        # Define dataset paths
        self.dataset_paths = {
            'DB1_B': self.base_path / 'DB1_B',
            'DB2_B': self.base_path / 'DB2_B', 
            'DB3_B': self.base_path / 'DB3_B',
            'DB4_B': self.base_path / 'DB4_B',
            'data_ready': self.base_path / 'data_ready',
            'labeled_patterns': self.base_path / 'labeled_patterns'
        }
        
        # Initialize all datasets
        self._scan_all_datasets()
    
    def _scan_all_datasets(self):
        """Scan and catalog all available datasets"""
        print("🔍 Scanning all available datasets...")
        
        total_images = 0
        
        for dataset_name, dataset_path in self.dataset_paths.items():
            if dataset_path.exists():
                print(f"\n📁 Scanning {dataset_name}...")
                dataset_info = self._scan_dataset(dataset_path, dataset_name)
                self.datasets[dataset_name] = dataset_info
                total_images += dataset_info['total_images']
                
                print(f"   ✅ {dataset_info['total_images']} images found")
                print(f"   📊 {len(dataset_info['subjects'])} subjects")
                if dataset_info['patterns']:
                    print(f"   🏷️ {len(dataset_info['patterns'])} pattern types")
            else:
                print(f"   ⚠️ {dataset_name} not found at {dataset_path}")
        
        print(f"\n🎯 Total Dataset Summary:")
        print(f"   📚 Datasets: {len(self.datasets)}")
        print(f"   🖼️ Total Images: {total_images}")
        
        # Save dataset catalog
        self._save_dataset_catalog()
    
    def _scan_dataset(self, dataset_path: Path, dataset_name: str) -> Dict:
        """Scan individual dataset"""
        dataset_info = {
            'name': dataset_name,
            'path': str(dataset_path),
            'total_images': 0,
            'subjects': set(),
            'fingers': set(),
            'patterns': set(),
            'image_files': [],
            'subdirectories': [],
            'file_structure': {}
        }
        
        # Find all image files
        image_extensions = ['*.tif', '*.jpg', '*.png', '*.bmp', '*.jpeg']
        
        for ext in image_extensions:
            # Search recursively
            for img_path in dataset_path.rglob(ext):
                try:
                    # Parse filename for metadata
                    metadata = self._parse_filename(img_path.name, dataset_name)
                    
                    dataset_info['image_files'].append({
                        'path': str(img_path),
                        'filename': img_path.name,
                        'size': img_path.stat().st_size,
                        'subject': metadata.get('subject'),
                        'finger': metadata.get('finger'),
                        'pattern': metadata.get('pattern'),
                        'subdirectory': str(img_path.parent.relative_to(dataset_path))
                    })
                    
                    # Collect metadata
                    if metadata.get('subject'):
                        dataset_info['subjects'].add(metadata['subject'])
                    if metadata.get('finger'):
                        dataset_info['fingers'].add(metadata['finger'])
                    if metadata.get('pattern'):
                        dataset_info['patterns'].add(metadata['pattern'])
                    
                    dataset_info['total_images'] += 1
                    
                except Exception as e:
                    print(f"      ⚠️ Error processing {img_path}: {e}")
        
        # Find subdirectories
        for subdir in dataset_path.iterdir():
            if subdir.is_dir():
                dataset_info['subdirectories'].append(subdir.name)
        
        # Convert sets to lists for JSON serialization
        dataset_info['subjects'] = sorted(list(dataset_info['subjects']))
        dataset_info['fingers'] = sorted(list(dataset_info['fingers']))
        dataset_info['patterns'] = sorted(list(dataset_info['patterns']))
        
        return dataset_info
    
    def _parse_filename(self, filename: str, dataset_name: str) -> Dict:
        """Parse filename to extract metadata"""
        metadata = {'subject': None, 'finger': None, 'pattern': None}
        
        try:
            # Remove extension
            name_without_ext = os.path.splitext(filename)[0]
            
            if dataset_name in ['DB1_B', 'DB2_B', 'DB3_B', 'DB4_B']:
                # Format: subject_finger.ext (e.g., 101_1.tif)
                parts = name_without_ext.split('_')
                if len(parts) >= 2:
                    metadata['subject'] = parts[0]
                    metadata['finger'] = parts[1]
            
            elif dataset_name == 'labeled_patterns':
                # Check if filename contains pattern information
                filename_lower = filename.lower()
                if 'arch' in filename_lower:
                    metadata['pattern'] = 'Arch'
                elif 'loop' in filename_lower:
                    metadata['pattern'] = 'Loop'
                elif 'whorl' in filename_lower:
                    metadata['pattern'] = 'Whorl'
                
                # Also try to extract subject/finger
                parts = name_without_ext.split('_')
                if len(parts) >= 2 and parts[0].isdigit():
                    metadata['subject'] = parts[0]
                    metadata['finger'] = parts[1]
            
            elif dataset_name == 'data_ready':
                # Check subdirectory structure for patterns
                # This would need parent directory context
                pass
                
        except Exception as e:
            pass  # Return empty metadata on parse error
        
        return metadata
    
    def get_dataset_summary(self) -> Dict:
        """Get comprehensive dataset summary"""
        summary = {
            'total_datasets': len(self.datasets),
            'total_images': sum(ds['total_images'] for ds in self.datasets.values()),
            'datasets': {},
            'combined_statistics': {
                'subjects': set(),
                'fingers': set(),
                'patterns': set()
            }
        }
        
        for name, dataset in self.datasets.items():
            summary['datasets'][name] = {
                'images': dataset['total_images'],
                'subjects': len(dataset['subjects']),
                'fingers': len(dataset['fingers']),
                'patterns': len(dataset['patterns']),
                'subdirectories': len(dataset['subdirectories'])
            }
            
            # Combine statistics
            summary['combined_statistics']['subjects'].update(dataset['subjects'])
            summary['combined_statistics']['fingers'].update(dataset['fingers'])
            summary['combined_statistics']['patterns'].update(dataset['patterns'])
        
        # Convert sets to counts
        summary['combined_statistics'] = {
            'total_subjects': len(summary['combined_statistics']['subjects']),
            'total_fingers': len(summary['combined_statistics']['fingers']),
            'total_patterns': len(summary['combined_statistics']['patterns'])
        }
        
        return summary
    
    def load_images_from_dataset(self, dataset_name: str, limit: int = None, 
                                pattern_filter: str = None, subject_filter: str = None) -> List[Dict]:
        """
        Load images from specific dataset with filtering
        
        Args:
            dataset_name: Name of dataset to load from
            limit: Maximum number of images to load
            pattern_filter: Filter by pattern type (Arch, Loop, Whorl)
            subject_filter: Filter by subject ID
            
        Returns:
            List of loaded images with metadata
        """
        if dataset_name not in self.datasets:
            print(f"❌ Dataset {dataset_name} not found")
            return []
        
        dataset = self.datasets[dataset_name]
        loaded_images = []
        
        print(f"📂 Loading images from {dataset_name}...")
        
        for i, img_info in enumerate(dataset['image_files']):
            # Apply filters
            if pattern_filter and img_info.get('pattern') != pattern_filter:
                continue
            if subject_filter and img_info.get('subject') != subject_filter:
                continue
            
            try:
                # Load image
                image = cv2.imread(img_info['path'], cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    loaded_images.append({
                        'image': image,
                        'metadata': img_info,
                        'dataset_source': dataset_name
                    })
                    
                    # Check limit
                    if limit and len(loaded_images) >= limit:
                        break
                        
            except Exception as e:
                print(f"   ⚠️ Failed to load {img_info['filename']}: {e}")
        
        print(f"   ✅ Loaded {len(loaded_images)} images")
        return loaded_images
    
    def create_unified_dataset(self, output_path: str = "unified_socofing_dataset.json") -> Dict:
        """Create unified dataset index"""
        print("🔗 Creating unified SOCOFing dataset index...")
        
        unified_dataset = {
            'creation_date': datetime.now().isoformat(),
            'total_images': 0,
            'datasets': {},
            'unified_index': [],
            'statistics': self.get_dataset_summary()
        }
        
        image_id = 0
        
        for dataset_name, dataset in self.datasets.items():
            unified_dataset['datasets'][dataset_name] = {
                'path': dataset['path'],
                'image_count': dataset['total_images'],
                'subjects': dataset['subjects'],
                'patterns': dataset['patterns']
            }
            
            # Add all images to unified index
            for img_info in dataset['image_files']:
                unified_dataset['unified_index'].append({
                    'id': image_id,
                    'dataset_source': dataset_name,
                    'path': img_info['path'],
                    'filename': img_info['filename'],
                    'subject': img_info.get('subject'),
                    'finger': img_info.get('finger'),
                    'pattern': img_info.get('pattern'),
                    'subdirectory': img_info.get('subdirectory')
                })
                image_id += 1
                unified_dataset['total_images'] += 1
        
        # Save unified dataset
        with open(output_path, 'w') as f:
            json.dump(unified_dataset, f, indent=2, default=str)
        
        print(f"✅ Unified dataset created: {output_path}")
        print(f"   📊 Total indexed images: {unified_dataset['total_images']}")
        
        return unified_dataset
    
    def create_training_validation_split(self, train_ratio: float = 0.7, val_ratio: float = 0.2) -> Dict:
        """Create training/validation/test split from all datasets"""
        print("🎯 Creating training/validation/test split...")
        
        all_images = []
        
        # Collect all images with complete metadata
        for dataset_name, dataset in self.datasets.items():
            for img_info in dataset['image_files']:
                if img_info.get('subject') and img_info.get('finger'):
                    all_images.append({
                        'path': img_info['path'],
                        'subject': img_info['subject'],
                        'finger': img_info['finger'],
                        'pattern': img_info.get('pattern', 'Unknown'),
                        'dataset_source': dataset_name
                    })
        
        # Group by subject to ensure subject-level splitting
        subjects = {}
        for img in all_images:
            subject = img['subject']
            if subject not in subjects:
                subjects[subject] = []
            subjects[subject].append(img)
        
        # Split subjects
        subject_list = list(subjects.keys())
        np.random.shuffle(subject_list)
        
        n_subjects = len(subject_list)
        train_end = int(n_subjects * train_ratio)
        val_end = int(n_subjects * (train_ratio + val_ratio))
        
        train_subjects = subject_list[:train_end]
        val_subjects = subject_list[train_end:val_end]
        test_subjects = subject_list[val_end:]
        
        # Create splits
        splits = {
            'train': [],
            'validation': [],
            'test': []
        }
        
        for subject in train_subjects:
            splits['train'].extend(subjects[subject])
        for subject in val_subjects:
            splits['validation'].extend(subjects[subject])
        for subject in test_subjects:
            splits['test'].extend(subjects[subject])
        
        split_info = {
            'split_date': datetime.now().isoformat(),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': 1.0 - train_ratio - val_ratio,
            'subjects': {
                'train': train_subjects,
                'validation': val_subjects,
                'test': test_subjects
            },
            'image_counts': {
                'train': len(splits['train']),
                'validation': len(splits['validation']),
                'test': len(splits['test'])
            },
            'splits': splits
        }
        
        # Save split information
        with open('socofing_dataset_splits.json', 'w') as f:
            json.dump(split_info, f, indent=2, default=str)
        
        print(f"✅ Dataset split created:")
        print(f"   🏋️ Training: {len(splits['train'])} images ({len(train_subjects)} subjects)")
        print(f"   ✅ Validation: {len(splits['validation'])} images ({len(val_subjects)} subjects)")
        print(f"   🧪 Test: {len(splits['test'])} images ({len(test_subjects)} subjects)")
        
        return split_info
    
    def _save_dataset_catalog(self):
        """Save complete dataset catalog"""
        catalog = {
            'scan_date': datetime.now().isoformat(),
            'base_path': str(self.base_path),
            'datasets': self.datasets,
            'summary': self.get_dataset_summary()
        }
        
        with open('socofing_dataset_catalog.json', 'w') as f:
            json.dump(catalog, f, indent=2, default=str)
        
        print(f"💾 Dataset catalog saved: socofing_dataset_catalog.json")
    
    def generate_dataset_report(self) -> str:
        """Generate comprehensive dataset report"""
        summary = self.get_dataset_summary()
        
        report = []
        report.append("=" * 80)
        report.append("SOCOFing DATASET INTEGRATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        report.append("DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total Datasets: {summary['total_datasets']}")
        report.append(f"Total Images: {summary['total_images']}")
        report.append(f"Total Subjects: {summary['combined_statistics']['total_subjects']}")
        report.append(f"Total Finger Types: {summary['combined_statistics']['total_fingers']}")
        report.append(f"Total Pattern Types: {summary['combined_statistics']['total_patterns']}")
        report.append("")
        
        # Individual datasets
        report.append("INDIVIDUAL DATASETS")
        report.append("-" * 40)
        for name, stats in summary['datasets'].items():
            report.append(f"{name}:")
            report.append(f"  Images: {stats['images']}")
            report.append(f"  Subjects: {stats['subjects']}")
            report.append(f"  Patterns: {stats['patterns']}")
            report.append(f"  Subdirectories: {stats['subdirectories']}")
            report.append("")
        
        # Dataset paths
        report.append("DATASET LOCATIONS")
        report.append("-" * 40)
        for name, dataset in self.datasets.items():
            report.append(f"{name}: {dataset['path']}")
        report.append("")
        
        # Integration status
        report.append("INTEGRATION STATUS")
        report.append("-" * 40)
        report.append("✓ All available datasets scanned and cataloged")
        report.append("✓ Unified dataset index created")
        report.append("✓ Training/validation splits available")
        report.append("✓ Metadata extraction completed")
        report.append("✓ Ready for machine learning workflows")
        
        return "\n".join(report)

def integrate_socofing_datasets():
    """Main function to integrate all SOCOFing datasets"""
    
    # Dataset base path
    base_path = r"C:\Users\alifa\Downloads\fingerprint_model-20250811T194218Z-1-001\fingerprint_model\data\archive\fingerprints"
    
    # Initialize dataset manager
    manager = SOCOFingDatasetManager(base_path)
    
    # Create unified dataset
    unified_dataset = manager.create_unified_dataset()
    
    # Create training splits
    split_info = manager.create_training_validation_split()
    
    # Generate comprehensive report
    report = manager.generate_dataset_report()
    
    # Save report
    with open('socofing_integration_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("🎯 SOCOFing Dataset Integration Complete!")
    print("="*60)
    print(f"📊 {unified_dataset['total_images']} images integrated")
    print(f"📚 {len(unified_dataset['datasets'])} datasets linked")
    print(f"💾 Files generated:")
    print(f"   • socofing_dataset_catalog.json")
    print(f"   • unified_socofing_dataset.json")
    print(f"   • socofing_dataset_splits.json")
    print(f"   • socofing_integration_report.txt")
    
    # Show sample from each dataset
    print(f"\n📷 Sample from each dataset:")
    for dataset_name in manager.datasets.keys():
        sample_images = manager.load_images_from_dataset(dataset_name, limit=2)
        if sample_images:
            print(f"   {dataset_name}: {len(sample_images)} sample images loaded")
    
    return manager

if __name__ == "__main__":
    integrate_socofing_datasets()
