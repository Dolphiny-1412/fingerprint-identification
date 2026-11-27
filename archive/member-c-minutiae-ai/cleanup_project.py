#!/usr/bin/env python3
"""
Project Cleanup Plan - Remove Unnecessary Files
===============================================

This script identifies and removes unnecessary files while keeping only essential production components.

Essential Files to KEEP:
1. Core System Modules
2. Final Production Components  
3. Key Documentation
4. Final Test Results
5. Production Models/Data

Files to REMOVE:
1. Development/Debug files
2. Duplicate implementations
3. Intermediate test results
4. Outdated versions
5. Experimental files
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """Clean up project by removing unnecessary files"""
    
    # Essential files to KEEP (production-ready system)
    essential_files = {
        # Core System Modules (Production-Ready)
        'complete_fingerprint_system.py',
        'minutiae_extraction.py', 
        'fast_fingerprint_matcher.py',
        'spoof_detection.py',
        'ai_forensic_assistant.py',
        'socofing_dataset_integration.py',
        
        # Multi-task Classification (Best Implementation)
        'multitask_fingerprint_pipeline.py',
        'multitask_predict.py',
        
        # Final Benchmarking
        'final_comprehensive_benchmark.py',
        'FINAL_benchmark_results_final_benchmark_1755636407.json',
        
        # Key Documentation
        'PROJECT_COMPLETION_SUMMARY.md',
        'FINAL_MULTITASK_SUMMARY.md',
        'MULTITASK_DEPLOYMENT_GUIDE.md',
        
        # Production Requirements
        'requirements.txt',
        'multitask_requirements.txt',
        
        # Dataset Catalogs
        'socofing_dataset_catalog.json',
        'unified_socofing_dataset.json',
        'socofing_dataset_splits.json',
        
        # Final Demo Results
        'demo_single_analysis.json',
        'demo_comparison.json',
        'demo_system_report.txt',
        'forensic_report.txt',
        
        # Final Test Results
        'spoof_detection_test.json',
        'minutiae_extraction_test.json',
        'fast_matching_results.json',
        
        # AI Explanations
        'ai_explanations.json'
    }
    
    # Essential directories to KEEP
    essential_dirs = {
        'multitask_output',    # Final multi-task models
        'production_output',   # Production models
        'enhanced_output',     # Enhanced pipeline results
        'data',               # SOCOFing dataset
        'agent',              # AI assistant components
        'utils',              # Utility functions
        'classifier',         # Classification components
        'minutiae',           # Minutiae detection components
        'matcher',            # Matching components
        '__pycache__',        # Python cache (keep for performance)
        '.venv'               # Virtual environment
    }
    
    # Files to REMOVE (development/duplicate/outdated)
    files_to_remove = []
    dirs_to_remove = []
    
    # Get current directory
    current_dir = Path('.')
    
    print("🧹 Project Cleanup Analysis")
    print("=" * 50)
    
    # Analyze files
    for item in current_dir.iterdir():
        if item.is_file():
            filename = item.name
            
            # Skip essential files
            if filename in essential_files:
                print(f"✅ KEEP: {filename}")
                continue
                
            # Identify files to remove
            should_remove = False
            reason = ""
            
            # Development/Debug files
            if any(x in filename.lower() for x in ['debug', 'test_', 'quick_', 'demo_api', 'diagnostic']):
                should_remove = True
                reason = "Development/Debug file"
            
            # Duplicate implementations
            elif any(x in filename.lower() for x in ['enhanced_demo', 'integration_demo', 'hybrid', 'refactored']):
                should_remove = True
                reason = "Duplicate/Alternative implementation"
            
            # Intermediate results
            elif any(x in filename.lower() for x in ['comprehensive_benchmark_results', 'benchmark_results', 'socofing_batch', 'socofing_validation']):
                should_remove = True
                reason = "Intermediate test results"
            
            # Outdated versions
            elif any(x in filename.lower() for x in ['optimized_', 'professional_', 'plan_b', 'enhanced_fingerprint']):
                should_remove = True
                reason = "Outdated version"
            
            # Log files
            elif filename.endswith(('.log', '.txt')) and 'report' not in filename.lower():
                should_remove = True
                reason = "Log/temporary file"
            
            # Image files (except key visualizations)
            elif filename.endswith(('.png', '.jpg', '.jpeg', '.tif')) and 'final' not in filename.lower():
                should_remove = True
                reason = "Intermediate visualization"
            
            # Experimental files
            elif any(x in filename.lower() for x in ['landmark', 'batch_', 'auto_', 'clean_', 'extract_', 'merge_', 'populate_']):
                should_remove = True
                reason = "Experimental/utility script"
            
            # Deployment scripts (keeping only final)
            elif filename in ['deploy.sh', 'production_deploy.sh']:
                should_remove = True
                reason = "Deployment script (superseded)"
                
            if should_remove:
                files_to_remove.append((filename, reason))
                print(f"❌ REMOVE: {filename} ({reason})")
            else:
                print(f"✅ KEEP: {filename}")
        
        elif item.is_dir():
            dirname = item.name
            
            # Skip essential directories
            if dirname in essential_dirs:
                print(f"📁 KEEP DIR: {dirname}")
                continue
                
            # Identify directories to remove
            should_remove = False
            reason = ""
            
            # Visualization directories (keeping only essential outputs)
            if dirname.startswith('visualizations_'):
                should_remove = True
                reason = "Intermediate visualizations"
            
            # Landmark output (experimental)
            elif dirname == 'landmark_output':
                should_remove = True
                reason = "Experimental landmark detection"
            
            # Scripts directory (if exists)
            elif dirname == 'scripts':
                should_remove = True
                reason = "Utility scripts"
                
            if should_remove:
                dirs_to_remove.append((dirname, reason))
                print(f"📁 REMOVE DIR: {dirname} ({reason})")
            else:
                print(f"📁 KEEP DIR: {dirname}")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"   Files to remove: {len(files_to_remove)}")
    print(f"   Directories to remove: {len(dirs_to_remove)}")
    print(f"   Essential files kept: {len(essential_files)}")
    print(f"   Essential directories kept: {len(essential_dirs)}")
    
    return files_to_remove, dirs_to_remove

def execute_cleanup(files_to_remove, dirs_to_remove, confirm=True):
    """Execute the cleanup"""
    
    if confirm:
        print(f"\n⚠️ WARNING: This will permanently delete {len(files_to_remove)} files and {len(dirs_to_remove)} directories!")
        response = input("Type 'CONFIRM' to proceed with cleanup: ")
        if response != 'CONFIRM':
            print("❌ Cleanup cancelled")
            return False
    
    print(f"\n🧹 Executing cleanup...")
    
    # Remove files
    removed_files = 0
    for filename, reason in files_to_remove:
        try:
            os.remove(filename)
            print(f"🗑️ Removed file: {filename}")
            removed_files += 1
        except Exception as e:
            print(f"❌ Failed to remove {filename}: {e}")
    
    # Remove directories
    removed_dirs = 0
    for dirname, reason in dirs_to_remove:
        try:
            shutil.rmtree(dirname)
            print(f"🗑️ Removed directory: {dirname}")
            removed_dirs += 1
        except Exception as e:
            print(f"❌ Failed to remove {dirname}: {e}")
    
    print(f"\n✅ Cleanup completed!")
    print(f"   Files removed: {removed_files}/{len(files_to_remove)}")
    print(f"   Directories removed: {removed_dirs}/{len(dirs_to_remove)}")
    
    return True

if __name__ == "__main__":
    # Analyze what to clean
    files_to_remove, dirs_to_remove = cleanup_project()
    
    # Show what will be kept
    print(f"\n🎯 FINAL CLEAN PROJECT STRUCTURE:")
    print("=" * 50)
    print("📁 Essential Production Files:")
    print("   • complete_fingerprint_system.py - Main system")
    print("   • minutiae_extraction.py - Minutiae detection")
    print("   • fast_fingerprint_matcher.py - Identity matching")
    print("   • spoof_detection.py - Liveness detection") 
    print("   • ai_forensic_assistant.py - AI explanations")
    print("   • multitask_fingerprint_pipeline.py - Pattern classification")
    print("   • final_comprehensive_benchmark.py - System validation")
    print("   • PROJECT_COMPLETION_SUMMARY.md - Final documentation")
    print("\n📁 Essential Directories:")
    print("   • multitask_output/ - Trained models")
    print("   • production_output/ - Production assets")
    print("   • data/ - SOCOFing dataset")
    print("   • agent/ - AI assistant components")
    
    # Ask for confirmation
    print(f"\n🤔 Would you like to proceed with cleanup?")
    
    # Execute cleanup with confirmation
    execute_cleanup(files_to_remove, dirs_to_remove, confirm=True)
