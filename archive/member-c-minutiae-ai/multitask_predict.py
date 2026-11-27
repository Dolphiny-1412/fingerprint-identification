#!/usr/bin/env python3
"""
Multi-Task Fingerprint Prediction System
========================================

Production-ready prediction script for simultaneous:
1. Pattern Classification (Arch/Loop/Whorl)
2. Finger Identity Classification (10 finger positions)

Usage:
    python multitask_predict.py image.jpg
    python multitask_predict.py batch_folder/
"""

import os
import sys
import numpy as np
import cv2
import joblib
import json
from typing import Dict, List, Tuple, Union
import argparse
from pathlib import Path

# Import feature extraction components
from multitask_fingerprint_pipeline import (
    ProductionLandmarkDetector,
    EnhancedFeatureExtractor,
    FingerIdentityParser
)

class MultiTaskPredictor:
    """Production predictor for both pattern and finger classification"""
    
    def __init__(self, models_dir="multitask_output/models"):
        """
        Initialize the multi-task predictor
        
        Args:
            models_dir: Directory containing trained models and preprocessing components
        """
        self.models_dir = Path(models_dir)
        
        # Initialize feature extraction pipeline
        self.detector = ProductionLandmarkDetector()
        self.feature_extractor = EnhancedFeatureExtractor()
        self.finger_parser = FingerIdentityParser()
        
        # Load models and preprocessing components
        self._load_models()
        self._load_preprocessing_components()
        
        # Initialize feature columns (will be set from training)
        self.feature_columns = None
        
        print(f"✅ Multi-task predictor initialized")
        print(f"   Pattern model: {self.pattern_model is not None}")
        print(f"   Finger model: {self.finger_model is not None}")
    
    def _load_models(self):
        """Load trained models"""
        try:
            # Load pattern classification model
            pattern_model_path = self.models_dir / "xgboost_pattern_model.pkl"
            if pattern_model_path.exists():
                self.pattern_model = joblib.load(pattern_model_path)
                print(f"✅ Pattern model loaded from {pattern_model_path}")
            else:
                print(f"⚠️ Pattern model not found at {pattern_model_path}")
                self.pattern_model = None
            
            # Load finger classification model
            finger_model_path = self.models_dir / "xgboost_finger_model.pkl"
            if finger_model_path.exists():
                self.finger_model = joblib.load(finger_model_path)
                print(f"✅ Finger model loaded from {finger_model_path}")
            else:
                print(f"⚠️ Finger model not found at {finger_model_path}")
                self.finger_model = None
        
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.pattern_model = None
            self.finger_model = None
    
    def _load_preprocessing_components(self):
        """Load scalers and encoders"""
        try:
            # Load scalers
            pattern_scaler_path = self.models_dir / "pattern_scaler.pkl"
            finger_scaler_path = self.models_dir / "finger_scaler.pkl"
            
            if pattern_scaler_path.exists():
                self.pattern_scaler = joblib.load(pattern_scaler_path)
            else:
                print(f"⚠️ Pattern scaler not found")
                self.pattern_scaler = None
            
            if finger_scaler_path.exists():
                self.finger_scaler = joblib.load(finger_scaler_path)
            else:
                print(f"⚠️ Finger scaler not found")
                self.finger_scaler = None
            
            # Load encoders
            pattern_encoder_path = self.models_dir / "pattern_encoder.pkl"
            finger_encoder_path = self.models_dir / "finger_encoder.pkl"
            
            if pattern_encoder_path.exists():
                self.pattern_encoder = joblib.load(pattern_encoder_path)
            else:
                print(f"⚠️ Pattern encoder not found")
                self.pattern_encoder = None
            
            if finger_encoder_path.exists():
                self.finger_encoder = joblib.load(finger_encoder_path)
            else:
                print(f"⚠️ Finger encoder not found")
                self.finger_encoder = None
        
        except Exception as e:
            print(f"❌ Error loading preprocessing components: {e}")
            self.pattern_scaler = None
            self.finger_scaler = None
            self.pattern_encoder = None
            self.finger_encoder = None
    
    def extract_features(self, image_path: str) -> Dict:
        """Extract features from a single image"""
        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Detect landmarks
            landmarks = self.detector.detect_landmarks(image)
            
            # Extract features
            features = self.feature_extractor.extract_features(landmarks, image.shape)
            
            return features
        
        except Exception as e:
            print(f"❌ Error extracting features from {image_path}: {e}")
            return {}
    
    def predict_single_image(self, image_path: str, include_confidence: bool = True) -> Dict:
        """
        Predict both pattern and finger identity for a single image
        
        Args:
            image_path: Path to fingerprint image
            include_confidence: Whether to include confidence scores
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        filename = os.path.basename(image_path)
        result = {
            'filename': filename,
            'image_path': image_path,
            'pattern_prediction': None,
            'finger_prediction': None,
            'pattern_confidence': None,
            'finger_confidence': None,
            'pattern_probabilities': None,
            'finger_probabilities': None,
            'success': False,
            'error': None
        }
        
        try:
            # Extract features
            features = self.extract_features(image_path)
            
            if not features:
                result['error'] = "Failed to extract features"
                return result
            
            # Parse finger identity from filename
            directory = os.path.dirname(image_path)
            parsed_finger = self.finger_parser.parse_finger_identity(filename, directory)
            result['parsed_finger'] = parsed_finger
            
            # Get feature vector
            if self.feature_columns is None:
                # Use all numeric features
                self.feature_columns = [k for k, v in features.items() 
                                      if isinstance(v, (int, float, np.number))]
            
            feature_vector = [features.get(col, 0) for col in self.feature_columns]
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Pattern prediction
            if self.pattern_model is not None and self.pattern_scaler is not None:
                feature_scaled = self.pattern_scaler.transform(feature_array)
                pattern_pred_encoded = self.pattern_model.predict(feature_scaled)[0]
                result['pattern_prediction'] = self.pattern_encoder.inverse_transform([pattern_pred_encoded])[0]
                
                if include_confidence:
                    pattern_proba = self.pattern_model.predict_proba(feature_scaled)[0]
                    result['pattern_confidence'] = float(np.max(pattern_proba))
                    
                    # Map probabilities to class names
                    pattern_classes = self.pattern_encoder.classes_
                    result['pattern_probabilities'] = {
                        cls: float(prob) for cls, prob in zip(pattern_classes, pattern_proba)
                    }
            
            # Finger prediction
            if self.finger_model is not None and self.finger_scaler is not None:
                feature_scaled = self.finger_scaler.transform(feature_array)
                finger_pred_encoded = self.finger_model.predict(feature_scaled)[0]
                result['finger_prediction'] = self.finger_encoder.inverse_transform([finger_pred_encoded])[0]
                
                if include_confidence:
                    finger_proba = self.finger_model.predict_proba(feature_scaled)[0]
                    result['finger_confidence'] = float(np.max(finger_proba))
                    
                    # Map probabilities to class names
                    finger_classes = self.finger_encoder.classes_
                    result['finger_probabilities'] = {
                        cls: float(prob) for cls, prob in zip(finger_classes, finger_proba)
                    }
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Error predicting {filename}: {e}")
        
        return result
    
    def predict_batch(self, image_paths: List[str], include_confidence: bool = True) -> List[Dict]:
        """
        Predict pattern and finger identity for multiple images
        
        Args:
            image_paths: List of image file paths
            include_confidence: Whether to include confidence scores
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        print(f"🔄 Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            result = self.predict_single_image(image_path, include_confidence)
            results.append(result)
            
            # Progress reporting
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images")
        
        print(f"✅ Batch prediction complete!")
        return results
    
    def format_prediction_output(self, result: Dict, verbose: bool = True) -> str:
        """Format prediction result for display"""
        if not result['success']:
            return f"❌ Error processing {result['filename']}: {result['error']}"
        
        output = []
        output.append(f"📷 Image: {result['filename']}")
        
        # Pattern prediction
        if result['pattern_prediction']:
            pattern = result['pattern_prediction']
            if result['pattern_confidence']:
                confidence = result['pattern_confidence'] * 100
                output.append(f"🎯 Pattern: {pattern} ({confidence:.1f}% confidence)")
            else:
                output.append(f"🎯 Pattern: {pattern}")
        
        # Finger prediction
        if result['finger_prediction']:
            finger = result['finger_prediction']
            if result['finger_confidence']:
                confidence = result['finger_confidence'] * 100
                output.append(f"👆 Finger: {finger} ({confidence:.1f}% confidence)")
            else:
                output.append(f"👆 Finger: {finger}")
        
        # Parsed finger (from filename)
        if result.get('parsed_finger') and result['parsed_finger'] != 'Unknown':
            output.append(f"📝 Parsed from filename: {result['parsed_finger']}")
        
        # Detailed probabilities in verbose mode
        if verbose and result['pattern_probabilities']:
            output.append("🎯 Pattern probabilities:")
            for pattern_class, prob in result['pattern_probabilities'].items():
                output.append(f"   {pattern_class}: {prob*100:.1f}%")
        
        if verbose and result['finger_probabilities']:
            output.append("👆 Finger probabilities (top 5):")
            sorted_fingers = sorted(result['finger_probabilities'].items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
            for finger_class, prob in sorted_fingers:
                output.append(f"   {finger_class}: {prob*100:.1f}%")
        
        return "\n".join(output)

def find_image_files(path: str) -> List[str]:
    """Find all image files in a directory or return single file"""
    path_obj = Path(path)
    
    if path_obj.is_file():
        return [str(path_obj)]
    
    elif path_obj.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(path_obj.glob(f"**/*{ext}"))
            image_files.extend(path_obj.glob(f"**/*{ext.upper()}"))
        
        return [str(f) for f in sorted(image_files)]
    
    else:
        raise ValueError(f"Path does not exist: {path}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Multi-Task Fingerprint Classification - Pattern & Finger Identity"
    )
    parser.add_argument("input_path", help="Path to image file or directory")
    parser.add_argument("--models-dir", default="multitask_output/models", 
                       help="Directory containing trained models")
    parser.add_argument("--output", "-o", help="Output JSON file for batch results")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show detailed probabilities")
    parser.add_argument("--no-confidence", action="store_true",
                       help="Skip confidence score calculation (faster)")
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = MultiTaskPredictor(args.models_dir)
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return 1
    
    # Find image files
    try:
        image_paths = find_image_files(args.input_path)
        if not image_paths:
            print(f"❌ No image files found in: {args.input_path}")
            return 1
        
        print(f"📷 Found {len(image_paths)} image(s)")
    
    except Exception as e:
        print(f"❌ Error finding images: {e}")
        return 1
    
    # Make predictions
    include_confidence = not args.no_confidence
    
    if len(image_paths) == 1:
        # Single image prediction
        result = predictor.predict_single_image(image_paths[0], include_confidence)
        print("\n" + "="*60)
        print(predictor.format_prediction_output(result, args.verbose))
        print("="*60)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Result saved to {args.output}")
    
    else:
        # Batch prediction
        results = predictor.predict_batch(image_paths, include_confidence)
        
        # Summary statistics
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        print(f"\n📊 Batch Results Summary:")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        
        if successful > 0:
            # Show sample results
            print(f"\n📷 Sample Results:")
            print("="*60)
            for i, result in enumerate(results[:3]):  # Show first 3
                if result['success']:
                    print(predictor.format_prediction_output(result, verbose=False))
                    print("-"*40)
            
            if len(results) > 3:
                print(f"... and {len(results)-3} more results")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 All results saved to {args.output}")
        else:
            # Default output file
            output_file = f"prediction_results_{len(results)}_images.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 All results saved to {output_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
