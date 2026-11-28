#!/usr/bin/env python3
"""
Explainable AI Forensic Assistant
=================================

Provides detailed explanations and forensic reasoning for:
1. Pattern classification decisions
2. Finger identity predictions
3. Minutiae detection results
4. Identity matching scores
5. Spoof detection outcomes

Generates comprehensive forensic reports with visual evidence.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime
import os

class ForensicAIAssistant:
    """
    Explainable AI assistant for forensic fingerprint analysis
    """
    
    def __init__(self):
        """Initialize the forensic AI assistant"""
        self.pattern_explanations = {
            'Arch': {
                'description': 'Ridges flow from one side to other with rise in center',
                'characteristics': ['No deltas', 'Continuous ridge flow', 'Tent or plain arch shape'],
                'rarity': 'Uncommon (5-10% of population)',
                'forensic_value': 'High uniqueness due to rarity'
            },
            'Loop': {
                'description': 'Ridges enter from one side, curve back, exit same side',
                'characteristics': ['One delta', 'Recurving ridges', 'Core formation'],
                'rarity': 'Most common (60-70% of population)',
                'forensic_value': 'Good identification due to delta and core features'
            },
            'Whorl': {
                'description': 'Circular or spiral ridge patterns with central core',
                'characteristics': ['Two deltas', 'Circular/spiral ridges', 'Central core'],
                'rarity': 'Moderately common (25-35% of population)',
                'forensic_value': 'Excellent for identification due to complex structure'
            }
        }
        
        self.finger_explanations = {
            'Thumb': 'Broader ridge patterns, often whorls or loops',
            'Index': 'Most commonly used finger, clear ridge patterns',
            'Middle': 'Largest finger surface, detailed ridge structure',
            'Ring': 'Similar to middle finger, good quality impressions',
            'Little': 'Smallest surface, can have unique characteristics'
        }
        
        self.spoof_indicators = {
            'ridge_continuity': 'Measures how well ridges connect - spoofed prints often have breaks',
            'ridge_valley_contrast': 'Contrast between ridges and valleys - low contrast suggests artificial material',
            'pore_density': 'Number of perspiration pores - real prints have natural pore distribution',
            'texture_complexity': 'Surface texture complexity - spoofed prints may lack natural texture variation'
        }
    
    def explain_pattern_classification(self, image: np.ndarray, prediction: str, 
                                     confidence: float, features: Dict) -> Dict:
        """
        Generate explanation for pattern classification
        
        Args:
            image: Input fingerprint image
            prediction: Predicted pattern class
            confidence: Prediction confidence
            features: Feature values used for classification
            
        Returns:
            Comprehensive explanation dictionary
        """
        explanation = {
            'analysis_type': 'Pattern Classification',
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'confidence': confidence,
            'explanation': {}
        }
        
        # Get pattern info
        pattern_info = self.pattern_explanations.get(prediction, {})
        
        explanation['explanation'] = {
            'primary_finding': f"The fingerprint shows characteristics of a {prediction} pattern",
            'description': pattern_info.get('description', 'Unknown pattern type'),
            'key_characteristics': pattern_info.get('characteristics', []),
            'population_frequency': pattern_info.get('rarity', 'Unknown'),
            'forensic_significance': pattern_info.get('forensic_value', 'Standard identification value'),
            'confidence_assessment': self._assess_confidence(confidence),
            'visual_evidence': self._identify_visual_markers(prediction),
            'alternative_interpretations': self._get_alternative_patterns(prediction, confidence)
        }
        
        return explanation
    
    def explain_finger_identification(self, image: np.ndarray, prediction: str, 
                                    confidence: float, features: Dict) -> Dict:
        """
        Generate explanation for finger position identification
        
        Args:
            image: Input fingerprint image
            prediction: Predicted finger position
            confidence: Prediction confidence
            features: Feature values used for classification
            
        Returns:
            Comprehensive explanation dictionary
        """
        explanation = {
            'analysis_type': 'Finger Position Identification',
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'confidence': confidence,
            'explanation': {}
        }
        
        finger_info = self.finger_explanations.get(prediction, 'Standard finger characteristics')
        
        explanation['explanation'] = {
            'primary_finding': f"Image characteristics consistent with {prediction} finger",
            'anatomical_reasoning': finger_info,
            'size_characteristics': self._analyze_size_features(image),
            'ridge_density_analysis': self._analyze_ridge_density(image),
            'pattern_distribution': self._analyze_pattern_distribution(image),
            'confidence_assessment': self._assess_confidence(confidence),
            'supporting_evidence': self._identify_finger_markers(prediction),
            'differential_diagnosis': self._get_alternative_fingers(prediction, confidence)
        }
        
        return explanation
    
    def explain_minutiae_detection(self, image: np.ndarray, minutiae: List, 
                                 quality_scores: List) -> Dict:
        """
        Generate explanation for minutiae detection
        
        Args:
            image: Input fingerprint image
            minutiae: List of detected minutiae points
            quality_scores: Quality scores for each minutiae
            
        Returns:
            Comprehensive explanation dictionary
        """
        explanation = {
            'analysis_type': 'Minutiae Detection',
            'timestamp': datetime.now().isoformat(),
            'total_minutiae': len(minutiae),
            'explanation': {}
        }
        
        # Analyze minutiae types
        ridge_endings = sum(1 for m in minutiae if m.get('type') == 'ridge_ending')
        bifurcations = sum(1 for m in minutiae if m.get('type') == 'bifurcation')
        
        # Quality analysis
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        high_quality_count = sum(1 for q in quality_scores if q > 0.7)
        
        explanation['explanation'] = {
            'detection_summary': f"Detected {len(minutiae)} minutiae points with {avg_quality:.1%} average quality",
            'minutiae_breakdown': {
                'ridge_endings': ridge_endings,
                'bifurcations': bifurcations,
                'total_points': len(minutiae)
            },
            'quality_assessment': {
                'average_quality': avg_quality,
                'high_quality_points': high_quality_count,
                'quality_distribution': self._analyze_quality_distribution(quality_scores)
            },
            'spatial_distribution': self._analyze_minutiae_distribution(minutiae, image.shape),
            'forensic_value': self._assess_minutiae_forensic_value(len(minutiae), avg_quality),
            'identification_potential': self._assess_identification_potential(len(minutiae), avg_quality),
            'methodology': {
                'algorithm': 'Crossing Number (CN) method with morphological preprocessing',
                'preprocessing': 'Gaussian blur, histogram equalization, binary threshold',
                'quality_filtering': 'Local gradient magnitude and ridge continuity assessment'
            }
        }
        
        return explanation
    
    def explain_identity_matching(self, template1: Dict, template2: Dict, 
                                match_score: float, is_match: bool) -> Dict:
        """
        Generate explanation for identity matching
        
        Args:
            template1: First fingerprint template
            template2: Second fingerprint template
            match_score: Matching score
            is_match: Whether prints match
            
        Returns:
            Comprehensive explanation dictionary
        """
        explanation = {
            'analysis_type': 'Identity Matching',
            'timestamp': datetime.now().isoformat(),
            'match_result': 'MATCH' if is_match else 'NO MATCH',
            'match_score': match_score,
            'explanation': {}
        }
        
        explanation['explanation'] = {
            'primary_conclusion': f"Fingerprints {'DO' if is_match else 'DO NOT'} originate from the same finger",
            'match_score_interpretation': self._interpret_match_score(match_score),
            'comparison_methodology': {
                'feature_extraction': 'ORB (Oriented FAST and Rotated BRIEF) keypoints',
                'matching_algorithm': 'Feature descriptor matching with geometric consistency',
                'scoring_method': 'Combined feature similarity and image statistics'
            },
            'statistical_analysis': {
                'feature_count_template1': template1.get('feature_count', 0),
                'feature_count_template2': template2.get('feature_count', 0),
                'matched_features': template1.get('matched_features', 0),
                'match_ratio': template1.get('match_ratio', 0.0)
            },
            'quality_factors': self._assess_matching_quality(template1, template2),
            'confidence_level': self._assess_matching_confidence(match_score, is_match),
            'forensic_interpretation': self._provide_forensic_interpretation(match_score, is_match),
            'potential_limitations': self._identify_matching_limitations(template1, template2)
        }
        
        return explanation
    
    def explain_spoof_detection(self, image: np.ndarray, result: Dict) -> Dict:
        """
        Generate explanation for spoof detection
        
        Args:
            image: Input fingerprint image
            result: Spoof detection result
            
        Returns:
            Comprehensive explanation dictionary
        """
        explanation = {
            'analysis_type': 'Presentation Attack Detection (PAD)',
            'timestamp': datetime.now().isoformat(),
            'determination': result['prediction'],
            'live_probability': result['live_probability'],
            'spoof_probability': result['spoof_probability'],
            'explanation': {}
        }
        
        features = result['features']
        
        explanation['explanation'] = {
            'primary_assessment': f"Image analysis indicates a {result['prediction'].upper()} fingerprint",
            'liveness_indicators': self._analyze_liveness_indicators(features),
            'spoof_risk_factors': self._analyze_spoof_indicators(features),
            'feature_analysis': {
                'ridge_characteristics': {
                    'continuity': features.get('ridge_ridge_continuity', 0),
                    'contrast': features.get('ridge_ridge_valley_contrast', 0),
                    'interpretation': self._interpret_ridge_features(features)
                },
                'pore_analysis': {
                    'density': features.get('pore_pore_density', 0),
                    'distribution': features.get('pore_pore_spacing_mean', 0),
                    'interpretation': self._interpret_pore_features(features)
                },
                'texture_analysis': {
                    'complexity': features.get('texture_lbp_entropy', 0),
                    'variance': features.get('texture_stat_variance', 0),
                    'interpretation': self._interpret_texture_features(features)
                }
            },
            'confidence_assessment': self._assess_spoof_confidence(result),
            'methodology': {
                'approach': 'Multi-modal feature analysis',
                'features': 'Ridge characteristics, pore distribution, texture analysis',
                'decision_method': 'Rule-based classification with probabilistic scoring'
            },
            'recommendations': self._provide_spoof_recommendations(result)
        }
        
        return explanation
    
    def generate_forensic_report(self, analyses: List[Dict], image_path: str = None) -> str:
        """
        Generate comprehensive forensic report
        
        Args:
            analyses: List of analysis results
            image_path: Path to analyzed image
            
        Returns:
            Formatted forensic report
        """
        report = []
        report.append("=" * 80)
        report.append("FORENSIC FINGERPRINT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Header
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Image Source: {image_path if image_path else 'Unknown'}")
        report.append(f"Analyses Performed: {len(analyses)}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(self._generate_executive_summary(analyses))
        report.append("")
        
        # Detailed Analyses
        for i, analysis in enumerate(analyses, 1):
            report.append(f"{i}. {analysis['analysis_type'].upper()}")
            report.append("-" * 50)
            
            if analysis['analysis_type'] == 'Pattern Classification':
                report.extend(self._format_pattern_analysis(analysis))
            elif analysis['analysis_type'] == 'Finger Position Identification':
                report.extend(self._format_finger_analysis(analysis))
            elif analysis['analysis_type'] == 'Minutiae Detection':
                report.extend(self._format_minutiae_analysis(analysis))
            elif analysis['analysis_type'] == 'Identity Matching':
                report.extend(self._format_matching_analysis(analysis))
            elif analysis['analysis_type'] == 'Presentation Attack Detection (PAD)':
                report.extend(self._format_spoof_analysis(analysis))
            
            report.append("")
        
        # Conclusions
        report.append("FORENSIC CONCLUSIONS")
        report.append("-" * 40)
        report.extend(self._generate_conclusions(analyses))
        report.append("")
        
        # Methodology
        report.append("METHODOLOGY AND LIMITATIONS")
        report.append("-" * 40)
        report.extend(self._generate_methodology_section())
        
        return "\n".join(report)
    
    # Helper methods for confidence assessment
    def _assess_confidence(self, confidence: float) -> str:
        """Assess confidence level"""
        if confidence >= 0.9:
            return "Very High Confidence - Strong evidence supports classification"
        elif confidence >= 0.8:
            return "High Confidence - Good evidence supports classification"
        elif confidence >= 0.7:
            return "Moderate Confidence - Reasonable evidence supports classification"
        elif confidence >= 0.6:
            return "Low Confidence - Limited evidence, consider additional analysis"
        else:
            return "Very Low Confidence - Classification uncertain, additional verification needed"
    
    def _identify_visual_markers(self, pattern: str) -> List[str]:
        """Identify visual markers for pattern"""
        markers = {
            'Arch': ['Continuous ridge flow', 'No deltas visible', 'Central ridge elevation'],
            'Loop': ['Delta point visible', 'Recurving ridges', 'Core formation'],
            'Whorl': ['Two delta points', 'Circular ridge pattern', 'Central whorl core']
        }
        return markers.get(pattern, [])
    
    def _get_alternative_patterns(self, prediction: str, confidence: float) -> List[str]:
        """Get alternative pattern interpretations"""
        if confidence < 0.8:
            all_patterns = ['Arch', 'Loop', 'Whorl']
            return [p for p in all_patterns if p != prediction]
        return []
    
    def _analyze_size_features(self, image: np.ndarray) -> str:
        """Analyze size characteristics"""
        h, w = image.shape
        area = h * w
        
        if area > 400 * 400:
            return "Large impression consistent with thumb or major finger"
        elif area > 200 * 200:
            return "Medium-sized impression typical of index/middle/ring fingers"
        else:
            return "Small impression consistent with little finger"
    
    def _analyze_ridge_density(self, image: np.ndarray) -> str:
        """Analyze ridge density characteristics"""
        # Simple ridge density estimation
        binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        ridge_density = np.sum(binary == 0) / (image.shape[0] * image.shape[1])
        
        if ridge_density > 0.6:
            return "High ridge density typical of thumb or stressed impressions"
        elif ridge_density > 0.4:
            return "Normal ridge density consistent with quality impressions"
        else:
            return "Low ridge density possibly indicating light pressure or finger condition"
    
    def _analyze_pattern_distribution(self, image: np.ndarray) -> str:
        """Analyze pattern distribution"""
        return "Pattern distribution analysis requires additional processing"
    
    def _identify_finger_markers(self, finger: str) -> List[str]:
        """Identify markers for finger position"""
        markers = {
            'Thumb': ['Broader ridge structure', 'Often whorl patterns', 'Larger surface area'],
            'Index': ['Clear ridge definition', 'Common loop patterns', 'Good quality impressions'],
            'Middle': ['Largest finger surface', 'Detailed ridge structure', 'Various patterns'],
            'Ring': ['Similar to middle finger', 'Good impression quality', 'Consistent patterns'],
            'Little': ['Smallest surface area', 'Unique characteristics', 'Sometimes distorted']
        }
        return markers.get(finger, [])
    
    def _get_alternative_fingers(self, prediction: str, confidence: float) -> List[str]:
        """Get alternative finger interpretations"""
        if confidence < 0.7:
            return ['Consider adjacent finger positions', 'Verify with additional impressions']
        return []
    
    def _analyze_quality_distribution(self, quality_scores: List) -> str:
        """Analyze quality score distribution"""
        if not quality_scores:
            return "No quality scores available"
        
        high_q = sum(1 for q in quality_scores if q > 0.8)
        medium_q = sum(1 for q in quality_scores if 0.5 < q <= 0.8)
        low_q = sum(1 for q in quality_scores if q <= 0.5)
        
        return f"High: {high_q}, Medium: {medium_q}, Low: {low_q}"
    
    def _analyze_minutiae_distribution(self, minutiae: List, image_shape: Tuple) -> str:
        """Analyze spatial distribution of minutiae"""
        if not minutiae:
            return "No minutiae for distribution analysis"
        
        # Simple spatial analysis
        positions = [(m.get('x', 0), m.get('y', 0)) for m in minutiae]
        
        return f"Minutiae distributed across impression with {len(positions)} valid points"
    
    def _assess_minutiae_forensic_value(self, count: int, quality: float) -> str:
        """Assess forensic value of minutiae"""
        if count >= 20 and quality > 0.7:
            return "Excellent forensic value - sufficient for positive identification"
        elif count >= 12 and quality > 0.6:
            return "Good forensic value - suitable for identification with additional evidence"
        elif count >= 8 and quality > 0.5:
            return "Moderate forensic value - may support identification"
        else:
            return "Limited forensic value - insufficient for reliable identification"
    
    def _assess_identification_potential(self, count: int, quality: float) -> str:
        """Assess identification potential"""
        if count >= 15 and quality > 0.8:
            return "High identification potential - meets forensic standards"
        elif count >= 10 and quality > 0.6:
            return "Moderate identification potential - additional verification recommended"
        else:
            return "Low identification potential - additional evidence required"
    
    def _interpret_match_score(self, score: float) -> str:
        """Interpret matching score"""
        if score >= 0.8:
            return "Very strong match - highly consistent features"
        elif score >= 0.6:
            return "Strong match - good feature consistency"
        elif score >= 0.4:
            return "Moderate match - some consistent features"
        elif score >= 0.2:
            return "Weak match - limited feature consistency"
        else:
            return "No significant match - minimal feature consistency"
    
    def _assess_matching_quality(self, template1: Dict, template2: Dict) -> Dict:
        """Assess quality factors for matching"""
        return {
            'template1_quality': template1.get('quality', 'Unknown'),
            'template2_quality': template2.get('quality', 'Unknown'),
            'feature_count_balance': abs(template1.get('feature_count', 0) - template2.get('feature_count', 0)) < 50
        }
    
    def _assess_matching_confidence(self, score: float, is_match: bool) -> str:
        """Assess confidence in matching result"""
        if is_match and score > 0.7:
            return "High confidence in positive match"
        elif not is_match and score < 0.3:
            return "High confidence in negative match"
        elif 0.4 <= score <= 0.6:
            return "Moderate confidence - borderline result"
        else:
            return "Standard confidence level"
    
    def _provide_forensic_interpretation(self, score: float, is_match: bool) -> str:
        """Provide forensic interpretation"""
        if is_match:
            return "Features consistent with same source fingerprint within reasonable scientific certainty"
        else:
            return "Features not consistent with same source fingerprint - likely different individuals"
    
    def _identify_matching_limitations(self, template1: Dict, template2: Dict) -> List[str]:
        """Identify potential limitations"""
        limitations = []
        
        if template1.get('feature_count', 0) < 50:
            limitations.append("Limited features in first template")
        if template2.get('feature_count', 0) < 50:
            limitations.append("Limited features in second template")
        
        return limitations
    
    def _analyze_liveness_indicators(self, features: Dict) -> Dict:
        """Analyze indicators of live fingerprint"""
        indicators = {}
        
        ridge_continuity = features.get('ridge_ridge_continuity', 0)
        if ridge_continuity > 0.8:
            indicators['ridge_quality'] = "High ridge continuity suggests live impression"
        
        pore_density = features.get('pore_pore_density', 0)
        if pore_density > 1.0:
            indicators['pore_presence'] = "Natural pore distribution detected"
        
        return indicators
    
    def _analyze_spoof_indicators(self, features: Dict) -> Dict:
        """Analyze indicators of spoofed fingerprint"""
        indicators = {}
        
        ridge_continuity = features.get('ridge_ridge_continuity', 1.0)
        if ridge_continuity < 0.5:
            indicators['ridge_breaks'] = "Unusual ridge discontinuity detected"
        
        contrast = features.get('ridge_ridge_valley_contrast', 1.0)
        if contrast < 0.3:
            indicators['low_contrast'] = "Abnormally low ridge-valley contrast"
        
        return indicators
    
    def _interpret_ridge_features(self, features: Dict) -> str:
        """Interpret ridge feature analysis"""
        continuity = features.get('ridge_ridge_continuity', 0)
        contrast = features.get('ridge_ridge_valley_contrast', 0)
        
        if continuity > 0.8 and contrast > 0.5:
            return "Ridge features consistent with natural fingerprint"
        elif continuity < 0.5 or contrast < 0.3:
            return "Ridge features show anomalies suggesting possible artificial origin"
        else:
            return "Ridge features show mixed characteristics requiring further analysis"
    
    def _interpret_pore_features(self, features: Dict) -> str:
        """Interpret pore feature analysis"""
        density = features.get('pore_pore_density', 0)
        
        if 1.0 <= density <= 3.0:
            return "Pore distribution within normal range for live fingerprints"
        elif density < 0.5:
            return "Very low pore density may indicate artificial material"
        elif density > 5.0:
            return "Unusually high pore count may indicate image artifacts"
        else:
            return "Pore characteristics require additional evaluation"
    
    def _interpret_texture_features(self, features: Dict) -> str:
        """Interpret texture feature analysis"""
        entropy = features.get('texture_lbp_entropy', 0)
        variance = features.get('texture_stat_variance', 0)
        
        if entropy > 2.5 and variance > 500:
            return "Texture complexity consistent with natural skin surface"
        elif entropy < 1.5 or variance < 100:
            return "Low texture complexity suggests possible artificial material"
        else:
            return "Texture characteristics show intermediate complexity"
    
    def _assess_spoof_confidence(self, result: Dict) -> str:
        """Assess confidence in spoof detection"""
        confidence = result['confidence']
        
        if confidence > 0.9:
            return "Very high confidence in spoof detection result"
        elif confidence > 0.8:
            return "High confidence in spoof detection result"
        elif confidence > 0.7:
            return "Moderate confidence in spoof detection result"
        else:
            return "Low confidence - additional verification recommended"
    
    def _provide_spoof_recommendations(self, result: Dict) -> List[str]:
        """Provide recommendations based on spoof detection"""
        recommendations = []
        
        if result['prediction'] == 'Spoof':
            recommendations.append("Reject impression - presentation attack detected")
            recommendations.append("Request new live impression under controlled conditions")
        else:
            recommendations.append("Accept impression for further processing")
            if result['confidence'] < 0.8:
                recommendations.append("Consider additional verification if high security required")
        
        return recommendations
    
    # Report formatting methods
    def _generate_executive_summary(self, analyses: List[Dict]) -> str:
        """Generate executive summary"""
        summary_points = []
        
        for analysis in analyses:
            if analysis['analysis_type'] == 'Pattern Classification':
                summary_points.append(f"Pattern: {analysis['prediction']} ({analysis['confidence']:.1%} confidence)")
            elif analysis['analysis_type'] == 'Finger Position Identification':
                summary_points.append(f"Finger: {analysis['prediction']} ({analysis['confidence']:.1%} confidence)")
            elif analysis['analysis_type'] == 'Minutiae Detection':
                summary_points.append(f"Minutiae: {analysis['total_minutiae']} points detected")
            elif analysis['analysis_type'] == 'Identity Matching':
                summary_points.append(f"Matching: {analysis['match_result']} (score: {analysis['match_score']:.3f})")
            elif analysis['analysis_type'] == 'Presentation Attack Detection (PAD)':
                summary_points.append(f"Liveness: {analysis['determination']} ({analysis['live_probability']:.1%} live probability)")
        
        return "\n".join(summary_points)
    
    def _format_pattern_analysis(self, analysis: Dict) -> List[str]:
        """Format pattern analysis section"""
        lines = []
        exp = analysis['explanation']
        
        lines.append(f"Prediction: {analysis['prediction']}")
        lines.append(f"Confidence: {analysis['confidence']:.1%}")
        lines.append(f"Description: {exp['description']}")
        lines.append("Key Characteristics:")
        for char in exp['key_characteristics']:
            lines.append(f"  • {char}")
        lines.append(f"Population Frequency: {exp['population_frequency']}")
        lines.append(f"Confidence Assessment: {exp['confidence_assessment']}")
        
        return lines
    
    def _format_finger_analysis(self, analysis: Dict) -> List[str]:
        """Format finger analysis section"""
        lines = []
        exp = analysis['explanation']
        
        lines.append(f"Prediction: {analysis['prediction']}")
        lines.append(f"Confidence: {analysis['confidence']:.1%}")
        lines.append(f"Primary Finding: {exp['primary_finding']}")
        lines.append(f"Anatomical Reasoning: {exp['anatomical_reasoning']}")
        lines.append(f"Size Characteristics: {exp['size_characteristics']}")
        lines.append(f"Confidence Assessment: {exp['confidence_assessment']}")
        
        return lines
    
    def _format_minutiae_analysis(self, analysis: Dict) -> List[str]:
        """Format minutiae analysis section"""
        lines = []
        exp = analysis['explanation']
        
        lines.append(f"Total Minutiae: {analysis['total_minutiae']}")
        lines.append(f"Detection Summary: {exp['detection_summary']}")
        lines.append("Minutiae Breakdown:")
        breakdown = exp['minutiae_breakdown']
        lines.append(f"  • Ridge Endings: {breakdown['ridge_endings']}")
        lines.append(f"  • Bifurcations: {breakdown['bifurcations']}")
        lines.append(f"Forensic Value: {exp['forensic_value']}")
        lines.append(f"Identification Potential: {exp['identification_potential']}")
        
        return lines
    
    def _format_matching_analysis(self, analysis: Dict) -> List[str]:
        """Format matching analysis section"""
        lines = []
        exp = analysis['explanation']
        
        lines.append(f"Result: {analysis['match_result']}")
        lines.append(f"Match Score: {analysis['match_score']:.3f}")
        lines.append(f"Primary Conclusion: {exp['primary_conclusion']}")
        lines.append(f"Score Interpretation: {exp['match_score_interpretation']}")
        lines.append(f"Confidence Level: {exp['confidence_level']}")
        lines.append(f"Forensic Interpretation: {exp['forensic_interpretation']}")
        
        return lines
    
    def _format_spoof_analysis(self, analysis: Dict) -> List[str]:
        """Format spoof analysis section"""
        lines = []
        exp = analysis['explanation']
        
        lines.append(f"Determination: {analysis['determination']}")
        lines.append(f"Live Probability: {analysis['live_probability']:.1%}")
        lines.append(f"Spoof Probability: {analysis['spoof_probability']:.1%}")
        lines.append(f"Primary Assessment: {exp['primary_assessment']}")
        lines.append(f"Confidence Assessment: {exp['confidence_assessment']}")
        
        feature_analysis = exp['feature_analysis']
        lines.append("Feature Analysis:")
        lines.append(f"  • Ridge Continuity: {feature_analysis['ridge_characteristics']['continuity']:.3f}")
        lines.append(f"  • Ridge Contrast: {feature_analysis['ridge_characteristics']['contrast']:.3f}")
        lines.append(f"  • Pore Density: {feature_analysis['pore_analysis']['density']:.3f}")
        
        return lines
    
    def _generate_conclusions(self, analyses: List[Dict]) -> List[str]:
        """Generate forensic conclusions"""
        lines = []
        
        lines.append("Based on the comprehensive analysis performed:")
        lines.append("")
        
        # Summarize each analysis
        for analysis in analyses:
            if analysis['analysis_type'] == 'Pattern Classification':
                lines.append(f"1. Pattern classification indicates {analysis['prediction']} pattern")
            elif analysis['analysis_type'] == 'Finger Position Identification':
                lines.append(f"2. Finger position consistent with {analysis['prediction']} finger")
            elif analysis['analysis_type'] == 'Minutiae Detection':
                lines.append(f"3. Minutiae analysis detected {analysis['total_minutiae']} identifying features")
            elif analysis['analysis_type'] == 'Identity Matching':
                lines.append(f"4. Identity comparison resulted in {analysis['match_result']}")
            elif analysis['analysis_type'] == 'Presentation Attack Detection (PAD)':
                lines.append(f"5. Liveness detection classified print as {analysis['determination']}")
        
        lines.append("")
        lines.append("These findings support forensic identification and comparison procedures.")
        
        return lines
    
    def _generate_methodology_section(self) -> List[str]:
        """Generate methodology section"""
        lines = []
        
        lines.append("Analytical Methods:")
        lines.append("• Pattern Classification: Deep learning multi-task classification")
        lines.append("• Minutiae Detection: Crossing Number (CN) algorithm with quality filtering")
        lines.append("• Identity Matching: ORB feature matching with geometric consistency")
        lines.append("• Spoof Detection: Multi-modal feature analysis (ridge, pore, texture)")
        lines.append("")
        lines.append("Limitations:")
        lines.append("• Results dependent on image quality and resolution")
        lines.append("• Some analyses require manual verification for forensic testimony")
        lines.append("• Spoof detection uses rule-based approach pending training data")
        lines.append("• Matching results should be confirmed with additional impressions")
        
        return lines

def test_explainable_ai():
    """Test explainable AI assistant"""
    print("🤖 Testing Explainable AI Forensic Assistant")
    print("=" * 60)
    
    # Initialize assistant
    assistant = ForensicAIAssistant()
    
    # Generate proper mock analyses using the assistant methods
    sample_image = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
    
    # Generate real analyses
    pattern_analysis = assistant.explain_pattern_classification(sample_image, 'Loop', 0.92, {})
    finger_analysis = assistant.explain_finger_identification(sample_image, 'Index', 0.85, {})
    minutiae_analysis = assistant.explain_minutiae_detection(sample_image, 
        [{'type': 'ridge_ending', 'x': 100, 'y': 150}] * 24, [0.8] * 24)
    
    # Mock matching analysis
    template1 = {'feature_count': 200, 'matched_features': 146, 'match_ratio': 0.73}
    template2 = {'feature_count': 195, 'matched_features': 146, 'match_ratio': 0.75}
    matching_analysis = assistant.explain_identity_matching(template1, template2, 0.73, True)
    
    # Mock spoof analysis
    spoof_result = {
        'prediction': 'Live',
        'live_probability': 0.88,
        'spoof_probability': 0.12,
        'confidence': 0.88,
        'features': {
            'ridge_ridge_continuity': 0.85,
            'ridge_ridge_valley_contrast': 0.72,
            'pore_pore_density': 2.1,
            'texture_lbp_entropy': 2.8,
            'texture_stat_variance': 800.0
        }
    }
    spoof_analysis = assistant.explain_spoof_detection(sample_image, spoof_result)
    
    sample_analyses = [pattern_analysis, finger_analysis, minutiae_analysis, 
                      matching_analysis, spoof_analysis]
    
    # Create sample image
    print("🔍 Generating explanations...")
    
    # Test pattern explanation
    pattern_exp = assistant.explain_pattern_classification(
        sample_image, 'Loop', 0.92, {}
    )
    print(f"✅ Pattern explanation: {len(str(pattern_exp))} characters")
    
    # Test spoof explanation
    spoof_result = {
        'prediction': 'Live',
        'live_probability': 0.88,
        'spoof_probability': 0.12,
        'confidence': 0.88,
        'features': {
            'ridge_ridge_continuity': 0.85,
            'ridge_ridge_valley_contrast': 0.72,
            'pore_pore_density': 2.1,
            'texture_lbp_entropy': 2.8
        }
    }
    
    spoof_exp = assistant.explain_spoof_detection(sample_image, spoof_result)
    print(f"✅ Spoof explanation: {len(str(spoof_exp))} characters")
    
    # Generate comprehensive report
    print("\n📋 Generating forensic report...")
    report = assistant.generate_forensic_report(sample_analyses, "test_image.tif")
    
    # Save report
    with open('forensic_report.txt', 'w') as f:
        f.write(report)
    
    # Save explanations
    with open('ai_explanations.json', 'w') as f:
        json.dump({
            'pattern_explanation': pattern_exp,
            'spoof_explanation': spoof_exp
        }, f, indent=2)
    
    print(f"✅ Generated comprehensive forensic report: {len(report)} characters")
    print(f"💾 Report saved: forensic_report.txt")
    print(f"💾 Explanations saved: ai_explanations.json")
    
    # Show sample from report
    print(f"\n📄 Report Preview:")
    print(report[:500] + "...")

if __name__ == "__main__":
    test_explainable_ai()
