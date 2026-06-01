"""
Person E - AI Assistant & Explainability
========================================

Implements AI-powered explanations for fingerprint analysis decisions.
"""

import os
import json
from typing import Dict, List, Optional, Any
import time

class AIAssistant:
    """
    AI Assistant for fingerprint analysis explanations
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        # Allow explicit key, config.py, or env var
        cfg_key = None
        try:
            # Prefer project config if available
            from config import OPENAI_API_KEY as CFG_KEY
            cfg_key = CFG_KEY
        except Exception:
            cfg_key = None

        self.openai_api_key = openai_api_key or cfg_key or os.environ.get("OPENAI_API_KEY")
        self.client = None
        
        # Try to initialize OpenAI client if API key is provided
        if self.openai_api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_api_key)
                print("✅ OpenAI client initialized")
            except ImportError:
                print("⚠️ OpenAI library not available, using rule-based explanations")
            except Exception as e:
                print(f"⚠️ OpenAI client initialization failed: {e}")
    
    def agent_generate(self, step: str, payload: Dict) -> Dict:
        """
        Generate AI explanation for a specific step
        
        Args:
            step: Analysis step ("classify", "minutiae", "match", "agent")
            payload: Data payload for the step
            
        Returns:
            Dictionary with text explanation and facts
        """
        try:
            if step == "classify":
                return self._explain_classification(payload)
            elif step == "minutiae":
                return self._explain_minutiae(payload)
            elif step == "match":
                return self._explain_matching(payload)
            elif step == "agent":
                return self._explain_comprehensive(payload)
            else:
                return self._create_error_response(f"Unknown step: {step}")
                
        except Exception as e:
            return self._create_error_response(str(e))
    
    def _explain_classification(self, payload: Dict) -> Dict:
        """Explain fingerprint pattern classification"""
        label = payload.get('label', 'Unknown')
        confidence = payload.get('confidence', 0.0)
        probabilities = payload.get('probabilities', {})
        
        # Generate explanation
        if self.client:
            explanation = self._generate_ai_explanation(
                "classification",
                {"label": label, "confidence": confidence, "probabilities": probabilities}
            )
        else:
            explanation = self._generate_rule_based_classification_explanation(
                label, confidence, probabilities
            )
        
        facts = {
            "pattern_type": label,
            "confidence_score": confidence,
            "classification_method": "Random Forest with texture and frequency features",
            "pattern_characteristics": self._get_pattern_characteristics(label)
        }
        
        return {
            "text": explanation,
            "facts": facts
        }
    
    def _explain_minutiae(self, payload: Dict) -> Dict:
        """Explain minutiae detection results"""
        minutiae = payload.get('minutiae', [])
        quality_score = payload.get('quality_score', 0.0)
        spoof_suspected = payload.get('spoof_suspected', False)
        
        # Count minutiae types
        endings = len([m for m in minutiae if m.get('type') == 'ending'])
        bifurcations = len([m for m in minutiae if m.get('type') == 'bifurcation'])
        total_minutiae = len(minutiae)
        
        # Generate explanation
        if self.client:
            explanation = self._generate_ai_explanation(
                "minutiae",
                {
                    "total_minutiae": total_minutiae,
                    "endings": endings,
                    "bifurcations": bifurcations,
                    "quality_score": quality_score,
                    "spoof_suspected": spoof_suspected
                }
            )
        else:
            explanation = self._generate_rule_based_minutiae_explanation(
                total_minutiae, endings, bifurcations, quality_score, spoof_suspected
            )
        
        facts = {
            "total_minutiae": total_minutiae,
            "ridge_endings": endings,
            "bifurcations": bifurcations,
            "quality_score": quality_score,
            "spoof_detected": spoof_suspected,
            "detection_method": "Crossing Number algorithm with confidence scoring"
        }
        
        return {
            "text": explanation,
            "facts": facts
        }
    
    def _explain_matching(self, payload: Dict) -> Dict:
        """Explain fingerprint matching results"""
        match_score = payload.get('match_score', 0.0)
        matched_id = payload.get('matched_id')
        access_granted = payload.get('access_granted', False)
        spoof_flag = payload.get('spoof_flag', False)
        
        # Generate explanation
        if self.client:
            explanation = self._generate_ai_explanation(
                "matching",
                {
                    "match_score": match_score,
                    "matched_id": matched_id,
                    "access_granted": access_granted,
                    "spoof_flag": spoof_flag
                }
            )
        else:
            explanation = self._generate_rule_based_matching_explanation(
                match_score, matched_id, access_granted, spoof_flag
            )
        
        facts = {
            "match_score": match_score,
            "matched_identity": matched_id,
            "access_granted": access_granted,
            "spoof_detected": spoof_flag,
            "matching_method": "Hungarian algorithm with minutiae correspondence",
            "threshold_accept": 0.25,
            "threshold_reject": 0.10
        }
        
        return {
            "text": explanation,
            "facts": facts
        }
    
    def _explain_comprehensive(self, payload: Dict) -> Dict:
        """Generate comprehensive explanation for all analysis steps"""
        # Extract data from payload
        classification = payload.get('classification', {})
        minutiae = payload.get('minutiae', {})
        matching = payload.get('matching', {})
        
        # Generate comprehensive explanation
        if self.client:
            explanation = self._generate_ai_explanation(
                "comprehensive",
                {
                    "classification": classification,
                    "minutiae": minutiae,
                    "matching": matching
                }
            )
        else:
            explanation = self._generate_rule_based_comprehensive_explanation(
                classification, minutiae, matching
            )
        
        # Compile comprehensive facts
        facts = {
            "analysis_timestamp": time.time(),
            "pattern_classification": classification,
            "minutiae_analysis": minutiae,
            "identity_matching": matching,
            "overall_confidence": self._calculate_overall_confidence(classification, minutiae, matching),
            "security_assessment": self._assess_security_risk(classification, minutiae, matching)
        }
        
        return {
            "text": explanation,
            "facts": facts
        }
    
    def _generate_ai_explanation(self, step: str, data: Dict) -> str:
        """Generate AI-powered explanation using OpenAI"""
        if not self.client:
            return "AI explanation not available"
        
        try:
            # Create prompt based on step
            if step == "classification":
                prompt = f"""
                Explain the fingerprint pattern classification result:
                - Pattern: {data['label']}
                - Confidence: {data['confidence']:.2%}
                - Probabilities: {data['probabilities']}
                
                Provide a clear, forensic-style explanation of what this pattern means and how confident the system is.
                """
            elif step == "minutiae":
                prompt = f"""
                Explain the minutiae detection results:
                - Total minutiae: {data['total_minutiae']}
                - Ridge endings: {data['endings']}
                - Bifurcations: {data['bifurcations']}
                - Quality score: {data['quality_score']:.2%}
                - Spoof suspected: {data['spoof_suspected']}
                
                Explain what these minutiae mean for fingerprint uniqueness and quality.
                """
            elif step == "matching":
                prompt = f"""
                Explain the fingerprint matching result:
                - Match score: {data['match_score']:.2%}
                - Matched ID: {data['matched_id']}
                - Access granted: {data['access_granted']}
                - Spoof detected: {data['spoof_flag']}
                
                Explain the matching decision and security implications.
                """
            else:  # comprehensive
                prompt = f"""
                Provide a comprehensive forensic analysis summary:
                - Classification: {data['classification']}
                - Minutiae: {data['minutiae']}
                - Matching: {data['matching']}
                
                Give a complete forensic report-style explanation.
                """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a forensic fingerprint analysis expert. Provide clear, technical explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"AI explanation failed: {str(e)}"
    
    def _generate_rule_based_classification_explanation(self, label: str, confidence: float, probabilities: Dict) -> str:
        """Generate rule-based classification explanation"""
        confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        
        explanation = f"The fingerprint has been classified as a {label} pattern with {confidence_level} confidence ({confidence:.1%}). "
        
        if label == "Loop":
            explanation += "Loop patterns are characterized by ridges that enter from one side, curve around, and exit from the same side. "
        elif label == "Whorl":
            explanation += "Whorl patterns feature ridges that form complete circles or spirals around a central point. "
        elif label == "Arch":
            explanation += "Arch patterns have ridges that enter from one side, rise in the center, and exit from the opposite side. "
        
        explanation += f"The classification confidence of {confidence:.1%} indicates {'strong' if confidence > 0.8 else 'moderate' if confidence > 0.6 else 'weak'} pattern recognition."
        
        return explanation
    
    def _generate_rule_based_minutiae_explanation(self, total: int, endings: int, bifurcations: int, quality: float, spoof: bool) -> str:
        """Generate rule-based minutiae explanation"""
        explanation = f"Minutiae analysis detected {total} total minutiae points: {endings} ridge endings and {bifurcations} bifurcations. "
        
        if total < 10:
            explanation += "The low minutiae count suggests poor image quality or insufficient ridge information. "
        elif total > 100:
            explanation += "The high minutiae count indicates good ridge detail and image quality. "
        else:
            explanation += "The minutiae count is within normal range for fingerprint analysis. "
        
        if spoof:
            explanation += "⚠️ Spoof detection algorithms have flagged this fingerprint as potentially artificial. "
        else:
            explanation += "✅ No spoof indicators detected - fingerprint appears to be from a live finger. "
        
        quality_level = "excellent" if quality > 0.8 else "good" if quality > 0.6 else "fair" if quality > 0.4 else "poor"
        explanation += f"The overall quality score of {quality:.1%} indicates {quality_level} image quality."
        
        return explanation
    
    def _generate_rule_based_matching_explanation(self, score: float, matched_id: str, access: bool, spoof: bool) -> str:
        """Generate rule-based matching explanation"""
        if spoof:
            return "🚨 ACCESS DENIED: Spoof detection algorithms have identified this as a potentially artificial fingerprint. Security protocols require live finger verification."
        
        if not matched_id:
            return f"❌ NO MATCH: The fingerprint did not match any entries in the database (similarity score: {score:.1%}). Access denied."
        
        if access:
            explanation = f"✅ ACCESS GRANTED: Fingerprint successfully matched to {matched_id} with {score:.1%} similarity. "
            if score > 0.9:
                explanation += "Very high confidence match."
            elif score > 0.8:
                explanation += "High confidence match."
            else:
                explanation += "Moderate confidence match."
        else:
            explanation = f"❌ ACCESS DENIED: Fingerprint matched to {matched_id} but similarity score ({score:.1%}) is below the acceptance threshold (70%). "
            explanation += "The match is not confident enough for access authorization."
        
        return explanation
    
    def _generate_rule_based_comprehensive_explanation(self, classification: Dict, minutiae: Dict, matching: Dict) -> str:
        """Generate comprehensive rule-based explanation"""
        explanation = "🔍 COMPREHENSIVE FINGERPRINT ANALYSIS REPORT\n\n"
        
        # Classification section
        if classification:
            label = classification.get('label', 'Unknown')
            confidence = classification.get('confidence', 0.0)
            explanation += f"📋 PATTERN CLASSIFICATION: {label} pattern detected with {confidence:.1%} confidence.\n"
        
        # Minutiae section
        if minutiae:
            total = minutiae.get('minutiae_count', 0)
            quality = minutiae.get('quality_score', 0.0)
            spoof = minutiae.get('spoof_suspected', False)
            explanation += f"🔬 MINUTIAE ANALYSIS: {total} minutiae points detected with {quality:.1%} quality score. "
            explanation += "Spoof detection: " + ("⚠️ SUSPECTED" if spoof else "✅ CLEAR") + "\n"
        
        # Matching section
        if matching:
            score = matching.get('match_score', 0.0)
            matched_id = matching.get('matched_id')
            access = matching.get('access_granted', False)
            explanation += f"🔐 IDENTITY MATCHING: "
            if matched_id:
                explanation += f"Matched to {matched_id} with {score:.1%} similarity. "
            else:
                explanation += f"No database match found (score: {score:.1%}). "
            explanation += "Access: " + ("✅ GRANTED" if access else "❌ DENIED") + "\n"
        
        # Overall assessment
        explanation += f"\n📊 OVERALL ASSESSMENT: "
        if matching.get('access_granted', False):
            explanation += "Fingerprint verification successful. Access authorized."
        else:
            explanation += "Fingerprint verification failed. Access denied."
        
        return explanation
    
    def _get_pattern_characteristics(self, pattern: str) -> Dict:
        """Get characteristics of fingerprint patterns"""
        characteristics = {
            "Loop": {
                "description": "Ridges enter from one side, curve around, and exit from the same side",
                "frequency": "Most common pattern (60-65% of population)",
                "uniqueness": "High - each loop has unique ridge characteristics"
            },
            "Whorl": {
                "description": "Ridges form complete circles or spirals around a central point",
                "frequency": "Second most common (30-35% of population)",
                "uniqueness": "Very high - complex spiral patterns are highly distinctive"
            },
            "Arch": {
                "description": "Ridges enter from one side, rise in center, and exit opposite side",
                "frequency": "Least common (5-10% of population)",
                "uniqueness": "High - simple but distinctive pattern"
            }
        }
        return characteristics.get(pattern, {"description": "Unknown pattern", "frequency": "Unknown", "uniqueness": "Unknown"})
    
    def _calculate_overall_confidence(self, classification: Dict, minutiae: Dict, matching: Dict) -> float:
        """Calculate overall confidence score"""
        scores = []
        
        if classification and 'confidence' in classification:
            scores.append(classification['confidence'])
        
        if minutiae and 'quality_score' in minutiae:
            scores.append(minutiae['quality_score'])
        
        if matching and 'match_score' in matching:
            scores.append(matching['match_score'])
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _assess_security_risk(self, classification: Dict, minutiae: Dict, matching: Dict) -> Dict:
        """Assess overall security risk"""
        risk_factors = []
        risk_score = 0.0
        
        # Check spoof detection
        if minutiae and minutiae.get('spoof_suspected', False):
            risk_factors.append("Spoof detected")
            risk_score += 0.5
        
        # Check quality
        if minutiae and minutiae.get('quality_score', 0.0) < 0.5:
            risk_factors.append("Poor image quality")
            risk_score += 0.3
        
        # Check matching confidence
        if matching and matching.get('match_score', 0.0) < 0.6:
            risk_factors.append("Low match confidence")
            risk_score += 0.2
        
        risk_level = "HIGH" if risk_score > 0.5 else "MEDIUM" if risk_score > 0.2 else "LOW"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors
        }
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create error response"""
        return {
            "text": f"Error generating explanation: {error_message}",
            "facts": {"error": error_message}
        }

# Global assistant instance
_assistant_instance = None

def get_assistant(openai_api_key: Optional[str] = None) -> AIAssistant:
    """Get or create global assistant instance"""
    global _assistant_instance
    if _assistant_instance is None:
        # Attempt to read from explicit arg or env var
        key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        _assistant_instance = AIAssistant(key)
    return _assistant_instance

def agent_generate(step: str, payload: Dict) -> Dict:
    """
    Main function to generate AI explanations
    
    Args:
        step: Analysis step
        payload: Data payload
        
    Returns:
        Explanation dictionary
    """
    assistant = get_assistant()
    return assistant.agent_generate(step, payload)

if __name__ == "__main__":
    # Test the AI assistant
    print("🤖 Testing AI Assistant...")
    
    assistant = get_assistant()
    
    # Test classification explanation
    print("\n📋 Testing classification explanation:")
    classification_payload = {
        "label": "Loop",
        "confidence": 0.85,
        "probabilities": {"Loop": 0.85, "Whorl": 0.10, "Arch": 0.05}
    }
    result = assistant.agent_generate("classify", classification_payload)
    print(f"Explanation: {result['text']}")
    
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
    result = assistant.agent_generate("minutiae", minutiae_payload)
    print(f"Explanation: {result['text']}")
    
    # Test matching explanation
    print("\n🔐 Testing matching explanation:")
    matching_payload = {
        "match_score": 0.85,
        "matched_id": "user_001",
        "access_granted": True,
        "spoof_flag": False
    }
    result = assistant.agent_generate("match", matching_payload)
    print(f"Explanation: {result['text']}")
    
    print("\n✅ AI Assistant testing complete!")
