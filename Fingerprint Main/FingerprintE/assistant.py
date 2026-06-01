import re
import random
from datetime import datetime

# Knowledge base for fingerprint forensics
FORENSIC_KNOWLEDGE = {
    "pattern_types": {
        "loop": {
            "description": "A loop pattern flows in and out of the same side of the finger.",
            "characteristics": ["One delta", "One core", "Ridges flow from one side to the same side"],
            "frequency": "Most common pattern type (60-65% of fingerprints)",
            "forensic_value": "High - loops are distinctive and reliable for identification"
        },
        "whorl": {
            "description": "A whorl pattern has ridges that form circular or spiral patterns.",
            "characteristics": ["Two deltas", "One core", "Circular or spiral ridge flow"],
            "frequency": "Second most common (30-35% of fingerprints)",
            "forensic_value": "Very high - whorls have distinctive core and delta features"
        },
        "arch": {
            "description": "An arch pattern has ridges that flow from one side to the other.",
            "characteristics": ["No deltas", "No core", "Ridges flow from one side to the other"],
            "frequency": "Least common (5-10% of fingerprints)",
            "forensic_value": "Moderate - arches can be more difficult to match due to fewer distinctive features"
        }
    },
    "minutiae_types": {
        "ridge_ending": {
            "description": "A ridge ending occurs where a ridge stops abruptly.",
            "forensic_value": "High - ridge endings are reliable identification points",
            "detection_method": "Look for abrupt termination of ridge lines"
        },
        "bifurcation": {
            "description": "A bifurcation occurs where a ridge splits into two ridges.",
            "forensic_value": "High - bifurcations are distinctive and reliable",
            "detection_method": "Look for Y-shaped ridge patterns"
        }
    },
    "analysis_steps": [
        "Image quality assessment",
        "Pattern classification",
        "Minutiae detection and marking",
        "Feature extraction and measurement",
        "Comparison with reference samples",
        "Statistical analysis of match probability"
    ],
    "quality_factors": [
        "Image resolution and clarity",
        "Contrast and lighting",
        "Ridge definition",
        "Presence of artifacts or noise",
        "Fingerprint area coverage"
    ]
}

def agent_response(user_query, context=None):
    """
    Provide AI-powered forensic guidance for fingerprint analysis
    
    Args:
        user_query: User's question about fingerprint analysis
        context: Additional context (e.g., uploaded image info)
        
    Returns:
        str: AI assistant's response
    """
    try:
        # Normalize the query
        query = user_query.lower().strip()
        
        # Check for different types of questions
        if is_pattern_question(query):
            return answer_pattern_question(query)
        elif is_minutiae_question(query):
            return answer_minutiae_question(query)
        elif is_analysis_question(query):
            return answer_analysis_question(query)
        elif is_quality_question(query):
            return answer_quality_question(query)
        elif is_general_question(query):
            return answer_general_question(query)
        else:
            return generate_general_response(query, context)
            
    except Exception as e:
        return f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try rephrasing your question about fingerprint analysis."

def is_pattern_question(query):
    """Check if the question is about fingerprint patterns"""
    pattern_keywords = ['pattern', 'loop', 'whorl', 'arch', 'type', 'classification']
    return any(keyword in query for keyword in pattern_keywords)

def is_minutiae_question(query):
    """Check if the question is about minutiae"""
    minutiae_keywords = ['minutiae', 'ridge ending', 'bifurcation', 'feature', 'point']
    return any(keyword in query for keyword in minutiae_keywords)

def is_analysis_question(query):
    """Check if the question is about analysis process"""
    analysis_keywords = ['analyze', 'process', 'step', 'method', 'procedure', 'how to']
    return any(keyword in query for keyword in analysis_keywords)

def is_quality_question(query):
    """Check if the question is about image quality"""
    quality_keywords = ['quality', 'good', 'bad', 'clear', 'blurry', 'resolution']
    return any(keyword in query for keyword in quality_keywords)

def is_general_question(query):
    """Check if the question is a general forensic question"""
    general_keywords = ['what is', 'explain', 'tell me', 'help', 'guide']
    return any(keyword in query for keyword in general_keywords)

def answer_pattern_question(query):
    """Answer questions about fingerprint patterns"""
    if 'loop' in query:
        pattern_info = FORENSIC_KNOWLEDGE["pattern_types"]["loop"]
        return f"""**Loop Pattern Analysis:**
        
🔍 **Description:** {pattern_info['description']}

📊 **Characteristics:**
{chr(10).join(f"• {char}" for char in pattern_info['characteristics'])}

📈 **Frequency:** {pattern_info['frequency']}

🎯 **Forensic Value:** {pattern_info['forensic_value']}

💡 **Analysis Tip:** Look for the core (center of the loop) and delta (triangular area) to confirm this pattern type."""

    elif 'whorl' in query:
        pattern_info = FORENSIC_KNOWLEDGE["pattern_types"]["whorl"]
        return f"""**Whorl Pattern Analysis:**
        
🔍 **Description:** {pattern_info['description']}

📊 **Characteristics:**
{chr(10).join(f"• {char}" for char in pattern_info['characteristics'])}

📈 **Frequency:** {pattern_info['frequency']}

🎯 **Forensic Value:** {pattern_info['forensic_value']}

💡 **Analysis Tip:** Whorls are excellent for identification due to their distinctive core and delta features."""

    elif 'arch' in query:
        pattern_info = FORENSIC_KNOWLEDGE["pattern_types"]["arch"]
        return f"""**Arch Pattern Analysis:**
        
🔍 **Description:** {pattern_info['description']}

📊 **Characteristics:**
{chr(10).join(f"• {char}" for char in pattern_info['characteristics'])}

📈 **Frequency:** {pattern_info['frequency']}

🎯 **Forensic Value:** {pattern_info['forensic_value']}

💡 **Analysis Tip:** Focus on minutiae points since arches lack distinctive core/delta features."""

    else:
        return """**Fingerprint Pattern Types:**

🔍 **Three Main Pattern Types:**

1. **Loop Pattern** (60-65% of fingerprints)
   - Ridges flow in and out of the same side
   - Has one delta and one core
   - Most common pattern type

2. **Whorl Pattern** (30-35% of fingerprints)
   - Circular or spiral ridge patterns
   - Has two deltas and one core
   - Very distinctive for identification

3. **Arch Pattern** (5-10% of fingerprints)
   - Ridges flow from one side to the other
   - No deltas or core
   - Least common pattern type

💡 **Forensic Significance:** Pattern classification is the first step in fingerprint analysis and helps narrow down potential matches."""

def answer_minutiae_question(query):
    """Answer questions about minutiae"""
    if 'ridge ending' in query:
        minutiae_info = FORENSIC_KNOWLEDGE["minutiae_types"]["ridge_ending"]
        return f"""**Ridge Ending Analysis:**
        
🔍 **Description:** {minutiae_info['description']}

🎯 **Forensic Value:** {minutiae_info['forensic_value']}

🔬 **Detection Method:** {minutiae_info['detection_method']}

💡 **Analysis Tips:**
• Ridge endings are reliable identification points
• Look for abrupt termination of ridge lines
• Count total ridge endings for statistical analysis
• Consider the angle and position of each ending"""

    elif 'bifurcation' in query:
        minutiae_info = FORENSIC_KNOWLEDGE["minutiae_types"]["bifurcation"]
        return f"""**Bifurcation Analysis:**
        
🔍 **Description:** {minutiae_info['description']}

🎯 **Forensic Value:** {minutiae_info['forensic_value']}

🔬 **Detection Method:** {minutiae_info['detection_method']}

💡 **Analysis Tips:**
• Bifurcations are highly distinctive features
• Note the angle of the split
• Consider the relative positions of the three ridge segments
• Bifurcations are excellent for matching algorithms"""

    else:
        return """**Minutiae Analysis:**

🔍 **What are Minutiae?**
Minutiae are the small, distinctive features in fingerprints that make each print unique.

📊 **Two Main Types:**

1. **Ridge Endings**
   - Where a ridge stops abruptly
   - High forensic value
   - Easy to identify and measure

2. **Bifurcations**
   - Where a ridge splits into two
   - Very distinctive features
   - Excellent for matching

💡 **Forensic Importance:**
• Minutiae are the primary basis for fingerprint identification
• Typically need 8-12 matching minutiae for positive identification
• Each minutiae has specific location, type, and orientation
• Statistical analysis considers minutiae distribution and quality"""

def answer_analysis_question(query):
    """Answer questions about analysis process"""
    steps = FORENSIC_KNOWLEDGE["analysis_steps"]
    return f"""**Fingerprint Analysis Process:**

🔬 **Step-by-Step Analysis:**

{chr(10).join(f"{i+1}. **{step}**" for i, step in enumerate(steps))}

💡 **Key Considerations:**

• **Quality First:** Always assess image quality before analysis
• **Systematic Approach:** Follow the steps in order
• **Documentation:** Record all findings and measurements
• **Verification:** Double-check classifications and measurements
• **Statistical Validation:** Use probability calculations for match confidence

🎯 **Best Practices:**
- Use consistent lighting and magnification
- Mark minutiae clearly and accurately
- Consider multiple pattern classifications if unclear
- Document any artifacts or quality issues
- Maintain chain of custody for evidence"""

def answer_quality_question(query):
    """Answer questions about image quality"""
    quality_factors = FORENSIC_KNOWLEDGE["quality_factors"]
    return f"""**Image Quality Assessment:**

📊 **Quality Factors to Consider:**

{chr(10).join(f"• **{factor}**" for factor in quality_factors)}

🔍 **Quality Indicators:**

✅ **Good Quality:**
- Clear, well-defined ridges
- Good contrast between ridges and valleys
- Adequate resolution (300+ DPI)
- Minimal noise or artifacts
- Complete fingerprint area visible

❌ **Poor Quality:**
- Blurry or unclear ridges
- Low contrast
- Excessive noise or artifacts
- Incomplete fingerprint area
- Poor lighting or shadows

💡 **Improving Quality:**
• Use proper lighting and contrast
• Ensure adequate resolution
• Clean the fingerprint surface
• Use appropriate enhancement techniques
• Consider multiple capture attempts"""

def answer_general_question(query):
    """Answer general forensic questions"""
    if 'what is' in query and 'fingerprint' in query:
        return """**What is Fingerprint Analysis?**

🔍 **Definition:**
Fingerprint analysis is the scientific examination of friction ridge skin impressions to identify individuals or determine characteristics.

📊 **Key Principles:**

1. **Uniqueness:** No two fingerprints are identical
2. **Permanence:** Fingerprint patterns remain unchanged throughout life
3. **Systematic Classification:** Patterns can be categorized and compared

🎯 **Applications:**
• Criminal investigations
• Identity verification
• Access control systems
• Historical document analysis
• Missing person identification

💡 **Scientific Basis:**
Fingerprint analysis is based on the uniqueness of friction ridge patterns, which develop during fetal development and remain stable throughout life."""

    elif 'help' in query or 'guide' in query:
        return """**Fingerprint Analysis Guide:**

🤖 **How I Can Help You:**

🔍 **Pattern Analysis:**
- Ask about loop, whorl, or arch patterns
- Get pattern characteristics and forensic value
- Learn identification techniques

🔬 **Minutiae Analysis:**
- Understand ridge endings and bifurcations
- Learn detection methods
- Get analysis tips

📊 **Quality Assessment:**
- Evaluate image quality
- Identify quality issues
- Get improvement suggestions

🔬 **Analysis Process:**
- Step-by-step analysis guidance
- Best practices and procedures
- Documentation requirements

💡 **Example Questions:**
• "What is a loop pattern?"
• "How do I identify minutiae?"
• "Is my fingerprint image good quality?"
• "What are the steps in fingerprint analysis?"
• "Explain ridge endings vs bifurcations"

🎯 **Just ask me anything about fingerprint forensics!**"""

    else:
        return generate_general_response(query)

def generate_general_response(query, context=None):
    """Generate a general response for unrecognized queries"""
    responses = [
        f"I understand you're asking about '{query}'. Let me help you with fingerprint analysis. Could you be more specific about what aspect of fingerprint forensics you'd like to learn about?",
        
        f"That's an interesting question about '{query}'. In fingerprint analysis, we focus on pattern classification, minutiae detection, and quality assessment. What specific area would you like to explore?",
        
        f"I'm here to help with fingerprint forensics! Your question about '{query}' might relate to pattern analysis, minutiae detection, or image quality. Could you clarify what you'd like to know?",
        
        f"Great question! For fingerprint analysis, we typically work with pattern types (loops, whorls, arches), minutiae points, and image quality. How can I help you with your analysis?"
    ]
    
    return random.choice(responses)

# Additional utility functions
def get_analysis_tips(pattern_type=None):
    """Get analysis tips for specific pattern types"""
    tips = {
        "general": [
            "Always assess image quality first",
            "Use consistent lighting and magnification",
            "Document all findings systematically",
            "Consider multiple pattern classifications if unclear",
            "Focus on minutiae for positive identification"
        ],
        "loop": [
            "Look for the core (center of the loop)",
            "Identify the delta (triangular area)",
            "Note the direction of ridge flow",
            "Count ridge endings and bifurcations",
            "Measure distances between key features"
        ],
        "whorl": [
            "Identify both deltas and the core",
            "Note the type of whorl (plain, central pocket, etc.)",
            "Measure distances between deltas",
            "Count ridges between core and deltas",
            "Look for distinctive spiral patterns"
        ],
        "arch": [
            "Focus on minutiae since there's no core/delta",
            "Note the overall ridge flow direction",
            "Count and map all minutiae points",
            "Consider tented arch vs plain arch",
            "Pay attention to ridge density"
        ]
    }
    
    if pattern_type and pattern_type in tips:
        return tips[pattern_type]
    else:
        return tips["general"]

def calculate_match_probability(minutiae_count, pattern_match=True):
    """Calculate theoretical match probability"""
    # Simplified probability calculation
    base_probability = 0.1  # Base probability for random match
    
    if pattern_match:
        base_probability *= 0.5  # Pattern match reduces probability
    
    # Minutiae count affects probability
    if minutiae_count >= 12:
        probability = base_probability * (0.1 ** (minutiae_count - 12))
    else:
        probability = base_probability * (0.5 ** (12 - minutiae_count))
    
    return max(probability, 1e-10)  # Minimum probability
