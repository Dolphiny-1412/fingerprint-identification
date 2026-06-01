"""
Modern Fingerprint Analysis System - Streamlit UI (2025 Design Standards)
=========================================================================

Redesigned with:
- Proper theming via config.toml
- Responsive layout primitives (columns, containers, tabs)
- Modern component patterns
- Improved accessibility
- Performance optimizations
- Professional visual hierarchy

Same 4-step wizard flow: Upload → Classify → Minutiae → Match + Agent
"""

import streamlit as st
import os
import sys
import json
import uuid
from datetime import datetime
from PIL import Image
import io
import base64
from pathlib import Path
import time

# Add the src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import our modules
from core.classifier import classify_fingerprint
from core.minutiae_detector import detect_minutiae
from core.fingerprint_matcher import match_minutiae, seed_database
from core.ai_assistant import agent_generate

# ----- Configuration -----
st.set_page_config(
    page_title="Fingerprint Analysis System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/fingerprint-system',
        'Report a bug': "https://github.com/yourusername/fingerprint-system/issues",
        'About': "# Forensic Fingerprint Analysis System\nVersion 2.0 (2025 Edition)"
    }
)

# ----- Modern CSS with Design Tokens -----
# Following 2025 best practices: CSS variables for design tokens
def inject_modern_theme():
    """Inject modern themed CSS with design tokens and accessibility improvements"""
    st.markdown("""
    <style>
    /* Design Tokens - Single source of truth */
    :root {
        /* Color palette */
        --color-primary: #DC143C;
        --color-primary-dark: #8B0000;
        --color-primary-light: #FF6B6B;
        --color-background: #0F0F0F;
        --color-surface: #1A1A1A;
        --color-surface-light: #242424;
        --color-text: #FFFFFF;
        --color-text-muted: #B8B8B8;
        --color-success: #00C851;
        --color-warning: #FFB300;
        --color-danger: #FF4444;
        --color-info: #33B5E5;
        
        /* Spacing scale */
        --spacing-xs: 0.25rem;
        --spacing-sm: 0.5rem;
        --spacing-md: 1rem;
        --spacing-lg: 1.5rem;
        --spacing-xl: 2rem;
        --spacing-xxl: 3rem;
        
        /* Typography scale */
        --font-size-xs: 0.75rem;
        --font-size-sm: 0.875rem;
        --font-size-base: 1rem;
        --font-size-lg: 1.125rem;
        --font-size-xl: 1.5rem;
        --font-size-xxl: 2rem;
        --font-weight-normal: 400;
        --font-weight-medium: 500;
        --font-weight-bold: 700;
        
        /* Border radius */
        --radius-sm: 0.25rem;
        --radius-md: 0.5rem;
        --radius-lg: 1rem;
        --radius-xl: 1.5rem;
        
        /* Shadows */
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        --shadow-md: 0 3px 6px rgba(0,0,0,0.15), 0 2px 4px rgba(0,0,0,0.12);
        --shadow-lg: 0 10px 20px rgba(0,0,0,0.15), 0 3px 6px rgba(0,0,0,0.10);
        --shadow-xl: 0 15px 25px rgba(0,0,0,0.15), 0 5px 10px rgba(0,0,0,0.05);
        
        /* Transitions */
        --transition-fast: 150ms ease-in-out;
        --transition-base: 250ms ease-in-out;
        --transition-slow: 350ms ease-in-out;
    }
    
    /* Global styles using tokens */
    .stApp {
        background: linear-gradient(135deg, var(--color-background) 0%, #1a0000 100%);
        color: var(--color-text);
    }
    
    /* Typography hierarchy */
    h1, h2, h3, h4, h5, h6 {
        color: var(--color-text);
        font-weight: var(--font-weight-bold);
        line-height: 1.2;
    }
    
    h1 { font-size: var(--font-size-xxl); }
    h2 { font-size: var(--font-size-xl); }
    h3 { font-size: var(--font-size-lg); }
    
    /* Modern card component */
    .modern-card {
        background: var(--color-surface);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        margin: var(--spacing-md) 0;
        box-shadow: var(--shadow-md);
        transition: all var(--transition-base);
        position: relative;
        overflow: hidden;
    }
    
    .modern-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--color-primary), var(--color-primary-light));
    }
    
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-xl);
        border-color: var(--color-primary);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, var(--color-surface) 0%, var(--color-surface-light) 100%);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: var(--radius-md);
        padding: var(--spacing-md);
        text-align: center;
        transition: all var(--transition-fast);
    }
    
    .metric-card:hover {
        border-color: var(--color-primary);
        box-shadow: 0 0 20px rgba(220,20,60,0.2);
    }
    
    .metric-value {
        font-size: var(--font-size-xl);
        font-weight: var(--font-weight-bold);
        color: var(--color-primary-light);
        margin: var(--spacing-xs) 0;
    }
    
    .metric-label {
        font-size: var(--font-size-sm);
        color: var(--color-text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Progress indicator */
    .progress-step {
        display: flex;
        align-items: center;
        padding: var(--spacing-sm) var(--spacing-md);
        margin: var(--spacing-xs) 0;
        border-radius: var(--radius-md);
        transition: all var(--transition-fast);
    }
    
    .progress-step.active {
        background: rgba(220,20,60,0.1);
        border-left: 3px solid var(--color-primary);
    }
    
    .progress-step.completed {
        opacity: 0.7;
    }
    
    .progress-icon {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: var(--spacing-sm);
        background: var(--color-surface-light);
        border: 2px solid var(--color-primary);
        font-size: var(--font-size-sm);
    }
    
    .progress-step.active .progress-icon {
        background: var(--color-primary);
        color: white;
    }
    
    /* Modern buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-dark) 100%);
        color: white;
        border: none;
        padding: var(--spacing-sm) var(--spacing-lg);
        border-radius: var(--radius-md);
        font-weight: var(--font-weight-medium);
        transition: all var(--transition-fast);
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        background: linear-gradient(135deg, var(--color-primary-light) 0%, var(--color-primary) 100%);
    }
    
    /* Sidebar improvements */
    section[data-testid="stSidebar"] {
        background: var(--color-surface);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* File uploader */
    .stFileUploader {
        background: var(--color-surface);
        border: 2px dashed rgba(220,20,60,0.3);
        border-radius: var(--radius-lg);
        padding: var(--spacing-xl);
        transition: all var(--transition-base);
    }
    
    .stFileUploader:hover {
        border-color: var(--color-primary);
        background: rgba(220,20,60,0.05);
    }
    
    /* Accessibility improvements */
    *:focus {
        outline: 2px solid var(--color-primary);
        outline-offset: 2px;
    }
    
    /* Text wrapping and readability improvements */
    .stMarkdown {
        word-wrap: break-word;
        overflow-wrap: break-word;
        hyphens: auto;
        line-height: 1.6;
    }
    
    .stMarkdown p {
        margin-bottom: 1rem;
        text-align: justify;
    }
    
    /* AI Assistant response styling */
    .ai-response {
        background: rgba(220, 20, 60, 0.05);
        border-left: 4px solid var(--color-primary);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .ai-response h1, .ai-response h2, .ai-response h3 {
        color: var(--color-primary);
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .ai-response ul, .ai-response ol {
        margin-left: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .ai-response li {
        margin-bottom: 0.5rem;
    }
    
    /* Loading states */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Responsive utilities */
    @media (max-width: 768px) {
        .modern-card {
            padding: var(--spacing-md);
        }
        
        h1 { font-size: var(--font-size-xl); }
        h2 { font-size: var(--font-size-lg); }
        
        .stButton > button {
            padding: var(--spacing-xs) var(--spacing-md);
            font-size: var(--font-size-sm);
        }
        
        .metric-card {
            padding: var(--spacing-sm);
        }
        
        .stFileUploader {
            padding: var(--spacing-md);
        }
        
        .ai-response {
            padding: var(--spacing-sm);
            font-size: var(--font-size-sm);
        }
        
        .stExpander {
            margin: var(--spacing-xs) 0;
        }
    }
    
    /* Mobile-first responsive design */
    @media (max-width: 480px) {
        :root {
            --spacing-lg: 1rem;
            --spacing-xl: 1.5rem;
            --font-size-xl: 1.25rem;
            --font-size-xxl: 1.5rem;
        }
        
        .stApp {
            padding: var(--spacing-sm);
        }
        
        .modern-card {
            margin: var(--spacing-xs) 0;
            padding: var(--spacing-sm);
        }
        
        .stButton > button {
            width: 100%;
            margin: var(--spacing-xs) 0;
        }
        
        .stFileUploader {
            padding: var(--spacing-sm);
        }
        
        /* Ensure text doesn't overflow on small screens */
        .stMarkdown, .stText {
            word-wrap: break-word;
            overflow-wrap: break-word;
            hyphens: auto;
        }
        
        /* Optimize columns for mobile */
        .stColumns {
            flex-direction: column;
        }
        
        .stColumn {
            width: 100% !important;
            margin: var(--spacing-xs) 0;
        }
    }
    
    /* Tablet optimization */
    @media (min-width: 481px) and (max-width: 1024px) {
        .modern-card {
            padding: var(--spacing-md);
        }
        
        .stButton > button {
            padding: var(--spacing-sm) var(--spacing-md);
        }
        
        /* Optimize 4-column layouts for tablets */
        .stColumns[data-testid="column"] {
            flex-wrap: wrap;
        }
        
        .stColumn:nth-child(1),
        .stColumn:nth-child(2) {
            width: 50% !important;
        }
        
        .stColumn:nth-child(3),
        .stColumn:nth-child(4) {
            width: 50% !important;
        }
    }
    
    /* Large screen optimization */
    @media (min-width: 1025px) {
        .stApp {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .modern-card {
            padding: var(--spacing-xl);
        }
        
        .stButton > button {
            padding: var(--spacing-md) var(--spacing-xl);
        }
    }
    
    /* High DPI displays */
    @media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
        .modern-card {
            border-width: 0.5px;
        }
        
        .stButton > button {
            border-width: 0.5px;
        }
    }
    
    /* Print styles */
    @media print {
        .stSidebar, .stButton {
            display: none !important;
        }
        
        .modern-card {
            border: 1px solid #000;
            box-shadow: none;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ----- Helper Functions (with performance optimizations) -----

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_image(file_path):
    """Load and cache images for performance"""
    return Image.open(file_path)

def create_metric_card(label, value, delta=None, delta_color="normal"):
    """Create a modern metric card with optional delta"""
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=f"Current {label.lower()}"
    )

def create_progress_indicator(current_step, total_steps=5):
    """Create a visual progress indicator"""
    progress = current_step / total_steps
    st.progress(progress, text=f"Step {current_step} of {total_steps}")

def display_analysis_section(title, description, icon="📊"):
    """Create a consistent section header with responsive design"""
    st.markdown(f"""
    <div class="modern-card" style="text-align: center;">
        <h3 style="margin-bottom: var(--spacing-sm); font-size: clamp(1.5rem, 4vw, 2rem);">
            {icon} {title}
        </h3>
        <p style="color: var(--color-text-muted); font-size: clamp(0.875rem, 2.5vw, 1rem); 
                  line-height: 1.5; margin: 0; max-width: 100%; overflow-wrap: break-word;">
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ----- File handling utilities (unchanged logic) -----
def create_directories():
    """Create required directories for file storage"""
    directories = ['src/data/uploads', 'src/data/derived', 'src/data/fingerprints_db', 'src/logs', 'src/models', 'src/reports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return file path"""
    if uploaded_file is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = str(uuid.uuid4())[:8]
        filename = f"fingerprint_{timestamp}_{file_id}.png"
        filepath = os.path.join("src/data/uploads", filename)
        
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return filepath
    return None

def save_analysis_log(data, step_name, original_filename):
    """Save analysis results as JSON log"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    filename = f"{base_name}_{step_name}_{timestamp}.json"
    filepath = os.path.join("src/logs", filename)
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "step": step_name,
        "original_file": original_filename,
        "results": data
    }
    
    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    return filepath

# ----- Main Application -----
def main():
    # Initialize
    create_directories()
    inject_modern_theme()
    
    # Session state initialization
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # ----- Sidebar with modern progress tracking -----
    with st.sidebar:
        st.title("🔍 Analysis Control Panel")
        
        # Progress tracking with visual indicators
        with st.container():
            st.subheader("Progress Tracker")
            create_progress_indicator(st.session_state.current_step)
            
            steps = [
                ("Upload", "📤", 1),
                ("Classify", "🏷️", 2),
                ("Minutiae", "🔬", 3),
                ("Match", "🔐", 4),
                ("AI Analysis", "🤖", 5)
            ]
            
            for name, icon, step_num in steps:
                status = "✅" if step_num < st.session_state.current_step else "🔄" if step_num == st.session_state.current_step else "⏳"
                active = "active" if step_num == st.session_state.current_step else "completed" if step_num < st.session_state.current_step else ""
                
                st.markdown(f"""
                <div class="progress-step {active}">
                    <div class="progress-icon">{step_num}</div>
                    <span>{icon} {name} {status}</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Database metrics in a modern card
        with st.container():
            st.subheader("🗄️ Database Metrics")
            try:
                from core.fingerprint_matcher import get_matcher
                matcher = get_matcher()
                stats = matcher.get_database_statistics()
                
                col1, col2 = st.columns(2)
                with col1:
                    create_metric_card("Entries", stats.get('total_entries', 0))
                with col2:
                    create_metric_card("Accesses", stats.get('total_accesses', 0))
                
                if stats.get('total_entries', 0) == 0:
                    if st.button("🌱 Initialize Database", use_container_width=True):
                        with st.spinner("Seeding database..."):
                            seed_database()
                        st.success("Database initialized!")
                        time.sleep(1)
                        st.rerun()
            except Exception as e:
                st.error(f"Database connection issue: {str(e)[:50]}...")
        
        st.divider()
        
        # AI Assistant in expandable section
        with st.expander("🤖 AI Assistant", expanded=False):
            st.info("💡 **Tip**: The AI Assistant now runs fresh analysis with the latest matching system for accurate results!")
            user_question = st.text_area(
                "Ask about the analysis:",
                placeholder="E.g., What type of pattern is this?",
                height=100,
                key="ai_question"
            )
            
            if st.button("Get AI Analysis", use_container_width=True, type="primary"):
                if user_question and st.session_state.uploaded_file:
                    with st.spinner("AI is thinking..."):
                        # Re-run analysis with current fixed system
                        try:
                            # Get fresh analysis results (using global imports)
                            
                            # Save uploaded file temporarily
                            temp_path = save_uploaded_file(st.session_state.uploaded_file)
                            
                            # Run fresh analysis
                            classification_result = classify_fingerprint(temp_path)
                            minutiae_result = detect_minutiae(temp_path)
                            matching_result = match_minutiae(minutiae_result)
                            
                            # Create fresh payload
                            payload = {
                                "classification": classification_result,
                                "minutiae": minutiae_result,
                                "matching": matching_result
                            }
                            
                            response = agent_generate("agent", payload)
                            st.session_state.chat_history.append({
                                "question": user_question,
                                "response": response.get('text', 'No response generated'),
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            # Update session state with fresh results
                            st.session_state.analysis_results = {
                                'classification': classification_result,
                                'minutiae': minutiae_result,
                                'matching': matching_result
                            }
                            
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                            return
                            
                    st.success("✅ AI response ready!")
                elif not st.session_state.uploaded_file:
                    st.warning("⚠️ Please upload a fingerprint image first!")
                else:
                    st.warning("⚠️ Please enter a question!")
            
            # Clear chat history button
            if st.session_state.chat_history:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("#### Recent Conversations")
                with col2:
                    if st.button("🗑️ Clear History", help="Clear all conversation history"):
                        st.session_state.chat_history = []
                        st.rerun()
                for i, chat in enumerate(reversed(st.session_state.chat_history[-3:]), 1):
                    with st.container():
                        st.markdown(f"**Q{i}:** {chat['question']}")
                        
                        # Create expandable response section
                        response_text = chat['response']
                        if len(response_text) > 300:
                            # Show preview and expandable full text
                            preview = response_text[:300] + "..."
                            st.markdown(f"**A{i}:** {preview}")
                            
                            # Create expandable section for full response
                            with st.expander(f"📖 Read full AI response for Q{i}", expanded=False):
                                st.markdown(f'<div class="ai-response">{response_text}</div>', unsafe_allow_html=True)
                        else:
                            # Show full response if it's short enough
                            st.markdown(f'<div class="ai-response">**A{i}:** {response_text}</div>', unsafe_allow_html=True)
                        
                        st.markdown("---")  # Separator between conversations
    
    # ----- Main Content Area -----
    
    # Header with gradient text effect and responsive design
    st.markdown("""
    <h1 style="background: linear-gradient(90deg, #DC143C, #FF6B6B); 
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent;
               text-align: center;
               font-size: clamp(2rem, 6vw, 3rem);
               margin-bottom: clamp(1rem, 3vw, 2rem);
               line-height: 1.2;
               word-wrap: break-word;
               overflow-wrap: break-word;
               hyphens: auto;">
        Forensic Fingerprint Analysis System
    </h1>
    """, unsafe_allow_html=True)
    
    # Dynamic content based on current step
    if st.session_state.current_step == 1:
        # Step 1: Upload with modern design
        display_analysis_section(
            "Upload Fingerprint",
            "Begin by uploading a high-quality fingerprint image for analysis.",
            "📤"
        )
        
        with st.container():
            # Responsive column layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Select fingerprint image",
                    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                    help="For best results, use high-resolution scans (min 300 DPI)",
                    key="file_upload"
                )
                
                if uploaded_file is not None:
                    # Show preview and file info
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Preview", use_container_width=True)
                    
                    # Save file
                    file_path = save_uploaded_file(uploaded_file)
                    st.session_state.uploaded_file_path = file_path
                    st.session_state.uploaded_file = uploaded_file
                    
                    # File metadata in metrics - responsive layout
                    with col2:
                        st.markdown("### File Information")
                        create_metric_card("Format", uploaded_file.type.split('/')[-1].upper())
                        create_metric_card("Width", f"{image.size[0]}px")
                        create_metric_card("Height", f"{image.size[1]}px")
                        create_metric_card("Size", f"{len(uploaded_file.getvalue())/1024:.1f}KB")
                    
                    # Action buttons with responsive layout
                    st.markdown("---")
                    # Use responsive button layout
                    if st.button("✨ Begin Analysis", type="primary", use_container_width=True):
                        st.session_state.current_step = 2
                        st.rerun()
    
    elif st.session_state.current_step == 2:
        # Step 2: Classification with modern cards
        display_analysis_section(
            "Pattern Classification",
            "AI-powered fingerprint pattern recognition and analysis.",
            "🏷️"
        )
        
        if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
            # Responsive two-column layout
            col_img, col_results = st.columns([1, 1])
            
            with col_img:
                image = load_image(st.session_state.uploaded_file_path)
                st.image(image, caption="Input Fingerprint", use_container_width=True)
            
            with col_results:
                with st.spinner("🔍 Analyzing pattern..."):
                    classification_result = classify_fingerprint(st.session_state.uploaded_file_path)
                    st.session_state.analysis_results['classification'] = classification_result
                    save_analysis_log(classification_result, "classification", st.session_state.uploaded_file_path)
                
                # Results in modern cards
                st.success("✅ Classification Complete!")
                
                # Primary result
                pattern = classification_result.get('label', 'Unknown')
                confidence = classification_result.get('confidence', 0)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Pattern Type</div>
                    <div class="metric-value">{pattern}</div>
                    <div style="color: var(--color-text-muted);">Confidence: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution
                if 'probabilities' in classification_result:
                    st.markdown("#### Probability Distribution")
                    for pattern_type, prob in classification_result['probabilities'].items():
                        st.progress(prob, text=f"{pattern_type}: {prob:.1%}")
            
            # Navigation with better layout
            st.divider()
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.button("← Previous", use_container_width=True):
                    st.session_state.current_step = 1
                    st.rerun()
            with col3:
                if st.button("Next →", type="primary", use_container_width=True):
                    st.session_state.current_step = 3
                    st.rerun()
    
    elif st.session_state.current_step == 3:
        # Step 3: Minutiae Detection with modern visualization
        display_analysis_section(
            "Minutiae Extraction",
            "Detecting and analyzing fingerprint ridge characteristics.",
            "🔬"
        )
        
        if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
            # Process minutiae
            with st.spinner("🔬 Extracting minutiae points..."):
                minutiae_result = detect_minutiae(st.session_state.uploaded_file_path)
                st.session_state.analysis_results['minutiae'] = minutiae_result
                save_analysis_log(minutiae_result, "minutiae", st.session_state.uploaded_file_path)
            
            # Display in tabs for better organization
            tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📈 Statistics", "🔍 Details"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    image = load_image(st.session_state.uploaded_file_path)
                    st.image(image, caption="Original", use_container_width=True)
                with col2:
                    if minutiae_result.get('overlay_path') and os.path.exists(minutiae_result['overlay_path']):
                        overlay = Image.open(minutiae_result['overlay_path'])
                        st.image(overlay, caption="Minutiae Overlay", use_container_width=True)
            
            with tab2:
                # Metrics in a grid
                minutiae_list = minutiae_result.get('minutiae', [])
                ridge_endings = len([m for m in minutiae_list if m.get('type') == 'ending'])
                bifurcations = len([m for m in minutiae_list if m.get('type') == 'bifurcation'])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    create_metric_card("Total Points", len(minutiae_list))
                with col2:
                    create_metric_card("Ridge Endings", ridge_endings)
                with col3:
                    create_metric_card("Bifurcations", bifurcations)
                with col4:
                    quality = minutiae_result.get('quality_score', 0)
                    create_metric_card("Quality", f"{quality:.0%}")
                
                # Spoof detection alert
                if minutiae_result.get('spoof_suspected', False):
                    st.error("⚠️ **Security Alert**: Potential spoof detected")
                else:
                    st.success("✅ **Security Check**: Authentic fingerprint")
            
            with tab3:
                # Detailed minutiae information in expandable format
                with st.expander("View Detailed Minutiae Data"):
                    st.json(minutiae_result.get('minutiae', [])[:10])  # Show first 10
            
            # Navigation
            st.divider()
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.button("← Previous", use_container_width=True):
                    st.session_state.current_step = 2
                    st.rerun()
            with col3:
                if st.button("Next →", type="primary", use_container_width=True):
                    st.session_state.current_step = 4
                    st.rerun()
    
    elif st.session_state.current_step == 4:
        # Step 4: Identity Matching with modern security indicators
        display_analysis_section(
            "Identity Verification",
            "Matching fingerprint against secure database.",
            "🔐"
        )
        
        if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
            # Matching process
            with st.spinner("🔐 Verifying identity..."):
                minutiae_data = st.session_state.analysis_results.get('minutiae', {})
                match_result = match_minutiae(minutiae_data)
                st.session_state.analysis_results['matching'] = match_result
                save_analysis_log(match_result, "matching", st.session_state.uploaded_file_path)
            
            # Modern result display with visual feedback
            match_score = match_result.get('match_score', 0)
            matched_id = match_result.get('matched_id', 'No Match')
            access_granted = match_result.get('access_granted', False)
            spoof_flag = match_result.get('spoof_flag', False)
            
            # Large visual status indicator
            if access_granted and not spoof_flag:
                st.success("# ✅ ACCESS GRANTED")
                status_color = "--color-success"
            elif spoof_flag:
                st.error("# ⚠️ SECURITY ALERT - SPOOF DETECTED")
                status_color = "--color-danger"
            else:
                st.warning("# ❌ ACCESS DENIED")
                status_color = "--color-warning"
            
            # Detailed metrics in cards
            col1, col2, col3 = st.columns(3)
            with col1:
                create_metric_card(
                    "Match Score",
                    f"{match_score:.1%}",
                    delta=f"{match_score-0.5:.1%}" if match_score > 0.5 else None,
                    delta_color="normal" if match_score > 0.7 else "inverse"
                )
            with col2:
                create_metric_card("Identity", matched_id if matched_id != 'No Match' else "Unknown")
            with col3:
                create_metric_card("Security", "Clear" if not spoof_flag else "Alert")
            
            # Decision reasoning
            if 'reason' in match_result:
                with st.expander("View Decision Details"):
                    st.info(f"**Reasoning**: {match_result['reason']}")
                    st.json(match_result)
            
            # Final actions
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("← Previous", use_container_width=True):
                    st.session_state.current_step = 3
                    st.rerun()
            with col2:
                if st.button("📊 Generate Report", use_container_width=True):
                    st.download_button(
                        label="Download Full Report",
                        data=json.dumps(st.session_state.analysis_results, indent=2),
                        file_name=f"fingerprint_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            with col3:
                if st.button("🔄 New Analysis", type="primary", use_container_width=True):
                    st.session_state.current_step = 1
                    st.session_state.uploaded_file_path = None
                    st.session_state.uploaded_file = None
                    st.session_state.analysis_results = {}
                    st.rerun()
    
    # Footer with version info
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: var(--color-text-muted); font-size: var(--font-size-sm);">
        Forensic Fingerprint Analysis System v2.0 | Built with Streamlit | © 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
