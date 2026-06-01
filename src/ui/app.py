"""
Main Fingerprint Analysis System - Streamlit UI
===============================================

Modernized 4-step wizard with enhanced design system:
Upload → Classify → Minutiae → Match + AI Agent

Features:
- Comprehensive design system with tokens
- Responsive layouts with containers and columns
- Performance optimizations with caching
- Accessibility support (WCAG compliant)
- Enhanced UI components
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
from typing import Dict, Any, Optional
import time

# Add the src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import our modules
from core.classifier import classify_fingerprint
from core.minutiae_detector import detect_minutiae
from core.fingerprint_matcher import match_minutiae, seed_database, get_matcher
from core.ai_assistant import agent_generate

# Import design system
from design_system import (
    init_design_system,
    DesignTokens,
    render_custom_card,
    render_stat_card,
    render_progress_card,
    create_download_button,
    render_timeline_step,
    render_breadcrumb
)

# ============================================================================
# PERFORMANCE OPTIMIZATIONS
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_image_cached(filepath: str) -> Image.Image:
    """Load and cache images for better performance"""
    return Image.open(filepath)

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def process_classification(filepath: str) -> Dict[str, Any]:
    """Cache classification results"""
    return classify_fingerprint(filepath)

@st.cache_data(ttl=1800)
def process_minutiae(filepath: str) -> Dict[str, Any]:
    """Cache minutiae detection results"""
    return detect_minutiae(filepath)

@st.cache_resource
def get_cached_matcher():
    """Cache the matcher instance"""
    return get_matcher()

def measure_performance(func):
    """Decorator to measure function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        st.sidebar.caption(f"⚡ Processing time: {end_time - start_time:.2f}s")
        return result
    return wrapper

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fingerprint Analysis System | M.A.H.E",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/fingerprint-analysis',
        'Report a bug': 'https://github.com/yourusername/fingerprint-analysis/issues',
        'About': "# Fingerprint Analysis System\nAdvanced biometric analysis with AI assistance"
    }
)

# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

@st.cache_resource
def create_directories():
    """Create required directories for file storage with proper structure"""
    directories = [
        'data/uploads',      # Raw uploaded files
        'data/processed',    # Processed images
        'data/derived',      # Analysis results
        'data/fingerprints_db',  # Database
        'logs',              # System logs
        'models',            # ML models
        'reports',           # Generated reports
        'cache'              # Cache directory
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return True

# ============================================================================
# FILE HANDLING UTILITIES
# ============================================================================

@measure_performance
def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file with unique naming and metadata"""
    if uploaded_file is not None:
        # Generate unique filename with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = str(uuid.uuid4())[:8]
        extension = os.path.splitext(uploaded_file.name)[1] or '.png'
        filename = f"fingerprint_{timestamp}_{file_id}{extension}"
        filepath = os.path.join("data/uploads", filename)
        
        # Save file
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Save metadata
        metadata = {
            "original_name": uploaded_file.name,
            "upload_time": datetime.now().isoformat(),
            "file_size": uploaded_file.size,
            "file_type": uploaded_file.type,
            "unique_id": file_id
        }
        
        metadata_path = filepath.replace(extension, '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath
    return None

@st.cache_data
def save_analysis_log(data: Dict[str, Any], step_name: str, original_filename: str) -> str:
    """Save analysis results as JSON log with caching"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    filename = f"{base_name}_{step_name}_{timestamp}.json"
    filepath = os.path.join("logs", filename)
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "step": step_name,
        "original_file": original_filename,
        "results": data,
        "version": "2.0",  # Version tracking
        "system": "M.A.H.E Fingerprint Analysis"
    }
    
    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    return filepath

# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def display_image_with_metadata(image: Image.Image, title: str, metadata: Optional[Dict] = None) -> None:
    """Display image with enhanced metadata and responsive layout"""
    with st.container():
        st.subheader(title)
        
        if metadata:
            cols = st.columns(4)
            for i, (key, value) in enumerate(metadata.items()):
                if i < 4:  # Display first 4 metadata items
                    cols[i].metric(key, value)
        
        # Display image with proper container
        st.image(image, use_container_width=True, caption=title)

def create_step_header(step_number: int, title: str, description: str) -> None:
    """Create a consistent step header with progress indicator"""
    st.markdown(f"""<div id="main-content" aria-label="Step {step_number}">
        <h2>Step {step_number}: {title}</h2>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">{description}</p>
    </div>""", unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application with enhanced design system"""
    
    # Initialize design system
    init_design_system()
    
    # Create necessary directories
    create_directories()
    
    # Initialize session state with comprehensive tracking
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing_times' not in st.session_state:
        st.session_state.processing_times = {}
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'theme': 'dark',
            'show_technical_details': False,
            'auto_proceed': False
        }
    
    # ============ SIDEBAR WITH ENHANCED UI ============
    with st.sidebar:
        st.title("🔍 M.A.H.E Analysis")
        st.caption("Advanced Fingerprint Recognition System")
        
        # Progress Overview
        with st.container():
            total_steps = 5
            current = st.session_state.current_step
            render_progress_card(
                "Overall Progress",
                current,
                total_steps,
                show_percentage=True
            )
        
        st.markdown("---")
        
        # Step Timeline
        st.subheader("📋 Analysis Pipeline")
        
        steps = [
            ("Upload Fingerprint", "Load biometric image"),
            ("Classification", "Identify pattern type"),
            ("Minutiae Detection", "Extract key points"),
            ("Identity Matching", "Database comparison"),
            ("AI Analysis", "Generate insights")
        ]
        
        for i, (title, desc) in enumerate(steps, 1):
            status = "completed" if i < st.session_state.current_step else \
                    "active" if i == st.session_state.current_step else "pending"
            render_timeline_step(i, title, status, desc)
        
        st.markdown("---")
        
        # Database Status with Enhanced Display
        with st.expander("🗜️ Database Status", expanded=True):
            try:
                matcher = get_cached_matcher()
                stats = matcher.get_database_statistics()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Entries", stats.get('total_entries', 0))
                with col2:
                    st.metric("Queries", stats.get('total_accesses', 0))
                
                if stats.get('total_entries', 0) == 0:
                    if st.button("🌱 Initialize Database", use_container_width=True):
                        with st.spinner("Initializing database..."):
                            seed_database()
                        st.success("Database initialized!")
                        st.rerun()
            except Exception as e:
                st.error(f"Database connection issue: {str(e)[:50]}...")
        
        # Settings & Preferences
        with st.expander("⚙️ Settings", expanded=False):
            st.session_state.user_preferences['show_technical_details'] = \
                st.checkbox("Show Technical Details", 
                           value=st.session_state.user_preferences['show_technical_details'])
            
            st.session_state.user_preferences['auto_proceed'] = \
                st.checkbox("Auto-proceed to next step",
                           value=st.session_state.user_preferences['auto_proceed'])
        
        # Performance Metrics
        if st.session_state.processing_times:
            with st.expander("⚡ Performance", expanded=False):
                for step, time_taken in st.session_state.processing_times.items():
                    st.caption(f"{step}: {time_taken:.2f}s")
    
    # ============ MAIN CONTENT AREA ============
    
    # Header with breadcrumb
    render_breadcrumb(["Home", "Fingerprint Analysis", f"Step {st.session_state.current_step}"])
    
    st.title("🔍 Forensic Fingerprint Analysis System")
    st.markdown("<p style='color: var(--color-text-secondary); font-size: 1.1rem;'>Advanced biometric analysis powered by AI</p>", 
                unsafe_allow_html=True)
    
    # Step 1: Upload Fingerprint with Enhanced UI
    if st.session_state.current_step == 1:
        create_step_header(1, "Upload Fingerprint", 
                          "Begin by uploading a high-quality fingerprint image for analysis")
        
        # Use tabs for different upload methods
        tab1, tab2, tab3 = st.tabs(["Upload File", "Camera Capture", "Sample Images"])
        
        with tab1:
            # Enhanced file uploader with drag-drop zone
            uploaded_file = st.file_uploader(
                "Drag and drop or browse files",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
                help="Supported formats: PNG, JPG, JPEG, BMP, TIFF, WebP. Maximum size: 200MB",
                label_visibility="visible"
            )
            
            if uploaded_file is not None:
                # Process and display with performance tracking
                start_time = time.time()
                
                # Create responsive columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    image = Image.open(uploaded_file)
                    
                    # Display with enhanced metadata
                    metadata = {
                        "Format": image.format,
                        "Size": f"{image.width}x{image.height}",
                        "Mode": image.mode,
                        "File Size": f"{uploaded_file.size / 1024:.1f} KB"
                    }
                    display_image_with_metadata(image, "Uploaded Fingerprint", metadata)
                
                with col2:
                    # Quality check indicators
                    st.markdown("### Quality Check")
                    
                    # Simple quality metrics (placeholder - enhance with actual checks)
                    quality_checks = {
                        "Resolution": image.width >= 300 and image.height >= 300,
                        "Format": image.format in ['PNG', 'JPEG', 'BMP'],
                        "Size": uploaded_file.size < 200 * 1024 * 1024,  # < 200MB
                        "Color Mode": image.mode in ['L', 'RGB', 'RGBA']
                    }
                    
                    for check, passed in quality_checks.items():
                        icon = "✅" if passed else "⚠️"
                        color = DesignTokens.COLORS['success'] if passed else DesignTokens.COLORS['warning']
                        st.markdown(f"<p style='color: {color};'>{icon} {check}</p>", 
                                  unsafe_allow_html=True)
                    
                    # Quality score
                    quality_score = sum(quality_checks.values()) / len(quality_checks) * 100
                    render_stat_card("Quality Score", f"{quality_score:.0f}%")
                
                # Save file with performance tracking
                file_path = save_uploaded_file(uploaded_file)
                st.session_state.uploaded_file_path = file_path
                
                # Track processing time
                processing_time = time.time() - start_time
                st.session_state.processing_times['upload'] = processing_time
                
                # Action buttons with better layout
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Proceed to Classification", 
                               type="primary", 
                               use_container_width=True):
                        st.session_state.current_step = 2
                        st.rerun()
                    
                    if st.button("🔄 Upload Different Image", 
                               use_container_width=True):
                        st.session_state.uploaded_file_path = None
                        st.rerun()
        
        with tab2:
            st.info("📷 Camera capture feature coming soon!")
            st.markdown("This feature will allow direct fingerprint capture from connected devices.")
        
        with tab3:
            st.info("🖼️ Sample images feature coming soon!")
            st.markdown("Pre-loaded sample fingerprints will be available for testing.")
    
    # Step 2: Classification with Enhanced UI
    elif st.session_state.current_step == 2:
        create_step_header(2, "Fingerprint Classification",
                          "Analyze the fingerprint pattern type using advanced ML models")
        
        if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
            # Performance tracking
            start_time = time.time()
            
            # Layout with columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = load_image_cached(st.session_state.uploaded_file_path)
                st.image(image, caption="Fingerprint Under Analysis", use_container_width=True)
            
            with col2:
                with st.container():
                    st.markdown("### 🤖 Processing")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate progress updates
                    for i in range(101):
                        progress_bar.progress(i)
                        if i < 30:
                            status_text.text("Preprocessing image...")
                        elif i < 70:
                            status_text.text("Running neural network...")
                        else:
                            status_text.text("Analyzing results...")
                        time.sleep(0.01)  # Simulated delay
                    
                    # Run actual classification with caching
                    classification_result = process_classification(st.session_state.uploaded_file_path)
                    st.session_state.analysis_results['classification'] = classification_result
                    
                    # Save results
                    save_analysis_log(classification_result, "classification", 
                                    st.session_state.uploaded_file_path)
                    
                    status_text.text("Analysis complete!")
                    progress_bar.progress(100)
            
            # Track time
            processing_time = time.time() - start_time
            st.session_state.processing_times['classification'] = processing_time
            
            # Display results with enhanced cards
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Main results in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                render_stat_card(
                    "Pattern Type",
                    classification_result.get('label', 'Unknown'),
                    delta="Primary classification"
                )
            
            with col2:
                confidence = classification_result.get('confidence', 0)
                render_stat_card(
                    "Confidence",
                    f"{confidence:.1%}",
                    delta="Model certainty",
                    delta_color=DesignTokens.COLORS['success'] if confidence > 0.8 else DesignTokens.COLORS['warning']
                )
            
            with col3:
                render_stat_card(
                    "Processing Time",
                    f"{processing_time:.2f}s",
                    delta="⚡ Fast processing"
                )
            
            with col4:
                render_stat_card(
                    "Model Version",
                    "v2.0",
                    delta="Latest model"
                )
            
            # Detailed probabilities in expandable section
            if 'probabilities' in classification_result:
                with st.expander("📊 Detailed Classification Probabilities", expanded=True):
                    prob_data = classification_result['probabilities']
                    
                    # Sort by probability
                    sorted_probs = sorted(prob_data.items(), key=lambda x: x[1], reverse=True)
                    
                    for pattern, prob in sorted_probs:
                        # Create visual probability bar
                        st.markdown(f"**{pattern}**")
                        st.progress(prob)
                        st.caption(f"{prob:.2%} confidence")
            
            # Technical details (if enabled)
            if st.session_state.user_preferences['show_technical_details']:
                with st.expander("🔬 Technical Details", expanded=False):
                    st.json(classification_result)
            
            # Download and navigation
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("⬅ Previous Step", use_container_width=True):
                    st.session_state.current_step = 1
                    st.rerun()
            
            with col2:
                create_download_button(
                    classification_result,
                    f"classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "Download Results"
                )
            
            with col3:
                if st.button("Next Step ➡", type="primary", use_container_width=True):
                    st.session_state.current_step = 3
                    st.rerun()
            
            # Auto-proceed if enabled
            if st.session_state.user_preferences['auto_proceed']:
                time.sleep(2)
                st.session_state.current_step = 3
                st.rerun()
        else:
            render_custom_card(
                "No Data Available",
                "Please return to Step 1 to upload a fingerprint image.",
                icon="⚠️",
                color_accent=DesignTokens.COLORS['warning']
            )
            if st.button("⬅ Back to Upload", type="primary"):
                st.session_state.current_step = 1
                st.rerun()
    
    # Step 3: Minutiae Detection with Enhanced UI
    elif st.session_state.current_step == 3:
        create_step_header(3, "Minutiae Detection",
                          "Extract and analyze fingerprint minutiae points using advanced algorithms")
        
        if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
            # Performance tracking
            start_time = time.time()
            
            # Load image
            image = load_image_cached(st.session_state.uploaded_file_path)
            
            # Processing with progress
            with st.spinner("🔬 Detecting minutiae points..."):
                # Run minutiae detection with caching
                minutiae_result = process_minutiae(st.session_state.uploaded_file_path)
                st.session_state.analysis_results['minutiae'] = minutiae_result
                
                # Save results
                save_analysis_log(minutiae_result, "minutiae", st.session_state.uploaded_file_path)
            
            # Track time
            processing_time = time.time() - start_time
            st.session_state.processing_times['minutiae'] = processing_time
            
            st.success(f"✅ Minutiae detection completed in {processing_time:.2f}s!")
            
            # Display results with enhanced layout
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Fingerprint", use_container_width=True)
            
            with col2:
                if minutiae_result.get('overlay_path') and os.path.exists(minutiae_result['overlay_path']):
                    overlay_image = load_image_cached(minutiae_result['overlay_path'])
                    st.image(overlay_image, caption="Minutiae Detection Overlay", use_container_width=True)
                else:
                    st.info("🖼️ Overlay visualization not available")
            
            # Minutiae statistics with enhanced cards
            st.markdown("---")
            st.subheader("📊 Detection Results")
            
            minutiae_list = minutiae_result.get('minutiae', [])
            ridge_endings = len([m for m in minutiae_list if m.get('type') == 'ending'])
            bifurcations = len([m for m in minutiae_list if m.get('type') == 'bifurcation'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_stat_card(
                    "Total Minutiae",
                    str(len(minutiae_list)),
                    delta="Key points found"
                )
            with col2:
                render_stat_card(
                    "Ridge Endings",
                    str(ridge_endings),
                    delta="Type 1 features"
                )
            with col3:
                render_stat_card(
                    "Bifurcations",
                    str(bifurcations),
                    delta="Type 2 features"
                )
            with col4:
                quality = minutiae_result.get('quality_score', 0)
                render_stat_card(
                    "Quality Score",
                    f"{quality:.0%}",
                    delta="Image quality",
                    delta_color=DesignTokens.COLORS['success'] if quality > 0.7 else DesignTokens.COLORS['warning']
                )
            
            # Spoof detection with custom card
            st.markdown("---")
            if minutiae_result.get('spoof_suspected', False):
                render_custom_card(
                    "Security Alert",
                    "Potential spoof indicators detected. This fingerprint may be artificial or altered.",
                    icon="⚠️",
                    color_accent=DesignTokens.COLORS['warning']
                )
            else:
                render_custom_card(
                    "Security Check Passed",
                    "No spoof indicators detected. Fingerprint appears to be authentic.",
                    icon="✅",
                    color_accent=DesignTokens.COLORS['success']
                )
            
            # Technical details
            if st.session_state.user_preferences['show_technical_details']:
                with st.expander("🔬 Technical Minutiae Details", expanded=False):
                    st.json(minutiae_result)
            
            # Navigation and download
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("⬅ Previous Step", use_container_width=True):
                    st.session_state.current_step = 2
                    st.rerun()
            
            with col2:
                create_download_button(
                    minutiae_result,
                    f"minutiae_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "Download Minutiae Data"
                )
            
            with col3:
                if st.button("Next Step ➡", type="primary", use_container_width=True):
                    st.session_state.current_step = 4
                    st.rerun()
        else:
            render_custom_card(
                "No Data Available",
                "Please return to Step 1 to upload a fingerprint image.",
                icon="⚠️",
                color_accent=DesignTokens.COLORS['warning']
            )
            if st.button("⬅ Back to Upload", type="primary"):
                st.session_state.current_step = 1
                st.rerun()
    
    # Step 4: Identity Matching with Enhanced UI
    elif st.session_state.current_step == 4:
        create_step_header(4, "Identity Matching",
                          "Compare fingerprint against database for identity verification")
        
        if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
            # Performance tracking
            start_time = time.time()
            
            # Load image
            image = load_image_cached(st.session_state.uploaded_file_path)
            
            # Display image with progress indicator
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Fingerprint for Matching", use_container_width=True)
            
            with col2:
                with st.spinner("🔍 Matching against database..."):
                    # Get minutiae data from previous step
                    minutiae_data = st.session_state.analysis_results.get('minutiae', {})
                    
                    # Run matching
                    match_result = match_minutiae(minutiae_data)
                    st.session_state.analysis_results['matching'] = match_result
                    
                    # Save results
                    save_analysis_log(match_result, "matching", st.session_state.uploaded_file_path)
            
            # Track time
            processing_time = time.time() - start_time
            st.session_state.processing_times['matching'] = processing_time
            
            st.success(f"✅ Identity matching completed in {processing_time:.2f}s!")
            
            # Display match results with enhanced cards
            st.markdown("---")
            st.subheader("🎆 Matching Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                match_score = match_result.get('match_score', 0)
                color = DesignTokens.COLORS['success'] if match_score > 0.7 else \
                       DesignTokens.COLORS['warning'] if match_score > 0.4 else \
                       DesignTokens.COLORS['error']
                render_stat_card(
                    "Match Score",
                    f"{match_score:.0%}",
                    delta="Similarity measure",
                    delta_color=color
                )
            
            with col2:
                matched_id = match_result.get('matched_id', 'No Match')
                render_stat_card(
                    "Matched ID",
                    matched_id,
                    delta="Database result"
                )
            
            with col3:
                access_granted = match_result.get('access_granted', False)
                status = "GRANTED" if access_granted else "DENIED"
                color = DesignTokens.COLORS['success'] if access_granted else DesignTokens.COLORS['error']
                render_stat_card(
                    "Access Status",
                    status,
                    delta="Security decision",
                    delta_color=color
                )
            
            with col4:
                spoof_flag = match_result.get('spoof_flag', False)
                spoof_status = "DETECTED" if spoof_flag else "CLEAR"
                color = DesignTokens.COLORS['error'] if spoof_flag else DesignTokens.COLORS['success']
                render_stat_card(
                    "Spoof Check",
                    spoof_status,
                    delta="Anti-fraud",
                    delta_color=color
                )
            
            # Detailed reason with custom card
            if 'reason' in match_result:
                st.markdown("---")
                render_custom_card(
                    "Decision Reasoning",
                    match_result['reason'],
                    icon="📝",
                    color_accent=DesignTokens.COLORS['info']
                )
            
            # Technical details
            if st.session_state.user_preferences['show_technical_details']:
                with st.expander("🔬 Technical Matching Details", expanded=False):
                    st.json(match_result)
            
            # Navigation and download
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("⬅ Previous Step", use_container_width=True):
                    st.session_state.current_step = 3
                    st.rerun()
            
            with col2:
                create_download_button(
                    match_result,
                    f"matching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "Download Match Results"
                )
            
            with col3:
                if st.button("Next Step ➡", type="primary", use_container_width=True):
                    st.session_state.current_step = 5
                    st.rerun()
        else:
            render_custom_card(
                "No Data Available",
                "Please return to Step 1 to upload a fingerprint image.",
                icon="⚠️",
                color_accent=DesignTokens.COLORS['warning']
            )
            if st.button("⬅ Back to Upload", type="primary"):
                st.session_state.current_step = 1
                st.rerun()
    
    # Step 5: AI Assistant
    elif st.session_state.current_step == 5:
        st.header("🤖 Step 5: AI Assistant")
        
        st.markdown("""
        ### Analysis Complete! 🎉
        
        All fingerprint analysis steps have been completed. You can now:
        - Ask the AI Assistant questions about the analysis
        - Review all results in the sidebar
        - Start a new analysis
        """)
        
        # Display comprehensive analysis
        if st.session_state.analysis_results:
            st.subheader("📊 Complete Analysis Summary")
            
            # Create comprehensive payload for AI
            payload = {
                "classification": st.session_state.analysis_results.get('classification', {}),
                "minutiae": st.session_state.analysis_results.get('minutiae', {}),
                "matching": st.session_state.analysis_results.get('matching', {})
            }
            
            # Generate comprehensive AI explanation
            with st.spinner("Generating comprehensive analysis..."):
                comprehensive_result = agent_generate("agent", payload)
            
            st.subheader("🧠 AI Analysis Report")
            st.write(comprehensive_result.get('text', 'No analysis available'))
            
            # Show facts
            if 'facts' in comprehensive_result:
                st.subheader("📋 Analysis Facts")
                facts = comprehensive_result['facts']
                for key, value in facts.items():
                    if isinstance(value, dict):
                        st.write(f"**{key}:**")
                        st.json(value)
                    else:
                        st.write(f"**{key}:** {value}")
            
            # Download comprehensive results
            create_download_button(
                st.session_state.analysis_results,
                f"complete_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "Download Complete Analysis"
            )
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Back to Matching"):
                st.session_state.current_step = 4
                st.rerun()
        with col2:
            if st.button("🔄 New Analysis"):
                # Reset session state
                st.session_state.current_step = 1
                st.session_state.uploaded_file_path = None
                st.session_state.analysis_results = {}
                st.session_state.chat_history = []
                st.rerun()
        with col3:
            if st.button("📁 View Files"):
                st.subheader("Generated Files")
                
                # Show uploaded files
                if os.path.exists("data/uploads"):
                    st.markdown("**Uploaded Files:**")
                    for file in os.listdir("data/uploads"):
                        st.text(f"📁 data/uploads/{file}")
                
                # Show derived files
                if os.path.exists("data/derived"):
                    st.markdown("**Processed Results:**")
                    for file in os.listdir("data/derived"):
                        st.text(f"📊 data/derived/{file}")
                
                # Show logs
                if os.path.exists("logs"):
                    st.markdown("**Analysis Logs:**")
                    for file in os.listdir("logs"):
                        st.text(f"📝 logs/{file}")

if __name__ == "__main__":
    main()
