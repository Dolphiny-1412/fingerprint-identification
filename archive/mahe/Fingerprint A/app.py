import streamlit as st
import os
import json
import uuid
from datetime import datetime
from PIL import Image
import io
import base64

# Import modules from other team members (Person B, C, D, E)
# These will be replaced with actual implementations
try:
    from PersonB.classifier import classify_fingerprint
    from PersonC.minutiae_detector import detect_minutiae
    from PersonD.matcher import match_fingerprint
    from PersonE.ai_assistant import agent_response
except ImportError:
    # Dummy implementations for development
    def classify_fingerprint(image):
        return {"label": "Whorl", "confidence": 0.92, "pattern_type": "Whorl"}
    
    def detect_minutiae(image):
        # Return processed image and minutiae list
        return image, [
            {"type": "ridge_ending", "x": 100, "y": 150, "angle": 45},
            {"type": "bifurcation", "x": 200, "y": 250, "angle": 90},
            {"type": "ridge_ending", "x": 300, "y": 350, "angle": 135}
        ]
    
    def match_fingerprint(image):
        return {
            "match_score": 0.87,
            "matched_id": "user_004",
            "access_granted": True,
            "confidence": 0.85
        }
    
    def agent_response(question, context):
        return f"Based on the analysis, {context.get('classification', {}).get('label', 'Unknown')} pattern was detected with {context.get('classification', {}).get('confidence', 0):.2%} confidence."

# Create necessary directories
def create_directories():
    """Create required directories for file storage"""
    directories = ['uploads', 'results', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# File handling utilities
def save_uploaded_file(uploaded_file):
    """Save uploaded file and return file path"""
    if uploaded_file is not None:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = str(uuid.uuid4())[:8]
        filename = f"fingerprint_{timestamp}_{file_id}.png"
        filepath = os.path.join("uploads", filename)
        
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return filepath
    return None

def save_processed_image(image, step_name, original_filename):
    """Save processed image with consistent naming"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    filename = f"{base_name}_{step_name}_{timestamp}.png"
    filepath = os.path.join("results", filename)
    
    if isinstance(image, Image.Image):
        image.save(filepath)
    else:
        # If it's a numpy array or other format, convert to PIL
        if hasattr(image, 'save'):
            image.save(filepath)
        else:
            Image.fromarray(image).save(filepath)
    
    return filepath

def save_analysis_log(data, step_name, original_filename):
    """Save analysis results as JSON log"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    filename = f"{base_name}_{step_name}_{timestamp}.json"
    filepath = os.path.join("logs", filename)
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "step": step_name,
        "original_file": original_filename,
        "results": data
    }
    
    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    return filepath

# UI Helper Functions
def display_image_with_info(image, title, info_text=None):
    """Display image with title and optional info"""
    st.subheader(title)
    if info_text:
        st.info(info_text)
    st.image(image, use_column_width=True)

def create_analysis_card(title, result, color="blue"):
    """Create a styled analysis result card"""
    with st.container():
        st.markdown(f"""
        <div style="
            border: 2px solid {color};
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: rgba(0,0,0,0.05);
        ">
            <h4 style="margin: 0; color: {color};">{title}</h4>
            <p style="margin: 5px 0;">{result}</p>
        </div>
        """, unsafe_allow_html=True)

# Main Streamlit App
def main():
    st.set_page_config(
        page_title="Fingerprint Analysis System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create directories
    create_directories()
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.title("🔍 Fingerprint Analysis")
        st.markdown("---")
        
        # Step indicator
        st.subheader("📋 Analysis Steps")
        steps = [
            "1. Upload Fingerprint",
            "2. Run Classification",
            "3. Detect Minutiae",
            "4. Match Identity",
            "5. AI Assistant"
        ]
        
        for i, step in enumerate(steps, 1):
            if i == st.session_state.current_step:
                st.markdown(f"**{step}** ✅")
            elif i < st.session_state.current_step:
                st.markdown(f"{step} ✅")
            else:
                st.markdown(f"{step} ⏳")
        
        st.markdown("---")
        
        # AI Assistant Chat
        st.subheader("🤖 AI Assistant")
        user_question = st.text_input("Ask about the analysis:", key="ai_question")
        if st.button("Ask AI") and user_question:
            if st.session_state.analysis_results:
                response = agent_response(user_question, st.session_state.analysis_results)
                st.session_state.chat_history.append({
                    "question": user_question,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
                st.success("AI Response received!")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for chat in st.session_state.chat_history[-3:]:  # Show last 3
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['response']}")
                st.markdown("---")
    
    # Main content area
    st.title("🔍 Forensic Fingerprint Analysis System")
    st.markdown("---")
    
    # Step 1: Upload Fingerprint
    if st.session_state.current_step == 1:
        st.header("📤 Step 1: Upload Fingerprint")
        st.markdown("Upload a fingerprint image to begin the analysis process.")
        
        uploaded_file = st.file_uploader(
            "Choose a fingerprint image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            display_image_with_info(
                image,
                "Uploaded Fingerprint",
                f"File: {uploaded_file.name} | Size: {image.size} | Mode: {image.mode}"
            )
            
            # Save file
            file_path = save_uploaded_file(uploaded_file)
            st.session_state.uploaded_file_path = file_path
            
            if st.button("✅ Proceed to Classification", type="primary"):
                st.session_state.current_step = 2
                st.rerun()
    
    # Step 2: Classification
    elif st.session_state.current_step == 2:
        st.header("🔍 Step 2: Fingerprint Classification")
        
        if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
            image = Image.open(st.session_state.uploaded_file_path)
            display_image_with_info(image, "Processing Fingerprint")
            
            with st.spinner("Running classification analysis..."):
                # Run classification
                classification_result = classify_fingerprint(image)
                st.session_state.analysis_results['classification'] = classification_result
                
                # Save results
                save_analysis_log(classification_result, "classification", st.session_state.uploaded_file_path)
            
            # Display results
            st.success("Classification completed!")
            
            col1, col2 = st.columns(2)
            with col1:
                create_analysis_card(
                    "Pattern Type",
                    classification_result.get('label', 'Unknown'),
                    "green"
                )
            with col2:
                confidence = classification_result.get('confidence', 0)
                create_analysis_card(
                    "Confidence",
                    f"{confidence:.2%}",
                    "blue"
                )
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("⬅️ Back to Upload"):
                    st.session_state.current_step = 1
                    st.rerun()
            with col3:
                if st.button("Detect Minutiae ➡️", type="primary"):
                    st.session_state.current_step = 3
                    st.rerun()
        else:
            st.error("No fingerprint uploaded. Please go back to Step 1.")
            if st.button("⬅️ Back to Upload"):
                st.session_state.current_step = 1
                st.rerun()
    
    # Step 3: Minutiae Detection
    elif st.session_state.current_step == 3:
        st.header("🔬 Step 3: Minutiae Detection")
        
        if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
            image = Image.open(st.session_state.uploaded_file_path)
            
            with st.spinner("Detecting minutiae points..."):
                # Run minutiae detection
                processed_image, minutiae_list = detect_minutiae(image)
                st.session_state.analysis_results['minutiae'] = {
                    'count': len(minutiae_list),
                    'points': minutiae_list
                }
                
                # Save processed image
                processed_path = save_processed_image(
                    processed_image, 
                    "minutiae_detection", 
                    st.session_state.uploaded_file_path
                )
                st.session_state.analysis_results['minutiae']['processed_image_path'] = processed_path
                
                # Save results
                save_analysis_log(st.session_state.analysis_results['minutiae'], "minutiae", st.session_state.uploaded_file_path)
            
            st.success("Minutiae detection completed!")
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                display_image_with_info(
                    image,
                    "Original Fingerprint"
                )
            with col2:
                display_image_with_info(
                    processed_image,
                    "Minutiae Detection Result",
                    f"Detected {len(minutiae_list)} minutiae points"
                )
            
            # Minutiae statistics
            ridge_endings = len([m for m in minutiae_list if m.get('type') == 'ridge_ending'])
            bifurcations = len([m for m in minutiae_list if m.get('type') == 'bifurcation'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                create_analysis_card("Total Minutiae", str(len(minutiae_list)), "purple")
            with col2:
                create_analysis_card("Ridge Endings", str(ridge_endings), "orange")
            with col3:
                create_analysis_card("Bifurcations", str(bifurcations), "red")
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("⬅️ Back to Classification"):
                    st.session_state.current_step = 2
                    st.rerun()
            with col3:
                if st.button("Match Identity ➡️", type="primary"):
                    st.session_state.current_step = 4
                    st.rerun()
        else:
            st.error("No fingerprint available. Please go back to Step 1.")
            if st.button("⬅️ Back to Upload"):
                st.session_state.current_step = 1
                st.rerun()
    
    # Step 4: Identity Matching
    elif st.session_state.current_step == 4:
        st.header("🔐 Step 4: Identity Matching")
        
        if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
            image = Image.open(st.session_state.uploaded_file_path)
            
            with st.spinner("Matching fingerprint against database..."):
                # Run matching
                match_result = match_fingerprint(image)
                st.session_state.analysis_results['matching'] = match_result
                
                # Save results
                save_analysis_log(match_result, "matching", st.session_state.uploaded_file_path)
            
            st.success("Identity matching completed!")
            
            # Display results
            display_image_with_info(image, "Fingerprint for Matching")
            
            # Match results
            col1, col2, col3 = st.columns(3)
            with col1:
                match_score = match_result.get('match_score', 0)
                create_analysis_card(
                    "Match Score",
                    f"{match_score:.2%}",
                    "green" if match_score > 0.8 else "orange" if match_score > 0.6 else "red"
                )
            with col2:
                matched_id = match_result.get('matched_id', 'No Match')
                create_analysis_card("Matched ID", matched_id, "blue")
            with col3:
                access_granted = match_result.get('access_granted', False)
                status = "✅ GRANTED" if access_granted else "❌ DENIED"
                color = "green" if access_granted else "red"
                create_analysis_card("Access Decision", status, color)
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("⬅️ Back to Minutiae"):
                    st.session_state.current_step = 3
                    st.rerun()
            with col3:
                if st.button("AI Assistant ➡️", type="primary"):
                    st.session_state.current_step = 5
                    st.rerun()
        else:
            st.error("No fingerprint available. Please go back to Step 1.")
            if st.button("⬅️ Back to Upload"):
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
        
        # Display summary
        if st.session_state.analysis_results:
            st.subheader("📊 Analysis Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                if 'classification' in st.session_state.analysis_results:
                    st.markdown("**Classification:**")
                    st.json(st.session_state.analysis_results['classification'])
                
                if 'minutiae' in st.session_state.analysis_results:
                    st.markdown("**Minutiae Detection:**")
                    st.json({
                        'count': st.session_state.analysis_results['minutiae']['count'],
                        'processed_image': st.session_state.analysis_results['minutiae'].get('processed_image_path', 'N/A')
                    })
            
            with col2:
                if 'matching' in st.session_state.analysis_results:
                    st.markdown("**Identity Matching:**")
                    st.json(st.session_state.analysis_results['matching'])
        
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
                if os.path.exists("uploads"):
                    st.markdown("**Uploaded Files:**")
                    for file in os.listdir("uploads"):
                        st.text(f"📁 uploads/{file}")
                
                # Show results
                if os.path.exists("results"):
                    st.markdown("**Processed Results:**")
                    for file in os.listdir("results"):
                        st.text(f"📊 results/{file}")
                
                # Show logs
                if os.path.exists("logs"):
                    st.markdown("**Analysis Logs:**")
                    for file in os.listdir("logs"):
                        st.text(f"📝 logs/{file}")

if __name__ == "__main__":
    main()
