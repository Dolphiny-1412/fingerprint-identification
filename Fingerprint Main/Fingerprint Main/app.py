import streamlit as st
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the fingerprint analysis modules
try:
    from FingerprintB.model import classify_fingerprint
    from FingerprintC.detector import detect_minutiae
    from FingerprintD.matcher import match_fingerprint
    from FingerprintE.assistant import agent_response
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("Please ensure all Fingerprint modules (A-E) are properly set up.")

st.set_page_config(page_title="Fingerprint Forensics Toolkit", layout="wide")

def main():
    st.title("🔍 AI-Guided Fingerprint Forensics Toolkit")
    st.markdown("---")

    # Sidebar for navigation
    st.sidebar.title("🔧 Analysis Tools")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["Classification", "Minutiae Detection", "Identity Matching", "AI Assistant"]
    )

    # File uploader
    uploaded = st.file_uploader("Upload a fingerprint image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

    if uploaded:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded, caption="Uploaded Fingerprint", use_column_width=True)
        
        with col2:
            st.subheader("📊 Image Information")
            st.write(f"**File name:** {uploaded.name}")
            st.write(f"**File size:** {uploaded.size} bytes")
            st.write(f"**File type:** {uploaded.type}")

        st.markdown("---")

        # Analysis section based on selection
        if analysis_type == "Classification":
            st.subheader("🔍 Pattern Classification")
            if st.button("Run Classification", type="primary"):
                with st.spinner("Analyzing fingerprint pattern..."):
                    try:
                        result_b = classify_fingerprint(uploaded)
                        st.success(f"📌 **Pattern Type:** {result_b['label']}")
                        st.metric("Confidence", f"{result_b['confidence']*100:.2f}%")
                        
                        # Display pattern details
                        st.subheader("📋 Pattern Details")
                        st.json(result_b)
                    except Exception as e:
                        st.error(f"❌ Classification failed: {e}")

        elif analysis_type == "Minutiae Detection":
            st.subheader("🔍 Minutiae Detection")
            if st.button("Detect Minutiae", type="primary"):
                with st.spinner("Detecting minutiae points..."):
                    try:
                        minutiae_img, minutiae_data = detect_minutiae(uploaded)
                        
                        # Display results in columns
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(minutiae_img, caption="Minutiae Overlay", use_column_width=True)
                        
                        with col2:
                            st.subheader("📊 Minutiae Statistics")
                            st.metric("Total Minutiae", len(minutiae_data.get('minutiae', [])))
                            st.metric("Ridge Endings", len([m for m in minutiae_data.get('minutiae', []) if m.get('type') == 'ridge_ending']))
                            st.metric("Bifurcations", len([m for m in minutiae_data.get('minutiae', []) if m.get('type') == 'bifurcation']))
                        
                        st.subheader("📋 Detailed Minutiae Data")
                        st.json(minutiae_data)
                    except Exception as e:
                        st.error(f"❌ Minutiae detection failed: {e}")

        elif analysis_type == "Identity Matching":
            st.subheader("🔍 Identity Matching")
            if st.button("Match Identity", type="primary"):
                with st.spinner("Matching fingerprint identity..."):
                    try:
                        match_result = match_fingerprint(uploaded)
                        
                        # Display match results
                        if match_result.get('match_found'):
                            st.success("✅ **Match Found!**")
                            st.metric("Match Score", f"{match_result.get('confidence', 0)*100:.2f}%")
                        else:
                            st.warning("❌ **No Match Found**")
                        
                        st.subheader("📋 Match Details")
                        st.json(match_result)
                    except Exception as e:
                        st.error(f"❌ Identity matching failed: {e}")

        elif analysis_type == "AI Assistant":
            st.subheader("🤖 AI Forensic Assistant")
            
            # Chat interface
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask the forensic AI assistant about fingerprint analysis..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("AI is thinking..."):
                        try:
                            reply = agent_response(prompt, context={"uploaded_image": uploaded})
                            st.markdown(reply)
                            st.session_state.messages.append({"role": "assistant", "content": reply})
                        except Exception as e:
                            error_msg = f"❌ AI Assistant error: {e}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

    else:
        st.info("👆 Please upload a fingerprint image to begin analysis.")
        
        # Show sample workflow
        st.subheader("📋 Sample Workflow")
        st.markdown("""
        1. **Upload** a fingerprint image (PNG, JPG, BMP, TIFF)
        2. **Choose** analysis type from the sidebar
        3. **Run** the selected analysis
        4. **Review** results and insights
        5. **Ask** the AI assistant for forensic guidance
        """)

if __name__ == "__main__":
    main()
