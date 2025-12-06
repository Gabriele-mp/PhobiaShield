import streamlit as st
import tempfile
import os
import sys

# Add root to path to find src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import REAL inference engine
try:
    from src.inference.video_processor import PhobiaVideoProcessor
except ImportError:
    st.error("Error: Could not import inference engine. Check PYTHONPATH.")
    st.stop()

st.set_page_config(page_title="PhobiaShield", page_icon="🛡️", layout="wide")

st.title("🛡️ PhobiaShield: Intelligent Protection System")
st.markdown("### Real-time Object Detection & Blurring Engine")

# SIDEBAR CONFIG
st.sidebar.header("🔧 Engine Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
debug_mode = st.sidebar.checkbox("Debug Mode (Show Bounding Boxes)", value=True)

st.sidebar.markdown("---")
st.sidebar.info("System Status: **ONLINE**\nModel: PhobiaNet-Tiny (v1.0)")

# MAIN AREA
uploaded_file = st.file_uploader("📂 Upload Video Source (MP4)", type=['mp4', 'mov'])

if uploaded_file:
    # Temporary File Management
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    input_path = tfile.name
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Input")
        st.video(input_path)

    # Action Button
    if st.sidebar.button("RUN PROTECTION ENGINE", type="primary"):
        output_filename = "processed_output.mp4"
        output_dir = tempfile.gettempdir()
        output_path = os.path.join(output_dir, output_filename)

        # Initialize Processor
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        processor = PhobiaVideoProcessor(output_dir=output_dir)
        
        status_text.text("⏳ Initializing Neural Network...")
        
        try:
            # REAL EXECUTION
            processor.process_video(
                input_path=input_path,
                output_name=output_filename,
                conf_threshold=conf_threshold,
                debug=debug_mode
            )
            
            progress_bar.progress(100)
            status_text.success("✅ Processing Complete!")
            
            with col2:
                st.subheader("Protected Output")
                if os.path.exists(output_path):
                    # BINARY OPEN (More robust for browsers)
                    with open(output_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes, format="video/mp4")
                    
                    # Download button reads the same file
                    with open(output_path, "rb") as f:
                        st.download_button("📥 Download Protected Video", f, file_name="phobiashield_safe.mp4")
                else:
                    st.error("Output file generation failed.")
                    
        except Exception as e:
            st.error(f"Runtime Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

else:
    st.info("👆 Upload a video trailer to start the protection engine.")