import streamlit as st
import tempfile
import os
import sys
import time

# Add project root to sys.path to allow importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.inference.video_processor import PhobiaVideoProcessor

# PAGE CONFIGURATION
st.set_page_config(
    page_title="PhobiaShield Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# HEADER
st.title("🛡️ PhobiaShield: Intelligent Phobia Protection")
st.markdown("""
**Automated Content Moderation System using Computer Vision.**
Upload a video trailer to automatically detect and blur phobia triggers.
""")

# SIDEBAR CONTROLS
st.sidebar.header("Configuration")

# Simulation mode toggle
simulate_mode = st.sidebar.checkbox(
    "Simulation Mode (Monte Carlo)", 
    value=True,
    help="If enabled, generates stochastic detections for testing without the AI model."
)

# Debug mode toggle
debug_mode = st.sidebar.checkbox(
    "Debug Mode (Show Boxes)", 
    value=True,
    help="Draws bounding boxes and labels around detected objects."
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Supported Classes:**\n"
    "- 🕷️ Spider\n"
    "- 🦈 Shark\n"
    "- 🤡 Clown\n"
    "- 🩸 Blood\n"
    "- 💉 Needle"
)

# MAIN INTERFACE

uploaded_file = st.file_uploader("Choose a video file...", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    # Display original video
    st.subheader("Original Video")
    st.video(video_path)
    
    # Processing button
    if st.button("Start protection filter", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize processor
        output_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")
        processor = PhobiaVideoProcessor(output_dir=tempfile.gettempdir())
        
        status_text.text("Initializing neural engine...")
        
        # Process Video
        # Note: We need to adapt the processor to yield progress updates for the UI
        # For now we run it as a block. Ideally process_video should be a generator
        
        try:
            with st.spinner('Processing video frames... This may take a moment.'):
                # We call the processor
                processor.process_video(
                    input_path=video_path,
                    output_name="processed_output.mp4",
                    simulate=simulate_mode,
                    debug=debug_mode
                )
            
            # Display result
            st.success("Processing complete!")
            st.subheader("Protected video")
            
            # Streamlit sometimes has issues playing local mp4s directly if codec is weird
            # We read the bytes and serve them
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
                st.video(video_bytes)
                
            # Cleanup temp files
            # os.unlink(video_path) 
            
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

else:
    # Placeholder / Instructions
    st.info("Please upload a video file to begin.")

# FOOTER
st.markdown("---")
st.markdown("*PhobiaShield Project - December 2025*")