import streamlit as st
import tempfile
import os
import sys
import time

# --- ROBUST PATH MANAGEMENT ---
# Essential to import modules from 'src' when running from 'app' folder
current_dir = os.path.dirname(os.path.abspath(__file__)) # app/
project_root = os.path.dirname(current_dir) # phobiashield root
if project_root not in sys.path:
    sys.path.append(project_root)

# Import logic safely
try:
    from src.inference.video_processor import PhobiaVideoProcessor
    PROCESSOR_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Critical Error: Could not import video_processor. {e}")
    PROCESSOR_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PhobiaShield Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (Professional Look) ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    h1 {
        color: #ff4b4b;
    }
    /* Hide Streamlit Menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("# 🛡️")
with col2:
    st.title("PhobiaShield: AI Content Moderation")
    st.markdown("### Automated Phobia Trigger Detection & Blurring System")

st.markdown("---")

# --- SIDEBAR: SETTINGS ---
st.sidebar.title("⚙️ Configuration")

st.sidebar.subheader("Detection Settings")
simulate_mode = st.sidebar.checkbox(
    "Simulation Mode (Monte Carlo)", 
    value=True,
    help="If enabled, generates stochastic detections to test the pipeline without the Neural Network."
)

debug_mode = st.sidebar.checkbox(
    "Debug Mode (Show Boxes)", 
    value=True,
    help="Draws bounding boxes and labels around detected objects for validation."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Supported Classes")
phobias = {
    "🕷️ Spiders": "Arachnophobia",
    "🦈 Sharks": "Galeophobia",
    "🤡 Clowns": "Coulrophobia",
    "🩸 Blood": "Hemophobia",
    "💉 Needles": "Trypanophobia"
}
for p, desc in phobias.items():
    st.sidebar.text(f"{p} - {desc}")

st.sidebar.markdown("---")
st.sidebar.caption(f"Member 3: Engineering Module\nStatus: {'✅ Online' if PROCESSOR_AVAILABLE else '❌ Offline'}")

# --- MAIN INTERFACE ---

if not PROCESSOR_AVAILABLE:
    st.warning("⚠️ Application Logic missing. Check src/inference/video_processor.py")
    st.stop()

# 1. File Uploader
uploaded_file = st.file_uploader("📂 Upload a Video Trailer (MP4)", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # Layout: Split screen
    col_source, col_result = st.columns(2)
    
    with col_source:
        st.subheader("🎬 Original Trailer")
        st.video(uploaded_file)
        
        # Save temp file for processing (OpenCV needs a file path)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        st.info(f"File loaded: {uploaded_file.name} ({len(uploaded_file.getvalue()) / 1024 / 1024:.2f} MB)")

    # 2. Processing Action
    with st.sidebar:
        st.write("## Actions")
        # Unique key helps prevent UI glitches
        start_btn = st.button("🚀 START PROTECTION FILTER", type="primary", key="start_processing")

    if start_btn:
        # Prepare Output Path (Use WebM for browser compatibility)
        output_filename = "processed_demo.webm"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        # Instantiate Processor
        processor = PhobiaVideoProcessor(output_dir=tempfile.gettempdir())
        
        # Progress UI
        progress_text = "Operation in progress. Please wait..."
        my_bar = st.progress(0, text=progress_text)
        
        try:
            with st.spinner('Neural Network (or Simulation) is analyzing frames...'):
                
                # EXECUTE PIPELINE
                # The video processor handles the heavy lifting
                processor.process_video(
                    input_path=video_path,
                    output_name=output_filename,
                    simulate=simulate_mode,
                    debug=debug_mode
                )
                
            my_bar.progress(100, text="Processing Complete!")
            
            # 3. Display Result
            # We display this INSIDE the if-block to ensure it shows up right after processing
            with col_result:
                st.subheader("🛡️ Protected Result")
                
                if os.path.exists(output_path):
                    # Read bytes to serve correctly in Streamlit
                    with open(output_path, 'rb') as f:
                        video_bytes = f.read()
                        st.video(video_bytes, format="video/webm")
                    
                    st.success(f"Output saved successfully!")
                    
                    # Download Button
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Protected Video",
                            data=file,
                            file_name="phobiashield_protected.webm",
                            mime="video/webm"
                        )
                else:
                    st.error("Error: Output file was not generated by the processor.")

        except Exception as e:
            st.error(f"An error occurred during pipeline execution: {e}")
            # Print stack trace to terminal for debugging
            import traceback
            traceback.print_exc()
            
        finally:
            # Cleanup source temp file (optional but recommended)
            # try:
            #     os.unlink(video_path)
            # except:
            #     pass
            pass

else:
    # Empty State
    st.info("👆 Please upload a video file to begin the demonstration.")
    
    # Optional: Demo Instructions
    with st.expander("ℹ️ How to use this demo"):
        st.markdown("""
        1. **Upload** a movie trailer (e.g., IT, Saw, Jaws).
        2. **Configure** simulation settings in the sidebar.
        3. Click **Start Protection Filter**.
        4. The system will detect triggers and apply **Gaussian Blur** (Convolution).
        """)

# --- FOOTER ---
st.markdown("---")
st.markdown("*PhobiaShield Project - Engineering Module (Member 3)*")