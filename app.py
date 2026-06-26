import os
import sys

# ── Aggressive tqdm monitor kill ──────────────────────────────────────
# Python 3.13 + Streamlit: tqdm's background monitor thread causes a
# fatal "PyThreadState_Get: GIL not held" crash. Environment variables
# alone are NOT sufficient because transformers/mlx_vlm import tqdm and
# may start the monitor before our env vars take effect.
os.environ["TQDM_DISABLE"] = "1"
os.environ["TQDM_MONITOR_INTERVAL"] = "0"

def _kill_tqdm_monitor():
    """Stop and remove the tqdm monitor thread if it was already started."""
    try:
        import tqdm
        monitor = getattr(tqdm, "_monitor", None) or getattr(tqdm.tqdm, "monitor", None)
        if monitor is not None:
            monitor.exit()
            try:
                monitor.join(timeout=2)
            except Exception:
                pass
        tqdm.tqdm.monitor_interval = 0
        tqdm.tqdm.monitor = None
        if hasattr(tqdm, "_monitor"):
            tqdm._monitor = None
    except Exception:
        pass


import streamlit as st
import json
import pandas as pd
from PIL import Image

# Adjust path to import ocr_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ocr_engine import TyreOCREngine, MLX_AVAILABLE

# Set page configuration
st.set_page_config(
    page_title="Tyre OCR Specification Extractor",
    page_icon="🛞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS for styling
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    /* Main body background & fonts */
    .stApp {
        background-color: #0c0d16;
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Premium Title styling */
    .title-container {
        text-align: center;
        padding: 30px 10px;
        margin-bottom: 20px;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(90deg, #6366f1 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        letter-spacing: -0.05em;
    }
    
    .sub-title {
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 300;
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111322 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Custom Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(20px);
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Section Headings */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 8px;
    }
    
    /* Specification metrics list */
    .spec-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
    }
    
    .spec-item {
        background: rgba(255, 255, 255, 0.01);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 14px 16px;
        transition: background 0.2s ease;
    }
    
    .spec-item:hover {
        background: rgba(99, 102, 241, 0.03);
    }
    
    .spec-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    
    .spec-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f5f9;
    }
    
    .spec-value-highlight {
        color: #06b6d4;
    }
    
    /* Text bubble */
    .chip {
        display: inline-block;
        background: rgba(6, 182, 212, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.2);
        color: #22d3ee;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.85rem;
        margin-right: 8px;
        margin-bottom: 8px;
        font-weight: 500;
    }
    
    /* Spinner override for style */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }
    
    /* File uploader styling overlay */
    section[data-testid="stFileUploader"] {
        border: 2px dashed rgba(99, 102, 241, 0.3) !important;
        border-radius: 16px !important;
        background: rgba(99, 102, 241, 0.01) !important;
        padding: 10px;
        transition: border-color 0.2s ease;
    }
    
    section[data-testid="stFileUploader"]:hover {
        border-color: #06b6d4 !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown(
    """
    <div class="title-container">
        <div class="main-title">🛞 TYRE OCR EXTRACTOR</div>
        <div class="sub-title">Extract stencil numbers, tyre sizes, tube sizes, and flap sizes from tyre sidewall images using open-source Vision Language Models.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar configurations
st.sidebar.markdown("### ⚙️ Engine Settings")
model_option = st.sidebar.selectbox(
    "Choose Model",
    [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2-VL-2B-Instruct",
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo11l.pt",
        "yolo11x.pt"
    ],
    help="Select Qwen-VL for full text specification OCR, or a YOLOv11 model for object detection + TrOCR text area extraction. Qwen2.5-VL-7B gives the best accuracy (recommended with MLX)."
)

is_yolo = "yolo11" in model_option.lower() or model_option.endswith(".pt")

# Extraction mode selector (only for VLM models)
extraction_mode = "stencil"  # default
if not is_yolo:
    extraction_mode = st.sidebar.selectbox(
        "Extraction Mode",
        ["stencil", "tyre_size", "tube_size", "flap_size"],
        format_func=lambda m: {
            "stencil": "🏭 Stencil Number",
            "tyre_size": "📏 Tyre Size & Pattern",
            "tube_size": "🔧 Tube Size",
            "flap_size": "📐 Flap Size",
        }[m],
        help="Select the type of information to extract from the tyre image."
    )

use_mlx_opt = False
if not is_yolo:
    if MLX_AVAILABLE:
        use_mlx_opt = st.sidebar.checkbox(
            "Use MLX Acceleration",
            value=True,
            help="Runs optimized 4-bit model directly on Apple Silicon GPU for up to 3x speedups."
        )
    else:
        st.sidebar.info("💡 MLX is not installed. Standard PyTorch (utilizing MPS/CUDA/CPU) will be used.")
else:
    st.sidebar.info("⚡ YOLO runs via standard PyTorch. Bounding boxes are cropped and processed by TrOCR automatically to read text.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### ℹ️ How to read Tyre Sidewalls
    1. **Brand**: e.g., *Michelin*, *Bridgestone*, *Continental*.
    2. **Size**: e.g., *205/55R16* where:
       - **205** is width (mm)
       - **55** is aspect ratio (%)
       - **R** is radial construction
       - **16** is rim diameter (inches)
    3. **Load/Speed**: e.g., *91V* where:
       - **91** is load index (615 kg)
       - **V** is speed rating (240 km/h)
    4. **DOT Code**: e.g., *DOT XT 9E 1819*
       - Last 4 digits (*1819*) represent **Week 18, 2019**.
    """
)

# Function to cache model loading
@st.cache_resource(show_spinner=False)
def load_engine(model_id, use_mlx):
    engine = TyreOCREngine(model_id=model_id, use_mlx=use_mlx)
    engine.load_model()
    _kill_tqdm_monitor()  # Kill any tqdm monitor thread spawned during model loading
    return engine

# Initialize engine loader state
engine = None
try:
    with st.spinner("Initializing neural engine (may take a moment on first load)..."):
        engine = load_engine(model_option, use_mlx_opt)
except Exception as e:
    st.error(f"Failed to load OCR model: {e}")
    st.info("Check if Hugging Face is accessible and you have sufficient RAM.")

# Create main interface divisions
col_upload, col_result = st.columns([1, 1.2], gap="large")

with col_upload:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-header">📸 Upload Tyre Sidewall Image</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader(
        "Select file...",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Style layout for the image preview
        st.image(image, use_column_width=True, caption="Source Image")
        
        # Save temp file for model ingestion
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        button_label = "🔍 Run Object Detection" if is_yolo else "🔍 Run Extraction"
        trigger_btn = st.button(button_label, use_container_width=True, type="primary")
        
        if trigger_btn:
            if engine is not None:
                # Setup progress tracking
                progress_placeholder = st.empty()
                if is_yolo:
                    progress_placeholder.info("⚡ Running YOLOv11 object detection...")
                else:
                    progress_placeholder.info("⚡ Extracting text and parsing markings using Qwen-VL...")
                
                try:
                    result = engine.extract_tyre_info(temp_path, mode=extraction_mode)
                    _kill_tqdm_monitor()  # Kill tqdm monitor that may respawn during inference
                    st.session_state["ocr_result"] = result
                    progress_placeholder.success("✅ Analysis complete!")
                except Exception as ex:
                    progress_placeholder.error(f"❌ Error during execution: {ex}")
            else:
                st.warning("Neural engine could not be loaded. Check logs.")

with col_result:
    header_title = "⚙️ Detection Overlay & Details" if (
        "ocr_result" in st.session_state and st.session_state["ocr_result"].get("is_detection")
    ) else "⚙️ Extracted Specifications"

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-header">{header_title}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if "ocr_result" in st.session_state:
        res = st.session_state["ocr_result"]
        
        if res.get("is_detection"):
            annotated_img_path = res.get("annotated_image_path")
            if annotated_img_path and os.path.exists(annotated_img_path):
                st.image(annotated_img_path, use_column_width=True, caption="YOLOv11 Detection Output")
                
            detections = res.get("detections", [])
            if detections:
                # Group by class to show metrics
                df = pd.DataFrame(detections)
                df_display = df.copy()
                df_display["confidence"] = df_display["confidence"].map(lambda x: f"{x * 100:.1f}%")
                df_display["box"] = df_display["box"].map(lambda x: f"[{', '.join([str(round(v, 1)) for v in x])}]")
                
                # Reorder and rename columns for a professional UI table
                cols = ["class", "confidence", "box"]
                if "text" in df_display.columns:
                    cols.append("text")
                df_display = df_display[cols]
                df_display = df_display.rename(columns={
                    "class": "Detected Object",
                    "confidence": "Confidence Score",
                    "box": "Bounding Box (xyxy)",
                    "text": "Extracted Text (TrOCR)"
                })
                
                class_counts = df["class"].value_counts().to_dict()
                
                # Render stats
                stats_html = "<div class='spec-grid'>"
                for cls_name, count in class_counts.items():
                    stats_html += f"""
                    <div class="spec-item">
                        <div class="spec-label">{cls_name} count</div>
                        <div class="spec-value spec-value-highlight">{count}</div>
                    </div>
                    """
                stats_html += "</div>"
                
                st.markdown("#### 📊 Detections Summary")
                st.markdown(stats_html, unsafe_allow_html=True)
                
                st.markdown("#### 📋 Detections List")
                st.dataframe(df_display, use_container_width=True)
            else:
                st.info("No objects detected in the image.")
                
            # Developer Output
            with st.expander("👾 Developer Output (Raw JSON)"):
                st.json(res)
                
            # Download results buttons
            csv_data = pd.DataFrame(detections).to_csv(index=False).encode('utf-8')
            json_data = json.dumps(res, indent=2).encode('utf-8')
            
            st.write("---")
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="📥 Download JSON Results",
                    data=json_data,
                    file_name="yolo_detections.json",
                    mime="application/json",
                    use_container_width=True
                )
            with dl_col2:
                st.download_button(
                    label="📥 Download CSV Summary",
                    data=csv_data,
                    file_name="yolo_detections.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            # Mode-aware result display
            mode = res.get("_mode", extraction_mode)
            debug_msg = res.get("debugMessage", "")
            confidence = res.get("confidence", "unknown")
            validation = res.get("_validation", "")
            quality = res.get("_quality", {})
            
            # Confidence badge colors
            conf_colors = {
                "high": ("#10b981", "rgba(16,185,129,0.1)"),
                "medium": ("#f59e0b", "rgba(245,158,11,0.1)"),
                "low": ("#ef4444", "rgba(239,68,68,0.1)"),
            }
            conf_color, conf_bg = conf_colors.get(confidence, ("#64748b", "rgba(100,116,139,0.1)"))
            
            # Quality + Confidence + Validation status bar
            status_html = f"""
            <div style="display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap;">
                <div style="padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;
                            background: {conf_bg}; color: {conf_color}; border: 1px solid {conf_color}30;">
                    Confidence: {confidence.upper()}
                </div>
            """
            if validation:
                v_color = "#10b981" if validation == "PASS" else "#ef4444" if validation in ("FAIL", "ERROR") else "#f59e0b"
                status_html += f"""
                <div style="padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;
                            background: {v_color}18; color: {v_color}; border: 1px solid {v_color}30;">
                    Validation: {validation}
                </div>
                """
            if quality.get("blur_score") is not None:
                blur = quality["blur_score"]
                blur_color = "#10b981" if blur >= 50 else "#ef4444"
                status_html += f"""
                <div style="padding: 6px 14px; border-radius: 20px; font-size: 0.85rem;
                            background: rgba(100,116,139,0.08); color: #94a3b8; border: 1px solid rgba(100,116,139,0.15);">
                    Sharpness: <span style="color: {blur_color}; font-weight: 600;">{blur}</span>
                </div>
                """
            status_html += "</div>"
            st.markdown(status_html, unsafe_allow_html=True)
            
            # Validation warning
            if validation == "FAIL":
                reason = res.get("_validation_reason", "")
                st.warning(f"⚠️ Validation failed: {reason}")
            elif validation == "SKIPPED":
                reason = res.get("_validation_reason", "")
                st.error(f"🚫 {reason}")

            if mode == "stencil":
                val = res.get("stencilNumber") or "Not Detected"
                specs_html = f"""
                <div class="glass-card">
                    <div class="spec-grid">
                        <div class="spec-item" style="grid-column: 1 / -1;">
                            <div class="spec-label">Stencil Number</div>
                            <div class="spec-value spec-value-highlight" style="font-size: 2rem; letter-spacing: 0.1em;">{val}</div>
                        </div>
                    </div>
                </div>
                """
            elif mode == "tyre_size":
                size_val = res.get("tyreSize") or "Not Detected"
                pattern_val = res.get("patternName") or "Not Detected"
                specs_html = f"""
                <div class="glass-card">
                    <div class="spec-grid">
                        <div class="spec-item">
                            <div class="spec-label">Tyre Size</div>
                            <div class="spec-value spec-value-highlight" style="font-size: 1.5rem;">{size_val}</div>
                        </div>
                        <div class="spec-item">
                            <div class="spec-label">Pattern Name</div>
                            <div class="spec-value" style="font-size: 1.5rem;">{pattern_val}</div>
                        </div>
                    </div>
                </div>
                """
            elif mode == "tube_size":
                val = res.get("tubeSize") or "Not Detected"
                specs_html = f"""
                <div class="glass-card">
                    <div class="spec-grid">
                        <div class="spec-item" style="grid-column: 1 / -1;">
                            <div class="spec-label">Tube Size</div>
                            <div class="spec-value spec-value-highlight" style="font-size: 2rem;">{val}</div>
                        </div>
                    </div>
                </div>
                """
            elif mode == "flap_size":
                val = res.get("flapSize") or "Not Detected"
                specs_html = f"""
                <div class="glass-card">
                    <div class="spec-grid">
                        <div class="spec-item" style="grid-column: 1 / -1;">
                            <div class="spec-label">Flap Size</div>
                            <div class="spec-value spec-value-highlight" style="font-size: 2rem;">{val}</div>
                        </div>
                    </div>
                </div>
                """
            else:
                specs_html = ""

            st.markdown(specs_html, unsafe_allow_html=True)

            # Debug message card
            if debug_msg:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-color: rgba(99, 102, 241, 0.15);">
                        <div class="spec-label">🔎 Model's Reasoning</div>
                        <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px; line-height: 1.6;">{debug_msg}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Raw JSON section
            with st.expander("👾 Developer Output (Raw JSON)"):
                st.json(res)
                
            # Download results
            json_data = json.dumps(res, indent=2).encode('utf-8')
            st.write("---")
            st.download_button(
                label="📥 Download JSON Results",
                data=json_data,
                file_name=f"tyre_{mode}_result.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        info_msg = (
            "Upload an image and click <b>Run Object Detection</b> to view bounding boxes."
            if is_yolo
            else "Upload an image and click <b>Run Extraction</b> to view specifications."
        )
        st.markdown(
            f"""
            <div style="text-align: center; padding: 60px 20px; border: 1px dashed rgba(255, 255, 255, 0.05); border-radius: 12px;">
                <span style="font-size: 3rem; display: block; margin-bottom: 10px;">🔍</span>
                <span style="color: #64748b; font-size: 1rem;">{info_msg}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# ════════════════════════════════════════════════════════════════
# BATCH REVIEW SECTION
# ════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    """
    <div class="glass-card">
        <div class="section-header">📋 Batch Review</div>
        <div style="color: #94a3b8; font-size: 0.9rem;">
            Upload a CSV generated by <code>batch.py</code> to review results, flag errors, and export annotated data.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

batch_csv = st.file_uploader(
    "Upload batch results CSV",
    type=["csv"],
    key="batch_csv_uploader",
    label_visibility="collapsed"
)

if batch_csv is not None:
    df = pd.read_csv(batch_csv)
    
    if "human_verdict" not in df.columns:
        df["human_verdict"] = ""
    if "human_notes" not in df.columns:
        df["human_notes"] = ""
    
    # Store in session state for persistence across reruns
    if "batch_df" not in st.session_state or st.session_state.get("_batch_file") != batch_csv.name:
        st.session_state["batch_df"] = df.copy()
        st.session_state["_batch_file"] = batch_csv.name
    
    batch_df = st.session_state["batch_df"]
    total_rows = len(batch_df)
    
    # Summary metrics
    reviewed = batch_df["human_verdict"].apply(lambda x: x != "" if isinstance(x, str) else False).sum()
    pass_count = (batch_df.get("validation", pd.Series()) == "PASS").sum()
    fail_count = (batch_df.get("validation", pd.Series()).isin(["FAIL", "ERROR"])).sum()
    skip_count = (batch_df.get("validation", pd.Series()) == "SKIPPED").sum()
    
    met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)
    met_col1.metric("Total", total_rows)
    met_col2.metric("✅ Passed", int(pass_count))
    met_col3.metric("❌ Failed", int(fail_count))
    met_col4.metric("⏭️ Skipped", int(skip_count))
    met_col5.metric("👁️ Reviewed", f"{int(reviewed)}/{total_rows}")
    
    # Filter controls
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        filter_validation = st.multiselect(
            "Filter by Validation",
            options=["PASS", "FAIL", "SKIPPED", "ERROR"],
            default=["PASS", "FAIL", "SKIPPED", "ERROR"],
        )
    with filter_col2:
        filter_reviewed = st.selectbox(
            "Filter by Review Status",
            options=["All", "Not Reviewed", "Reviewed"],
        )
    
    # Apply filters
    mask = batch_df["validation"].isin(filter_validation) if "validation" in batch_df.columns else pd.Series([True]*total_rows)
    if filter_reviewed == "Not Reviewed":
        mask = mask & (batch_df["human_verdict"] == "")
    elif filter_reviewed == "Reviewed":
        mask = mask & (batch_df["human_verdict"] != "")
    
    filtered_df = batch_df[mask].reset_index(drop=False)  # keep original index
    
    if len(filtered_df) == 0:
        st.info("No rows match the current filters.")
    else:
        # Paginated review — show 10 rows at a time
        page_size = 10
        total_pages = max(1, (len(filtered_df) + page_size - 1) // page_size)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))
        
        for i in range(start_idx, end_idx):
            row = filtered_df.iloc[i]
            orig_idx = row["index"]  # original dataframe index
            
            # Color-code by validation status
            validation_status = row.get("validation", "")
            if validation_status == "PASS":
                border_color = "rgba(16,185,129,0.3)"
            elif validation_status in ("FAIL", "ERROR"):
                border_color = "rgba(239,68,68,0.3)"
            elif validation_status == "SKIPPED":
                border_color = "rgba(245,158,11,0.3)"
            else:
                border_color = "rgba(255,255,255,0.06)"
            
            verdict = row.get("human_verdict", "")
            verdict_icon = {"correct": "✅", "wrong": "❌", "unsure": "❓"}.get(verdict, "⬜")
            
            st.markdown(
                f"""
                <div style="background: rgba(255,255,255,0.02); border: 1px solid {border_color};
                            border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: 700; color: #f1f5f9;">{verdict_icon} {row.get('filename', 'N/A')}</span>
                            <span style="color: #64748b; font-size: 0.85rem; margin-left: 12px;">blur: {row.get('blur_score', 'N/A')}</span>
                        </div>
                        <div style="display: flex; gap: 8px;">
                            <span style="padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;
                                        background: {'rgba(16,185,129,0.1)' if row.get('confidence')=='high' else 'rgba(245,158,11,0.1)' if row.get('confidence')=='medium' else 'rgba(239,68,68,0.1)'};
                                        color: {'#10b981' if row.get('confidence')=='high' else '#f59e0b' if row.get('confidence')=='medium' else '#ef4444'};">
                                {row.get('confidence', 'N/A').upper()}
                            </span>
                            <span style="padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;
                                        background: {'rgba(16,185,129,0.1)' if validation_status=='PASS' else 'rgba(239,68,68,0.1)' if validation_status in ('FAIL','ERROR') else 'rgba(245,158,11,0.1)'};
                                        color: {'#10b981' if validation_status=='PASS' else '#ef4444' if validation_status in ('FAIL','ERROR') else '#f59e0b'};">
                                {validation_status}
                            </span>
                        </div>
                    </div>
                    <div style="margin-top: 8px; font-size: 1.1rem; color: #06b6d4; font-weight: 600; letter-spacing: 0.05em;">
                        {row.get('extracted_value', '') or '<span style="color: #64748b;">null</span>'}
                        {'<span style="color: #94a3b8; font-size: 0.9rem; margin-left: 10px;">' + str(row.get('pattern_name', '')) + '</span>' if row.get('pattern_name') else ''}
                    </div>
                    {'<div style="color: #f87171; font-size: 0.8rem; margin-top: 4px;">' + str(row.get('validation_reason', '')) + '</div>' if row.get('validation_reason') else ''}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Review buttons
            btn_cols = st.columns([1, 1, 1, 3])
            with btn_cols[0]:
                if st.button("✅ Correct", key=f"correct_{orig_idx}", use_container_width=True):
                    st.session_state["batch_df"].at[orig_idx, "human_verdict"] = "correct"
                    st.rerun()
            with btn_cols[1]:
                if st.button("❌ Wrong", key=f"wrong_{orig_idx}", use_container_width=True):
                    st.session_state["batch_df"].at[orig_idx, "human_verdict"] = "wrong"
                    st.rerun()
            with btn_cols[2]:
                if st.button("❓ Unsure", key=f"unsure_{orig_idx}", use_container_width=True):
                    st.session_state["batch_df"].at[orig_idx, "human_verdict"] = "unsure"
                    st.rerun()
        
        st.markdown(f"<div style='text-align: center; color: #64748b; margin-top: 10px;'>Page {page} of {total_pages} ({len(filtered_df)} rows)</div>", unsafe_allow_html=True)
    
    # Export reviewed results
    st.markdown("---")
    reviewed_df = st.session_state["batch_df"]
    reviewed_csv = reviewed_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Reviewed CSV",
        data=reviewed_csv,
        file_name="batch_reviewed.csv",
        mime="text/csv",
        use_container_width=True,
    )
