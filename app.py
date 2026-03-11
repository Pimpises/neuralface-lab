import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps
from pathlib import Path
import pandas as pd
import numpy as np
import time
import os
import shutil

# [NEW] Import for Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import cv2
    GRADCAM_AVAILABLE = True
    GRADCAM_IMPORT_ERROR = ""
except Exception as e:
    GRADCAM_AVAILABLE = False
    GRADCAM_IMPORT_ERROR = str(e)

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NeuralFace Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

if not GRADCAM_AVAILABLE:
    st.warning(f"Grad-CAM unavailable on this deployment: {GRADCAM_IMPORT_ERROR}")

# 🎨 CLEAN LAB DESIGN SYSTEM (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .verdict-ai {
        color: #ef4444;
        font-weight: 800;
        font-size: 1.5rem;
        background: #fee2e2;
        padding: 8px 16px;
        border-radius: 8px;
        display: inline-block;
    }
    
    .verdict-real {
        color: #10b981;
        font-weight: 800;
        font-size: 1.5rem;
        background: #d1fae5;
        padding: 8px 16px;
        border-radius: 8px;
        display: inline-block;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #10b981, #ef4444);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL ENGINE (LOADER)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model" / "efficientnet_b1_cropped_face_paper_tech_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP_DIR = "temp_lab_previews"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@st.cache_resource
def load_engine():
    try:
        mtcnn = MTCNN(
            image_size=224,
            margin=20,
            keep_all=False,
            select_largest=True,
            device=DEVICE
        )
        
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        
        if MODEL_PATH.exists():
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            return mtcnn, model, True
        else:
            return None, None, False
            
    except Exception as e:
        st.error(f"Critical Error: {str(e)}")
        return None, None, False

mtcnn, model, model_loaded = load_engine()

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================
def process_face(face_tensor):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if isinstance(face_tensor, torch.Tensor):
        face_pil = transforms.ToPILImage()((face_tensor + 1) / 2)
    else:
        face_pil = face_tensor
        
    return transform(face_pil).unsqueeze(0).to(DEVICE), face_pil

def get_gradcam_visualization(model, input_tensor, face_pil):
    if not GRADCAM_AVAILABLE:
        return None

    try:
        model.eval()
        model.zero_grad()
        
        target_layers = [model.features[-1]]

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            face_np = np.array(face_pil).astype(np.float32) / 255.0
            
            if grayscale_cam.shape != face_np.shape[:2]:
                grayscale_cam = cv2.resize(grayscale_cam, (face_np.shape[1], face_np.shape[0]))
                
            visualization = show_cam_on_image(face_np, grayscale_cam, use_rgb=True)
            return visualization
            
    except Exception as e:
        st.error(f"Grad-CAM Error Details: {str(e)}")
        return None

def inference(img_pil, generate_gradcam=False):
    face = mtcnn(img_pil)
    
    if face is None:
        return None, None, None, None
        
    input_tensor, face_visual = process_face(face)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        fake_prob = probs[0][1].item()
    
    gradcam_vis = None
    if generate_gradcam and GRADCAM_AVAILABLE:
        gradcam_vis = get_gradcam_visualization(model, input_tensor, face_visual)

    return fake_prob, face_visual, input_tensor, gradcam_vis

# ==========================================
# 4. SIDEBAR (CONTROLS)
# ==========================================
with st.sidebar:
    st.markdown("Controls Panel")
    
    st.markdown("---")
    st.markdown("**Model Config**")
    
    threshold = st.selectbox(
        "Decision Threshold",
        options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        index=4,
        help="Cutoff point for classifying as AI"
    )
    
    st.markdown("---")
    st.markdown("**System Status**")
    if model_loaded:
        st.success(f"🟢 Model Active ({DEVICE})")
    else:
        st.error("🔴 Model Not Found")
        st.info(f"Path: {MODEL_PATH}")
        
    if not GRADCAM_AVAILABLE:
        st.warning("Grad-CAM is disabled on this deployment.")
        
    st.markdown("---")
    st.markdown("<div style='font-size:12px; color:#94a3b8;'>NeuralFace Lab v2.1<br>Built for Research Validation</div>", unsafe_allow_html=True)

# ==========================================
# 5. MAIN INTERFACE (TABS)
# ==========================================
st.title("AI FACE Detection_DEMO")
st.markdown("Test environment for **DeepFake Detection Model** (EfficientNet-B1) with **Explainable AI (XAI)**")

if not model_loaded:
    st.warning("⚠️ Please place the model file in the correct directory to proceed.")
    st.stop()

tab_single, tab_batch = st.tabs(["Quick Test (Single)", "Batch Audit (Bulk)"])

# --- TAB 1: SINGLE TEST ---
with tab_single:
    col_input, col_result = st.columns([1, 1.5], gap="large")
    
    with col_input:
        st.subheader("1. Input")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="single_up")
        
        if uploaded_file:
            img = Image.open(uploaded_file)
            img = ImageOps.exif_transpose(img).convert("RGB")
            st.image(img, caption="Original Image", use_container_width=True)
            
    with col_result:
        st.subheader("2. Analysis")
        
        if uploaded_file:
            enable_gradcam = st.checkbox(
                "Generate Grad-CAM (Explainable AI)",
                value=GRADCAM_AVAILABLE,
                disabled=not GRADCAM_AVAILABLE,
                help="See where the model is focusing." if GRADCAM_AVAILABLE else "Grad-CAM dependency is unavailable on this deployment."
            )
            
            with st.spinner("Running Inference..."):
                start_time = time.time()
                fake_score, face_crop, _, gradcam_vis = inference(img, generate_gradcam=enable_gradcam)
                end_time = time.time()
                
            if fake_score is not None:
                is_ai = fake_score > threshold
                verdict_class = "verdict-ai" if is_ai else "verdict-real"
                verdict_text = "AI GENERATED" if is_ai else "REAL PERSON"
                icon = "🤖" if is_ai else "👤"
                
                st.markdown(f"""
                <div class="result-card">
                    <div class="{verdict_class}">{icon} {verdict_text}</div>
                    <div style="margin-top: 20px;">
                        <div class="metric-value">{fake_score*100:.2f}%</div>
                        <div class="metric-label">AI Probability Score</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Confidence Spectrum**")
                st.progress(fake_score)
                st.caption("0% (Real) ⟷ 100% (AI)")
                
                st.info(f"Inference Time: {(end_time - start_time)*1000:.1f} ms")
                
                st.markdown("---")
                st.subheader("3. Model Focus (XAI)")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Model Input (Cropped)**")
                    st.image(face_crop, use_container_width=True, caption="What the model sees")
                with c2:
                    st.markdown("**Grad-CAM Heatmap**")
                    if gradcam_vis is not None:
                        st.image(gradcam_vis, use_container_width=True, caption="Where the model focuses (Red = High Impact)")
                    elif enable_gradcam:
                        st.warning("Could not generate Grad-CAM.")
                    else:
                        st.caption("Grad-CAM disabled.")
                        
            else:
                st.error("No Face Detected. Please try an image with a clear frontal face.")
        else:
            st.info("Upload an image to start analysis.")

# --- TAB 2: BATCH TEST ---
with tab_batch:
    st.markdown("### Bulk Processing")
    st.caption("Note: Grad-CAM is disabled in batch mode for performance.")
    batch_files = st.file_uploader("Select multiple images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key="batch_up")
    
    if batch_files:
        if st.button(f"Process {len(batch_files)} Images", type="primary"):
            progress_bar = st.progress(0)
            results_data = []
            
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR)
            
            for idx, file in enumerate(batch_files):
                try:
                    img = Image.open(file)
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    
                    thumb = img.copy()
                    thumb.thumbnail((100, 100))
                    thumb_path = os.path.join(TEMP_DIR, f"thumb_{idx}.png")
                    thumb.save(thumb_path)
                    
                    fake_score, _, _, _ = inference(img, generate_gradcam=False)
                    
                    if fake_score is not None:
                        res_type = "AI" if fake_score > threshold else "REAL"
                        results_data.append({
                            "Preview": thumb_path,
                            "Filename": file.name,
                            "AI Score": fake_score,
                            "Verdict": res_type
                        })
                    else:
                        results_data.append({
                            "Preview": thumb_path,
                            "Filename": file.name,
                            "AI Score": 0.0,
                            "Verdict": "No Face"
                        })
                        
                except Exception as e:
                    results_data.append({
                        "Preview": None,
                        "Filename": file.name,
                        "AI Score": 0.0,
                        "Verdict": f"Error: {str(e)}"
                    })
                
                progress_bar.progress((idx + 1) / len(batch_files))
            
            if results_data:
                df = pd.DataFrame(results_data)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Images", len(df))
                c2.metric("AI Detected", len(df[df['Verdict'] == 'AI']))
                c3.metric("Real Detected", len(df[df['Verdict'] == 'REAL']))
                
                st.divider()
                
                st.dataframe(
                    df,
                    column_config={
                        "Preview": st.column_config.ImageColumn("Image", width="small"),
                        "AI Score": st.column_config.ProgressColumn(
                            "Probability",
                            format="%.4f",
                            min_value=0,
                            max_value=1
                        )
                    },
                    use_container_width=True,
                    height=600
                )
