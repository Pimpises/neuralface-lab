import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import models, transforms

# Safe OpenCV import (fix Streamlit Cloud crash)
try:
    import cv2
except Exception:
    cv2 = None

# Safe GradCAM import
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except Exception:
    GradCAM = None

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="NeuralFace Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. APP CONFIG
# ==========================================
MODEL_PATH = "Model/efficientnet_b1_cropped_face_paper_tech_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REAL_CLASS_INDEX = 0
AI_CLASS_INDEX = 1

CLASS_NAMES = {
    REAL_CLASS_INDEX: "REAL",
    AI_CLASS_INDEX: "AI"
}

if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp(prefix="neuralface_lab_")
TEMP_DIR = st.session_state.temp_dir

if "camera_log" not in st.session_state:
    st.session_state.camera_log = []

# ==========================================
# 3. DESIGN SYSTEM (2026 UI)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0b1020;
        --panel: rgba(255,255,255,0.06);
        --panel-2: rgba(255,255,255,0.08);
        --border: rgba(255,255,255,0.14);
        --border-soft: rgba(255,255,255,0.10);
        --text: #edf2ff;
        --muted: #b7c0d8;
        --muted-2: #95a3c7;
        --accent: #7c9cff;
        --accent-2: #63e6be;
        --danger: #ff6b81;
        --warning: #ffd166;
        --success: #20c997;
        --shadow: 0 10px 35px rgba(0,0,0,0.28);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(124,156,255,0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(99,230,190,0.12), transparent 25%),
            linear-gradient(180deg, #0b1020 0%, #11182b 45%, #0d1324 100%);
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background: rgba(8, 12, 24, 0.95);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #eaf0ff !important;
    }

    .hero-card {
        background:
            linear-gradient(135deg, rgba(124,156,255,0.18), rgba(99,230,190,0.10)),
            rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 28px;
        padding: 28px 30px;
        box-shadow: var(--shadow);
        margin-bottom: 18px;
    }

    .hero-kicker {
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #dbe4ff;
        background: rgba(124,156,255,0.14);
        border: 1px solid rgba(124,156,255,0.28);
        border-radius: 999px;
        padding: 8px 12px;
        margin-bottom: 14px;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.08;
        color: #f8fbff;
        margin-bottom: 10px;
    }

    .hero-desc {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.65;
        max-width: 920px;
    }

    .status-chip-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 18px;
    }

    .status-chip {
        padding: 9px 14px;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 600;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        color: #eef3ff;
    }

    .status-chip.good {
        background: rgba(32,201,151,0.12);
        border-color: rgba(32,201,151,0.28);
        color: #c9ffe9;
    }

    .status-chip.warn {
        background: rgba(255,209,102,0.12);
        border-color: rgba(255,209,102,0.24);
        color: #fff1bf;
    }

    .metric-panel {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 22px;
        padding: 20px 18px;
        box-shadow: var(--shadow);
        text-align: left;
    }

    .metric-panel .label {
        color: #c7d1ea;
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 10px;
        font-weight: 700;
    }

    .metric-panel .value {
        color: #ffffff;
        font-weight: 800;
        font-size: 1.95rem;
        line-height: 1.1;
    }

    .metric-panel .sub {
        color: #b6c1db;
        font-size: 0.94rem;
        margin-top: 6px;
    }

    .section-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 24px;
        padding: 22px;
        box-shadow: var(--shadow);
        margin-top: 10px;
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #f2f6ff;
        margin-bottom: 14px;
    }

    .section-note {
        color: #c0cae4;
        font-size: 0.97rem;
        line-height: 1.62;
    }

    .result-card {
        background:
            linear-gradient(135deg, rgba(124,156,255,0.16), rgba(99,230,190,0.08)),
            rgba(255,255,255,0.06);
        border-radius: 26px;
        padding: 28px;
        box-shadow: var(--shadow);
        border: 1px solid rgba(255,255,255,0.12);
        text-align: center;
        margin-bottom: 20px;
    }

    .verdict-ai, .verdict-real {
        font-weight: 800;
        font-size: 1.15rem;
        padding: 10px 16px;
        border-radius: 999px;
        display: inline-block;
        letter-spacing: 0.02em;
    }

    .verdict-ai {
        color: #ffd7dc;
        background: rgba(255,107,129,0.15);
        border: 1px solid rgba(255,107,129,0.25);
    }

    .verdict-real {
        color: #d0ffef;
        background: rgba(32,201,151,0.15);
        border: 1px solid rgba(32,201,151,0.24);
    }

    .score-big {
        margin-top: 18px;
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
    }

    .score-caption {
        color: #c6d0e8;
        margin-top: 8px;
        font-size: 0.92rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        font-weight: 700;
    }

    .small-kicker {
        display: inline-block;
        color: #e3ebff;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 14px;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        padding: 16px;
        border-radius: 20px;
    }

    [data-testid="stMetricLabel"] {
        color: #d2dbf0 !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }

    div[data-testid="stFileUploader"], div[data-testid="stCameraInput"] {
        background: rgba(255,255,255,0.04);
        border: 1px dashed rgba(255,255,255,0.22);
        border-radius: 20px;
        padding: 10px;
    }

    div[data-testid="stAlert"] {
        border-radius: 18px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255,255,255,0.03);
        padding: 8px;
        border-radius: 16px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 12px;
        padding: 0 16px;
        background: transparent;
        color: #d7e0f7;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(124,156,255,0.14) !important;
        color: #ffffff !important;
    }

    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg, #20c997, #7c9cff, #ff6b81);
        border-radius: 999px;
    }

    .footer-mini {
        color: #b1bfdc;
        font-size: 0.84rem;
        text-align: center;
        margin-top: 14px;
    }

    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(124,156,255,0.20), rgba(99,230,190,0.12));
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        padding: 0.55rem 1rem !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.16);
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: rgba(255,255,255,0.26) !important;
        background: linear-gradient(135deg, rgba(124,156,255,0.28), rgba(99,230,190,0.18));
    }

    .streamlit-expanderHeader {
        color: #eef3ff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
    }

    .streamlit-expanderContent {
        color: #dbe5ff !important;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #c7d2ea !important;
        opacity: 1 !important;
    }

    label[data-testid="stWidgetLabel"] p {
        color: #eef3ff !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. MODEL LOADER
# ==========================================
def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        cleaned_state_dict[new_key] = v

    return cleaned_state_dict


@st.cache_resource
def load_engine():
    try:
        mtcnn_model = MTCNN(
            image_size=256,
            margin=25,
            keep_all=False,
            select_largest=True,
            device=DEVICE
        )

        classifier_model = models.efficientnet_b1(weights=None)
        classifier_model.classifier[1] = nn.Linear(
            classifier_model.classifier[1].in_features, 2
        )

        if not os.path.exists(MODEL_PATH):
            return None, None, False, f"Model file not found at: {MODEL_PATH}"

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = extract_state_dict(checkpoint)
        classifier_model.load_state_dict(state_dict, strict=True)

        classifier_model.to(DEVICE)
        classifier_model.eval()

        return mtcnn_model, classifier_model, True, None

    except Exception as e:
        return None, None, False, f"Critical Error: {str(e)}"


mtcnn, model, model_loaded, model_error = load_engine()

# ==========================================
# 5. UTILITY FUNCTIONS
# ==========================================
def process_face(face_tensor):
    classifier_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if isinstance(face_tensor, torch.Tensor):
        face_tensor = ((face_tensor + 1.0) / 2.0).clamp(0, 1)
        face_pil = transforms.ToPILImage()(face_tensor.cpu())
    else:
        face_pil = face_tensor

    input_tensor = classifier_transform(face_pil).unsqueeze(0).to(DEVICE)
    return input_tensor, face_pil


def detect_faces_with_info(img_pil):
    if mtcnn is None:
        return None, 0, "Model engine is not available."

    try:
        boxes, _ = mtcnn.detect(img_pil)

        if boxes is None or len(boxes) == 0:
            return None, 0, "No face detected."

        face_count = len(boxes)
        main_face = mtcnn(img_pil)

        if main_face is None:
            return None, face_count, "Face detection failed during crop extraction."

        return main_face, face_count, None

    except Exception as e:
        return None, 0, f"Face detection error: {str(e)}"


def draw_max_activation_marker(gradcam_vis, grayscale_cam):
    try:
        if gradcam_vis is None or grayscale_cam is None:
            return gradcam_vis

        max_pos = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)
        y, x = max_pos

        pil_img = Image.fromarray(gradcam_vis)
        draw = ImageDraw.Draw(pil_img)

        r = 10
        draw.ellipse((x - r, y - r, x + r, y + r), outline="white", width=2)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="white")

        text = "Peak AI activation"
        text_x = max(10, x - 45)
        text_y = max(10, y - 20)
        draw.text((text_x, text_y), text, fill="white")

        return np.array(pil_img)
    except Exception:
        return gradcam_vis


def get_gradcam_visualizations(classifier_model, input_tensor, face_pil):

    if GradCAM is None or cv2 is None:
        return None, None, None

    try:
        classifier_model.eval()
        classifier_model.zero_grad()

        target_layers = [classifier_model.features[-1]]
        targets = [ClassifierOutputTarget(AI_CLASS_INDEX)]

        with GradCAM(model=classifier_model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]

            face_np = np.array(face_pil).astype(np.float32) / 255.0

            if grayscale_cam.shape != face_np.shape[:2]:
                cam_img = Image.fromarray((grayscale_cam * 255).astype(np.uint8))
                cam_img = cam_img.resize((face_np.shape[1], face_np.shape[0]), Image.BILINEAR)
                grayscale_cam = np.array(cam_img).astype(np.float32) / 255.0

            plain_vis = show_cam_on_image(face_np, grayscale_cam, use_rgb=True)
            annotated_vis = draw_max_activation_marker(plain_vis, grayscale_cam)

            return plain_vis, annotated_vis, grayscale_cam

    except Exception as e:
        st.error(f"Grad-CAM Error Details: {str(e)}")
        return None, None, None


def inference(img_pil, generate_gradcam=False):
    if mtcnn is None or model is None:
        return None, None, None, None, None, 0, "Model engine is not available."

    face, face_count, detect_err = detect_faces_with_info(img_pil)

    if detect_err is not None:
        return None, None, None, None, None, face_count, detect_err

    input_tensor, face_visual = process_face(face)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        ai_prob = probs[0][AI_CLASS_INDEX].item()

    gradcam_plain = None
    gradcam_annotated = None
    if generate_gradcam:
        gradcam_plain, gradcam_annotated, _ = get_gradcam_visualizations(
            model,
            input_tensor,
            face_visual
        )

    return ai_prob, face_visual, input_tensor, gradcam_plain, gradcam_annotated, face_count, None


def reset_temp_dir(temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)


def pil_to_bytes(img_pil, format="PNG"):
    buf = io.BytesIO()
    img_pil.save(buf, format=format)
    return buf.getvalue()


def ndarray_to_png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def add_camera_log_entry(mode_name, ai_score, verdict, face_count, note, processing_ms):
    st.session_state.camera_log.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Mode": mode_name,
        "AI Score": round(float(ai_score), 6) if ai_score is not None else None,
        "Verdict": verdict,
        "Faces": int(face_count),
        "Processing Time (ms)": round(float(processing_ms), 2),
        "Note": note
    })

# ==========================================
# 6. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### Control Center")
    st.caption("Configure the demo environment and system behavior.")

    st.markdown("---")
    st.markdown("**Inference Settings**")

    threshold = st.selectbox(
        "Decision Threshold",
        options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        index=4,
        help="Cutoff point for classifying an input as AI-generated"
    )

    st.markdown("---")
    st.markdown("**System Status**")

    if model_loaded:
        st.success(f"Model Ready on {DEVICE}")
    else:
        st.error("Model unavailable")
        st.info(f"Expected path: {MODEL_PATH}")
        if model_error:
            st.caption(model_error)

    st.markdown("---")
    st.markdown("**Input Recommendation**")
    st.caption(
        "Use a clear frontal face, one main subject, good lighting, and minimal blur for the most reliable result."
    )

    st.markdown("---")
    st.markdown(
        "<div class='footer-mini'>NeuralFace Lab · 2026 Demo Interface<br>DeepFake Detection + XAI</div>",
        unsafe_allow_html=True
    )

# ==========================================
# 7. HERO SECTION
# ==========================================
device_text = str(DEVICE).upper()
model_status_chip = "good" if model_loaded else "warn"
model_status_text = "Model Ready" if model_loaded else "Model Offline"

st.markdown(f"""
<div class="hero-card">
    <div class="hero-kicker">NeuralFace Lab · 2026 Interface</div>
    <div class="hero-title">AI Face Authenticity Analysis</div>
    <div class="hero-desc">
        End-to-end demo for <b>Real vs AI-generated face detection</b> using
        <b>EfficientNet-B1</b>, automatic face extraction, and <b>Grad-CAM</b> for explainability.
        This interface is designed to demonstrate that the trained model can be used in a real application workflow.
    </div>
    <div class="status-chip-row">
        <div class="status-chip {model_status_chip}">{model_status_text}</div>
        <div class="status-chip">Backend: EfficientNet-B1</div>
        <div class="status-chip">Detector: MTCNN</div>
        <div class="status-chip">Device: {device_text}</div>
        <div class="status-chip">XAI: Grad-CAM</div>
    </div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="metric-panel">
        <div class="label">Primary Task</div>
        <div class="value">Real vs AI</div>
        <div class="sub">Binary face authenticity classification</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-panel">
        <div class="label">Threshold</div>
        <div class="value">{threshold:.2f}</div>
        <div class="sub">Current decision boundary for AI detection</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-panel">
        <div class="label">Explainability</div>
        <div class="value">Active</div>
        <div class="sub">Visual evidence through Grad-CAM heatmaps</div>
    </div>
    """, unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️ Please place the model file in the correct directory to proceed.")
    st.stop()

# ==========================================
# 8. MAIN TABS
# ==========================================
tab_single, tab_batch, tab_camera = st.tabs(
    ["Quick Test (Single)", "Batch Audit (Bulk)", "Live Camera (Webcam)"]
)

# ==========================================
# 9. SINGLE TAB
# ==========================================
with tab_single:
    left_col, right_col = st.columns([1.02, 1.35], gap="large")

    with left_col:
        st.markdown("""
        <div class="section-card">
            <div class="small-kicker">Input Zone</div>
            <div class="section-title">Upload an image for analysis</div>
            <div class="section-note">
                Best results are achieved with a clear frontal face, one main subject,
                good lighting, and minimal blur.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Recommended image conditions"):
            st.markdown("""
            - Clear frontal or near-frontal face  
            - Good lighting  
            - Minimal blur  
            - One main visible subject  
            - Limited occlusion around the face  
            - Best suited for one-person photos  
            """)

        with st.expander("Example input cases"):
            e1, e2, e3 = st.columns(3)
            example_paths = [
                "assets/examples/example_real_1.jpg",
                "assets/examples/example_ai_1.jpg",
                "assets/examples/example_real_2.jpg"
            ]
            example_captions = [
                "Clear frontal real face",
                "Clear frontal AI-generated face",
                "Good lighting / single subject"
            ]
            for col, path, cap in zip([e1, e2, e3], example_paths, example_captions):
                with col:
                    if os.path.exists(path):
                        st.image(path, caption=cap, use_container_width=True)
                    else:
                        st.caption(cap)
                        st.write("Example not found")

        uploaded_file = st.file_uploader(
            "Drop image here",
            type=["jpg", "jpeg", "png"],
            key="single_up"
        )

        img = None
        if uploaded_file:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                st.markdown("""
                <div class="section-card">
                    <div class="small-kicker">Original Input</div>
                    <div class="section-title">Source image</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(img, use_container_width=True)
            except Exception as e:
                st.error(f"Could not open image: {str(e)}")

    with right_col:
        st.markdown("""
        <div class="section-card">
            <div class="small-kicker">Inference Panel</div>
            <div class="section-title">Model decision and explainability</div>
            <div class="section-note">
                The system detects the main face, preprocesses it, runs classification,
                and optionally visualizes AI-related evidence using Grad-CAM.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if uploaded_file and img is not None:
            enable_gradcam = st.toggle(
                "Enable Grad-CAM visualization",
                value=True,
                help="Show the regions that contribute most to AI-class evidence."
            )

            with st.spinner("Analyzing image..."):
                start_time = time.time()
                ai_score, face_crop, _, gradcam_plain, gradcam_annotated, face_count, err = inference(
                    img,
                    generate_gradcam=enable_gradcam
                )
                end_time = time.time()

            if err is not None:
                if "No face detected" in err:
                    st.toast("No face detected in the image.")
                    st.error("No face detected. Please upload a clear frontal face.")
                    st.info("Tip: Use one visible face with better lighting and less blur.")
                else:
                    st.toast("Image analysis failed.")
                    st.error(err)
            else:
                if face_count > 1:
                    st.toast("Multiple faces detected. Main face selected.")
                    st.warning(
                        f"Detected {face_count} faces. The system analyzed the largest / main face only."
                    )
                else:
                    st.success("Face detected successfully.")

                is_ai = ai_score > threshold
                verdict_class = "verdict-ai" if is_ai else "verdict-real"
                verdict_text = "AI GENERATED" if is_ai else "REAL PERSON"
                icon = "🤖" if is_ai else "👤"

                st.markdown(f"""
                <div class="result-card">
                    <div class="{verdict_class}">{icon} {verdict_text}</div>
                    <div class="score-big">{ai_score*100:.2f}%</div>
                    <div class="score-caption">AI Probability Score</div>
                </div>
                """, unsafe_allow_html=True)

                a, b, c = st.columns(3)
                with a:
                    st.metric("Decision Threshold", f"{threshold:.2f}")
                with b:
                    st.metric("Faces Detected", f"{face_count}")
                with c:
                    st.metric("Processing Time", f"{(end_time - start_time)*1000:.1f} ms")

                st.markdown("**Confidence Spectrum**")
                st.progress(float(ai_score))
                st.caption("0% = Real-side confidence · 100% = AI-side confidence")

                st.markdown("---")
                x1, x2, x3 = st.columns(3)

                with x1:
                    st.markdown("""
                    <div class="section-card">
                        <div class="small-kicker">Model Input</div>
                        <div class="section-title">Cropped face used by the model</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(
                        face_crop,
                        use_container_width=True,
                        caption="The classifier only sees this cropped face region."
                    )

                with x2:
                    st.markdown("""
                    <div class="section-card">
                        <div class="small-kicker">Explainability</div>
                        <div class="section-title">Grad-CAM heatmap</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if gradcam_plain is not None:
                        st.image(
                            gradcam_plain,
                            use_container_width=True,
                            caption="Plain Grad-CAM overlay: highlighted regions = stronger contribution to AI evidence."
                        )
                    elif enable_gradcam:
                        st.warning("Could not generate Grad-CAM.")
                    else:
                        st.caption("Grad-CAM disabled.")

                with x3:
                    st.markdown("""
                    <div class="section-card">
                        <div class="small-kicker">Explainability+</div>
                        <div class="section-title">Grad-CAM with peak marker</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if gradcam_annotated is not None:
                        st.image(
                            gradcam_annotated,
                            use_container_width=True,
                            caption="White marker = peak AI activation, highlighted regions = stronger contribution to AI evidence."
                        )
                    elif enable_gradcam:
                        st.warning("Could not generate annotated Grad-CAM.")
                    else:
                        st.caption("Grad-CAM disabled.")
        else:
            st.info("Upload an image to start the analysis workflow.")

# ==========================================
# 10. BATCH TAB
# ==========================================
with tab_batch:
    st.markdown("""
    <div class="section-card">
        <div class="small-kicker">Bulk Audit</div>
        <div class="section-title">Analyze multiple images in one run</div>
        <div class="section-note">
            Batch mode is useful for dataset screening and fast auditing.
            Grad-CAM is disabled here for performance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Select multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_up"
    )

    st.caption("Recommended: use images with one clear main face for more stable results.")

    if batch_files:
        if st.button(f"Process {len(batch_files)} Images", type="primary"):
            progress_bar = st.progress(0)
            results_data = []

            reset_temp_dir(TEMP_DIR)

            for idx, file in enumerate(batch_files):
                thumb_path = None

                try:
                    img = Image.open(file).convert("RGB")

                    thumb = img.copy()
                    thumb.thumbnail((100, 100))
                    thumb_path = os.path.join(TEMP_DIR, f"thumb_{idx}.png")
                    thumb.save(thumb_path)

                    ai_score, _, _, _, _, face_count, err = inference(
                        img,
                        generate_gradcam=False
                    )

                    if err is None and ai_score is not None:
                        verdict_note = f"Multiple faces ({face_count}) → used main face" if face_count > 1 else "OK"
                        res_type = "AI" if ai_score > threshold else "REAL"
                        results_data.append({
                            "Preview": thumb_path,
                            "Filename": file.name,
                            "AI Score": ai_score,
                            "Verdict": res_type,
                            "Faces": face_count,
                            "Note": verdict_note,
                            "Error": ""
                        })
                    else:
                        results_data.append({
                            "Preview": thumb_path,
                            "Filename": file.name,
                            "AI Score": 0.0,
                            "Verdict": "No Face",
                            "Faces": face_count,
                            "Note": "",
                            "Error": err or "No face detected."
                        })

                except Exception as e:
                    results_data.append({
                        "Preview": thumb_path,
                        "Filename": getattr(file, "name", f"file_{idx}"),
                        "AI Score": 0.0,
                        "Verdict": "Error",
                        "Faces": 0,
                        "Note": "",
                        "Error": str(e)
                    })

                progress_bar.progress((idx + 1) / len(batch_files))

            if results_data:
                df = pd.DataFrame(results_data)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Images", len(df))
                c2.metric("AI Detected", int((df["Verdict"] == "AI").sum()))
                c3.metric("Real Detected", int((df["Verdict"] == "REAL").sum()))
                c4.metric("No Face", int((df["Verdict"] == "No Face").sum()))
                c5.metric("Errors", int((df["Verdict"] == "Error").sum()))

                st.divider()

                st.dataframe(
                    df,
                    column_config={
                        "Preview": st.column_config.ImageColumn("Image", width="small"),
                        "AI Score": st.column_config.ProgressColumn(
                            "Probability",
                            format="%.4f",
                            min_value=0.0,
                            max_value=1.0
                        ),
                        "Filename": st.column_config.TextColumn("Filename", width="medium"),
                        "Verdict": st.column_config.TextColumn("Verdict", width="small"),
                        "Faces": st.column_config.NumberColumn("Faces"),
                        "Note": st.column_config.TextColumn("Note", width="medium"),
                        "Error": st.column_config.TextColumn("Error", width="large")
                    },
                    use_container_width=True,
                    height=600
                )

# ==========================================
# 11. CAMERA TAB
# ==========================================
with tab_camera:
    cam_left, cam_right = st.columns([1.02, 1.35], gap="large")

    with cam_left:
        st.markdown("""
        <div class="section-card">
            <div class="small-kicker">Live Camera Mode</div>
            <div class="section-title">Capture from webcam and analyze</div>
            <div class="section-note">
                This mode simulates real-world usage by allowing the user to capture
                a face image directly from a webcam and run the same model pipeline.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Recommended image conditions"):
            st.markdown("""
            - Sit facing the camera (frontal or near-frontal face)  
            - Use good lighting  
            - Keep one main face in the frame  
            - Avoid strong blur or motion  
            - Keep the face reasonably large in the frame  
            """)

        enable_camera_gradcam = st.toggle(
            "Enable Grad-CAM for captured frame",
            value=True,
            key="camera_gradcam_toggle",
            help="Generate Grad-CAM after capturing an image from the webcam."
        )

        camera_image = st.camera_input("Take photo from webcam", key="webcam_capture")

        if camera_image is not None:
            try:
                cam_img = Image.open(camera_image).convert("RGB")
                st.markdown("""
                <div class="section-card">
                    <div class="small-kicker">Captured Frame</div>
                    <div class="section-title">Webcam snapshot</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(cam_img, use_container_width=True)
            except Exception as e:
                cam_img = None
                st.error(f"Could not read captured frame: {str(e)}")
        else:
            cam_img = None

    with cam_right:
        st.markdown("""
        <div class="section-card">
            <div class="small-kicker">Camera Inference</div>
            <div class="section-title">Analyze the captured webcam frame</div>
            <div class="section-note">
                After a frame is captured, the system detects the face, selects the main subject,
                performs classification, and optionally produces Grad-CAM evidence.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if cam_img is not None:
            analyze_now = st.button("Analyze Captured Frame", type="primary", key="analyze_camera_frame")

            if analyze_now:
                with st.spinner("Analyzing webcam capture..."):
                    start_time = time.time()
                    ai_score, face_crop, _, gradcam_plain, gradcam_annotated, face_count, err = inference(
                        cam_img,
                        generate_gradcam=enable_camera_gradcam
                    )
                    end_time = time.time()

                processing_ms = (end_time - start_time) * 1000.0

                if err is not None:
                    if "No face detected" in err:
                        st.toast("No face detected in the webcam frame.")
                        st.error("No face detected. Please face the camera more clearly.")
                        st.info("Tip: Move closer, improve lighting, and keep one visible frontal face.")
                        add_camera_log_entry(
                            mode_name="Webcam",
                            ai_score=0.0,
                            verdict="No Face",
                            face_count=face_count,
                            note=err,
                            processing_ms=processing_ms
                        )
                    else:
                        st.toast("Camera analysis failed.")
                        st.error(err)
                        add_camera_log_entry(
                            mode_name="Webcam",
                            ai_score=0.0,
                            verdict="Error",
                            face_count=face_count,
                            note=err,
                            processing_ms=processing_ms
                        )
                else:
                    if face_count > 1:
                        st.toast("Multiple faces detected. Main face selected.")
                        st.warning(
                            f"Detected {face_count} faces. The system analyzed the largest / main face only."
                        )
                        note_text = f"Multiple faces detected ({face_count}) → used main face"
                    else:
                        st.success("Face detected successfully from webcam.")
                        note_text = "Single face detected"

                    is_ai = ai_score > threshold
                    verdict_class = "verdict-ai" if is_ai else "verdict-real"
                    verdict_text = "AI GENERATED" if is_ai else "REAL PERSON"
                    icon = "🤖" if is_ai else "👤"
                    verdict_log = "AI" if is_ai else "REAL"

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="{verdict_class}">{icon} {verdict_text}</div>
                        <div class="score-big">{ai_score*100:.2f}%</div>
                        <div class="score-caption">AI Probability Score</div>
                    </div>
                    """, unsafe_allow_html=True)

                    ca, cb, cc = st.columns(3)
                    with ca:
                        st.metric("Decision Threshold", f"{threshold:.2f}")
                    with cb:
                        st.metric("Faces Detected", f"{face_count}")
                    with cc:
                        st.metric("Processing Time", f"{processing_ms:.1f} ms")

                    st.markdown("**Confidence Spectrum**")
                    st.progress(float(ai_score))
                    st.caption("0% = Real-side confidence · 100% = AI-side confidence")

                    st.markdown("---")
                    y1, y2, y3 = st.columns(3)

                    with y1:
                        st.markdown("""
                        <div class="section-card">
                            <div class="small-kicker">Model Input</div>
                            <div class="section-title">Cropped face used by the model</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(
                            face_crop,
                            use_container_width=True,
                            caption="The classifier only sees this cropped face region."
                        )
                        st.download_button(
                            "Download Cropped Face",
                            data=pil_to_bytes(face_crop, format="PNG"),
                            file_name=f"webcam_cropped_face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            key="download_cropped_face_camera"
                        )

                    with y2:
                        st.markdown("""
                        <div class="section-card">
                            <div class="small-kicker">Explainability</div>
                            <div class="section-title">Grad-CAM heatmap</div>
                        </div>
                        """, unsafe_allow_html=True)

                        if gradcam_plain is not None:
                            st.image(
                                gradcam_plain,
                                use_container_width=True,
                                caption="Plain Grad-CAM overlay: highlighted regions = stronger contribution to AI evidence."
                            )
                            st.download_button(
                                "Download Plain Grad-CAM",
                                data=ndarray_to_png_bytes(gradcam_plain),
                                file_name=f"webcam_gradcam_plain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                key="download_gradcam_plain_camera"
                            )
                        elif enable_camera_gradcam:
                            st.warning("Could not generate plain Grad-CAM.")
                        else:
                            st.caption("Grad-CAM disabled.")

                    with y3:
                        st.markdown("""
                        <div class="section-card">
                            <div class="small-kicker">Explainability+</div>
                            <div class="section-title">Grad-CAM with peak marker</div>
                        </div>
                        """, unsafe_allow_html=True)

                        if gradcam_annotated is not None:
                            st.image(
                                gradcam_annotated,
                                use_container_width=True,
                                caption="White marker = peak AI activation, highlighted regions = stronger contribution to AI evidence."
                            )
                            st.download_button(
                                "Download Annotated Grad-CAM",
                                data=ndarray_to_png_bytes(gradcam_annotated),
                                file_name=f"webcam_gradcam_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                key="download_gradcam_annotated_camera"
                            )
                        elif enable_camera_gradcam:
                            st.warning("Could not generate annotated Grad-CAM.")
                        else:
                            st.caption("Grad-CAM disabled.")

                    add_camera_log_entry(
                        mode_name="Webcam",
                        ai_score=ai_score,
                        verdict=verdict_log,
                        face_count=face_count,
                        note=note_text,
                        processing_ms=processing_ms
                    )
        else:
            st.info("Take a webcam photo to start the analysis workflow.")

    st.markdown("---")

    log_left, log_right = st.columns([1.2, 1.0], gap="large")

    with log_left:
        st.markdown("""
        <div class="section-card">
            <div class="small-kicker">Session Log</div>
            <div class="section-title">Recent webcam analysis results</div>
            <div class="section-note">
                Each webcam analysis is stored in the current session for quick review and export.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.camera_log:
            log_df = pd.DataFrame(st.session_state.camera_log)
            st.dataframe(log_df, use_container_width=True, height=320)

            csv_data = log_df.to_csv(index=False).encode("utf-8")
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Export Camera Log (CSV)",
                    data=csv_data,
                    file_name=f"camera_session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="export_camera_log_csv"
                )
            with d2:
                if st.button("Clear Camera Log", key="clear_camera_log"):
                    st.session_state.camera_log = []
                    st.toast("Camera log cleared.")
                    st.rerun()
        else:
            st.caption("No webcam analysis results yet.")

    with log_right:
        st.markdown("""
        <div class="section-card">
            <div class="small-kicker">Webcam Demo Notes</div>
            <div class="section-title">What this mode demonstrates</div>
            <div class="section-note">
                This mode extends the trained model from static image upload to a practical
                webcam-based workflow: capture → detect face → classify → explain.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info(
            "This webcam mode is ideal for classroom demos because it shows that the model can be used in an interactive application."
        )
        st.warning(
            "For the most stable result, keep one clear frontal face in the camera view."
        )
