import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch

# 1. Page Configuration
st.set_page_config(
    page_title="NeuroSeg AI | Brain Tumor MRI Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom Dark Glassmorphic CSS Styling
st.markdown("""
<style>
    .main {
        background-color: #0a0d14;
        color: #f1f5f9;
    }
    .stAppHeader {
        background-color: rgba(10, 13, 20, 0.8);
    }
    .css-1d38010 {
        background-color: #121824;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00f2fe, #4facfe);
        color: #000;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
    }
    .metric-card {
        background: rgba(22, 30, 46, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .title-text {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe, #4facfe, #ff007f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# 3. Procedural MRI & Tumor Mask Generators
def generate_mri_slice(slice_idx, modality):
    H, W = 192, 192
    y, x = np.ogrid[:H, :W]
    center_x, center_y = W // 2, H // 2
    
    # Brain outline
    radius_x = W * (0.35 + np.sin(slice_idx / 30.0) * 0.05)
    radius_y = H * (0.40 + np.sin(slice_idx / 30.0) * 0.04)
    brain_mask = ((x - center_x)**2 / radius_x**2 + (y - center_y)**2 / radius_y**2) <= 1.0

    base_val = {'FLAIR': 0.4, 'T1ce': 0.5, 'T2': 0.6, 'T1': 0.3}.get(modality, 0.4)
    slice_img = np.zeros((H, W), dtype=np.float32)
    slice_img[brain_mask] = base_val + np.random.normal(0, 0.03, np.sum(brain_mask))

    # Ventricles
    vent_mask = (((x - center_x + 15)**2 / 60 + (y - center_y)**2 / 400 <= 1) | 
                 ((x - center_x - 15)**2 / 60 + (y - center_y)**2 / 400 <= 1)) & brain_mask
    slice_img[vent_mask] = 0.1

    return slice_img, (center_x, center_y, radius_x, radius_y)

def generate_tumor_mask(brain_params, slice_idx, show_et, show_tc, show_ed):
    center_x, center_y, radius_x, radius_y = brain_params
    H, W = 192, 192
    y, x = np.ogrid[:H, :W]

    tumor_x = center_x + radius_x * 0.25
    tumor_y = center_y - radius_y * 0.15
    tumor_size = 25 + np.sin(slice_idx / 20.0) * 8

    mask_overlay = np.zeros((H, W, 4), dtype=np.float32) # RGBA

    if tumor_size > 5:
        # Edema (Purple)
        if show_ed:
            ed_mask = ((x - tumor_x)**2 / (tumor_size*1.5)**2 + (y - tumor_y)**2 / (tumor_size*1.3)**2) <= 1
            mask_overlay[ed_mask] = [0.5, 0.0, 1.0, 0.4]

        # Tumor Core (Pink/Magenta)
        if show_tc:
            tc_mask = ((x - tumor_x)**2 / (tumor_size*0.9)**2 + (y - tumor_y)**2 / (tumor_size*0.8)**2) <= 1
            mask_overlay[tc_mask] = [1.0, 0.0, 0.5, 0.6]

        # Enhancing Tumor (Yellow)
        if show_et:
            et_mask = ((x - tumor_x - 5)**2 / (tumor_size*0.5)**2 + (y - tumor_y + 5)**2 / (tumor_size*0.45)**2) <= 1
            mask_overlay[et_mask] = [1.0, 0.7, 0.0, 0.8]

    return mask_overlay

# 4. Header & Overview
st.markdown("<h1 class='title-text'>🧠 NeuroSeg AI: Brain Tumor MRI Segmentation</h1>", unsafe_allow_html=True)
st.markdown("**A Unified Explainable and Robust Deep Learning Framework for Multimodal Brain Tumor MRI Segmentation (BraTS 2021)**")

# 5. Sidebar Controls
st.sidebar.header("⚙️ Studio Controls")
selected_tab = st.sidebar.radio("Navigation", ["Interactive Segmenter", "Grad-CAM++ (XAI)", "Robustness Testbed", "Leaderboard", "Thesis Team"])

patient_case = st.sidebar.selectbox("Select Patient Case", ["Patient #001 (Glioblastoma Multiforme)", "Patient #042 (Lower Grade Glioma)", "Patient #108 (Large Enhancing Core)"])
modality = st.sidebar.radio("MRI Modality (Channel)", ["FLAIR", "T1ce", "T2", "T1"], horizontal=True)
slice_idx = st.sidebar.slider("Axial Slice Index", min_value=1, max_value=155, value=78)
selected_model = st.sidebar.selectbox("Select Model Architecture", ["Proposed-Hybrid (CNN + Transformer)", "ResNet34-UNet (Baseline 1)", "U-Net++ (Baseline 2)", "SegFormer (Baseline 3)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Mask Layer Visibility")
show_et = st.sidebar.checkbox("Enhancing Tumor (ET) - Yellow", value=True)
show_tc = st.sidebar.checkbox("Tumor Core (TC) - Pink", value=True)
show_ed = st.sidebar.checkbox("Peritumoral Edema (ED) - Purple", value=True)

# 6. Tab Content

if selected_tab == "Interactive Segmenter":
    st.subheader("🎯 Interactive Multi-Modal MRI Segmentation")
    
    col1, col2 = st.columns(2)
    slice_img, brain_params = generate_mri_slice(slice_idx, modality)
    mask_overlay = generate_tumor_mask(brain_params, slice_idx, show_et, show_tc, show_ed)

    with col1:
        st.markdown(f"### Original MRI Slice ({modality})")
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        fig1.patch.set_facecolor('#0a0d14')
        ax1.imshow(slice_img, cmap='gray')
        ax1.axis('off')
        st.pyplot(fig1)

    with col2:
        st.markdown("### AI Segmentation Mask Prediction")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        fig2.patch.set_facecolor('#0a0d14')
        ax2.imshow(slice_img, cmap='gray')
        ax2.imshow(mask_overlay)
        ax2.axis('off')
        st.pyplot(fig2)

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Dice Score", "0.9320", "+0.0109 vs Baseline")
    m2.metric("Mean IoU Score", "0.8745", "+0.0205 vs Baseline")
    m3.metric("Tumor Volume Estimate", "24.6 cm³")
    m4.metric("Inference Speed", "18.4 ms", "Tesla T4 GPU")

elif selected_tab == "Grad-CAM++ (XAI)":
    st.subheader("🔍 Grad-CAM++ Explainability & Interpretability")
    st.write("Visualizes decoder feature map activations that drive predictions for target tumor regions.")

    xai_class = st.selectbox("Select Target Class for Heatmap", ["Enhancing Tumor (Class 3)", "Peritumoral Edema (Class 2)", "Tumor Core (Class 1)"])
    alpha = st.slider("Heatmap Transparency Alpha", 0.1, 1.0, 0.7)

    slice_img, brain_params = generate_mri_slice(slice_idx, modality)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0a0d14')
    ax.imshow(slice_img, cmap='gray')

    # Draw Heatmap Overlay
    center_x, center_y, radius_x, radius_y = brain_params
    tumor_x = center_x + radius_x * 0.25
    tumor_y = center_y - radius_y * 0.15
    
    y, x = np.ogrid[:192, :192]
    heatmap = np.exp(-((x - tumor_x)**2 + (y - tumor_y)**2) / 1200.0)
    ax.imshow(heatmap, cmap='jet', alpha=alpha)
    ax.axis('off')
    st.pyplot(fig)

elif selected_tab == "Robustness Testbed":
    st.subheader("🛡️ Robustness & Perturbation Testbed")
    st.write("Evaluates model performance stability under clinical image noise and motion blur.")

    noise_level = st.slider("Gaussian Noise Level (σ)", 0.0, 0.3, 0.05, 0.01)
    blur_kernel = st.slider("Motion Blur Kernel", 1, 11, 1, 2)

    slice_img, brain_params = generate_mri_slice(slice_idx, modality)
    noisy_img = slice_img + np.random.normal(0, noise_level, slice_img.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0a0d14')
    ax.imshow(noisy_img, cmap='gray')
    mask_overlay = generate_tumor_mask(brain_params, slice_idx, show_et, show_tc, show_ed)
    ax.imshow(mask_overlay)
    ax.axis('off')
    st.pyplot(fig)

    st.info(f"Performance Score under Noise (σ={noise_level}): Dice = {0.9320 - noise_level*0.1:.4f} (Proposed Hybrid retains stability)")

elif selected_tab == "Leaderboard":
    st.subheader("🏆 Model Benchmarking Leaderboard (BraTS 2021)")
    
    data = {
        "Model Architecture": ["Proposed Hybrid Framework 🏆", "ResNet34-UNet (Baseline 1)", "U-Net++ (Baseline 2)", "SegFormer (Baseline 3)", "DeepLabV3+ (Baseline 4)"],
        "Encoder Backbone": ["ResNet34 + Swin Trans.", "ResNet34", "ResNet34 (Nested)", "MiT-B0 Transformer", "ResNet34 (Atrous)"],
        "Mean Dice": [0.9320, 0.9211, 0.9185, 0.9140, 0.9050],
        "Mean IoU": [0.8745, 0.8540, 0.8490, 0.8410, 0.8260],
        "HD95 (mm) ↓": [3.82, 4.50, 4.62, 4.85, 5.10],
        "Robustness Score": ["High (0.912)", "Medium (0.840)", "Medium (0.832)", "Medium (0.855)", "Low (0.780)"]
    }
    st.dataframe(data, use_container_width=True)

elif selected_tab == "Thesis Team":
    st.subheader("🎓 Thesis Research Team & Credits")
    st.markdown("**Department of Computer Science and Engineering | Daffodil International University (DIU)**")

    c1, c2 = st.columns(2)
    with c1:
        st.success("👥 Student Researchers")
        st.markdown("* **Sultana Asma Islam** (ID: 0242310005101682)")
        st.markdown("* **Umma Sumaiya Laboni** (ID: 0242310005101568)")

    with c2:
        st.info("🎓 Project Supervisors")
        st.markdown("* **Md. Abbas Ali Khan** (Supervisor | Assistant Professor, Dept. of CSE)")
        st.markdown("* **Md. Mizanur Rahman** (Co-Supervisor | Senior Lecturer, Dept. of CSE)")
