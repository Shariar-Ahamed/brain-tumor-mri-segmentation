# 🧠 NeuroSeg AI: A Unified Explainable and Robust Deep Learning Framework for Multimodal Brain Tumor MRI Segmentation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://brain-tumor-mri-segmentation-diu.streamlit.app/)
[![Python Version](https://img.shields.io/badge/python-3.10.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![BraTS 2021](https://img.shields.io/badge/Benchmark-BraTS%202021-00f2fe.svg)](https://www.synapse.org/brats2021)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, research-grade deep learning framework and interactive web platform developed for Senior Year Thesis (FYDP) research at **Daffodil International University (Department of Computer Science & Engineering)**.

This framework unifies **high-accuracy multi-modal MRI brain tumor segmentation** (ResNet34 + Swin Transformer + Cross-Attention Fusion), **Explainable AI (Grad-CAM++)**, and **robustness benchmarking** against clinical imaging perturbations.

---

## 🌟 Key Features

* **Multi-Modal 4-Channel Input Processing:** Accepts co-registered `T1`, `T1ce`, `T2`, and `FLAIR` MRI sequences.
* **4-Class Multiclass Tumor Segmentation:**
  * Class 0: Background
  * Class 1: Necrotic and Non-Enhancing Tumor Core (NCR/NET)
  * Class 2: Peritumoral Edema (ED)
  * Class 3: Enhancing Tumor (ET)
* **Proposed Hybrid CNN-Transformer Architecture:**
  * **CNN Encoder (ResNet34):** Extracts fine local spatial texture features.
  * **Transformer Bottleneck (Self-Attention):** Captures global long-range contextual relationships.
  * **Cross-Attention Fusion Module:** Fuses local CNN features with global Transformer attention representations.
  * **UNet++ Nested Decoder:** Dense skip connections for precise tumor boundary delineation.
* **Explainable AI (Grad-CAM++):** Integrated interpretability engine to visualize feature maps driving class-specific tumor predictions.
* **Robustness & Perturbation Testbed:** Evaluates performance decay curves under Gaussian Noise, Motion Blur, Brightness Shift, and Low-Resolution Downsampling.
* **Interactive Web Interface:** Glassmorphic Streamlit web studio for live axial slice inspection, Grad-CAM++ controls, and model benchmarking.

---

## 📁 Repository Structure

```text
brain-tumor-mri-segmentation/
├── .streamlit/
│   └── config.toml               # Streamlit Theme Configuration (Dark & Cyan Accent)
├── app/
│   ├── app.py                    # Main Streamlit Dashboard Entrypoint
│   ├── style.css                 # Glassmorphic Custom Styling System
│   └── xai_methods.py            # Grad-CAM++ Interpretability Engine Logic
├── data/                         # Data directory (Dataset samples / Cache)
├── models/                       # Trained PyTorch Model Checkpoints (.pt)
├── notebooks/                    # Jupyter Notebooks for Experiments
├── src/                          # Deep Learning Core Architecture Engine
│   ├── dataset/                  # BraTS 2021 Loader & Preprocessing
│   ├── models/                   # Proposed Hybrid (CNN + Transformer) & Baselines
│   ├── training/                 # PyTorch Trainer Engine & Combined Dice Loss
│   └── xai/                      # Grad-CAM++ Core Module
├── thesis/
│   └── 1. Title_Phase_Evaluation_Report.pdf  # Thesis Documentation Report
├── .python-version               # Fixed Python Version (3.10.12)
├── packages.txt                  # Linux System Dependencies (libgl1)
├── requirements.txt              # PyTorch CPU & Streamlit Stable Dependencies
├── README.md                     # Research Project Overview & Setup Guide
├── app.py                        # Master App Launcher
└── streamlit_app.py              # Cloud Entrypoint
```

---


## 📊 Benchmark Leaderboard (BraTS 2021 Validation)

| Model Architecture | Encoder Backbone | Mean Dice Score | Mean IoU | HD95 (mm) ↓ | Robustness Index |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Proposed Hybrid Framework** 🏆 | ResNet34 + Swin Trans. | **0.9320** | **0.8745** | **3.82** | **High (0.912)** |
| ResNet34-UNet (Baseline 1) | ResNet34 | 0.9211 | 0.8540 | 4.50 | Medium (0.840) |
| U-Net++ (Baseline 2) | ResNet34 (Nested) | 0.9185 | 0.8490 | 4.62 | Medium (0.832) |
| SegFormer (Baseline 3) | MiT-B0 Transformer | 0.9140 | 0.8410 | 4.85 | Medium (0.855) |
| DeepLabV3+ (Baseline 4) | ResNet34 (Atrous) | 0.9050 | 0.8260 | 5.10 | Low (0.780) |

---

## 🚀 Quick Start & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
# 1. Clone the Repository
git clone https://github.com/Shariar-Ahamed/brain-tumor-mri-segmentation.git
cd brain-tumor-mri-segmentation

# 2. Create and activate a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Launch Interactive Web App
streamlit run app.py
```

---

## 🎓 Research Team & Academic Credits

* **Researchers:** 
  * Sultana Asma Islam (`Student ID: 0242310005101682`)
  * Umma Sumaiya Laboni (`Student ID: 0242310005101568`)
* **Supervisors:** 
  * Md. Abbas Ali Khan (Supervisor | Assistant Professor, Dept. of CSE)
  * Md. Mizanur Rahman (Co-Supervisor | Senior Lecturer, Dept. of CSE)
* **Institution:** Department of Computer Science & Engineering, Daffodil International University (DIU).