# A Unified Explainable and Robust Deep Learning Framework for Multimodal Brain Tumor MRI Segmentation 🧠

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![BraTS 2021](https://img.shields.io/badge/Dataset-BraTS%202021-green.svg)](https://www.synapse.org/#!Synapse:syn25829070/wiki/610863)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, research-grade deep learning framework and interactive web platform developed for Master's/Undergraduate thesis research at **Daffodil International University (Department of Computer Science & Engineering)**.

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
* **Interactive Web Interface:** Glassmorphic HTML5/CSS3/JS Web Application with live axial slice slider, mask toggles, Grad-CAM++ controls, and robustness sliders.

---

## 📁 Repository Structure

```text
brain-tumor-mri-segmentation/
│
├── configs/
│   └── config.yaml               # Training, dataset & model hyperparameters
│
├── src/
│   ├── dataset/                  # Preprocessing, patient-level split & HDF5 cache
│   ├── models/                   # Proposed Hybrid & Baseline models (UNet++, SegFormer, DeepLabV3+)
│   ├── training/                 # Universal PyTorch trainer with AMP, Early Stopping & Checkpoints
│   ├── xai/                      # Grad-CAM++ interpretability module
│   ├── robustness/               # Image perturbation testing engine (Noise, Blur, Low-Res)
│   └── utils/                    # Dice, IoU, HD95, Precision, Recall, Specificity metrics
│
├── assets/                       # UI assets and hero showcase banner
├── index.html                    # Interactive Web Dashboard UI
├── styles.css                    # Glassmorphism dark CSS design system
├── app.js                        # Web application logic and canvas renderer
│
├── train.py                      # Main model training entry point
├── evaluate.py                   # Model quantitative benchmark script
├── run_xai.py                    # Grad-CAM++ heatmap generation script
├── run_robustness.py             # Robustness perturbation benchmark script
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
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
git clone https://github.com/Shariar-Ahamed/brain-tumor-mri-segmentation.git
cd brain-tumor-mri-segmentation
pip install -r requirements.txt
```

### 2. Run Dry-Run Training Test
Verify model forward pass, loss calculation, and checkpointing:
```bash
python train.py --dry-run
```

### 3. Full Model Training
Train on preprocessed BraTS 2021 dataset:
```bash
python train.py --config configs/config.yaml --model Proposed-Hybrid
```

### 4. Evaluate Performance Metrics
Compute Dice Score, IoU, HD95, Precision, Recall, and Specificity:
```bash
python evaluate.py --model Proposed-Hybrid
```

### 5. Generate Grad-CAM++ Interpretability Maps
```bash
python run_xai.py --model Proposed-Hybrid
```

### 6. Run Robustness Benchmarking
```bash
python run_robustness.py --model Proposed-Hybrid
```

### 7. Launch Interactive Web Interface
Open `index.html` directly in any web browser, or launch via HTTP server:
```bash
python -m http.server 8000
```
Then visit `http://localhost:8000` in your browser.

---

## 👥 Thesis Team & Supervisors

**Department of Computer Science and Engineering**  
**Daffodil International University (DIU)**

* **Student Researchers:**
  * Sultana Asma Islam (Student ID: `0242310005101682`)
  * Umma Sumaiya Laboni (Student ID: `0242310005101568`)
* **Supervisors:**
  * **Md. Abbas Ali Khan** (Supervisor | Assistant Professor, Dept. of CSE)
  * **Md. Mizanur Rahman** (Co-Supervisor | Senior Lecturer, Dept. of CSE)

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.