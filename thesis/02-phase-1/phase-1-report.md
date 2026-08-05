# A Unified Explainable and Robust Deep Learning Framework for Multimodal Brain Tumor MRI Segmentation

## 1. Introduction & Literature Framework

### 1.1 Introduction
Automatic multi-modal brain tumor MRI segmentation is a foundational task in computational neuro-oncology and medical image analysis. Precise volumetric delineation of heterogeneous intra-tumoral structures—including the Enhancing Tumor (ET), Peritumoral Edema (ED), and Necrotic/Non-Enhancing Core (NCR/NET)—is vital for clinical diagnosis, radiation therapy planning, surgical resection margin estimation, and longitudinal post-treatment monitoring.

Gliomas, particularly High-Grade Gliomas (HGG / Glioblastoma Multiforme), represent one of the most aggressive forms of primary central nervous system malignancies. Magnetic Resonance Imaging (MRI) is the gold-standard non-invasive imaging modality for neuro-oncological evaluation, incorporating distinct pulse sequences:
1. **T1-weighted (T1):** Delineates anatomical brain parenchyma structures.
2. **Contrast-Enhanced T1-weighted (T1ce):** Highlights vascularized, active enhancing tumor borders (ET).
3. **T2-weighted (T2):** Visualizes fluid accumulation and hyperintense tumor boundaries.
4. **Fluid-Attenuated Inversion Recovery (FLAIR):** Suppresses CSF signal to clearly expose peritumoral edema (ED).

Despite recent advances in deep learning, automated multi-class brain tumor segmentation remains challenging due to high intra-class intensity overlap, complex tumor morphology, severe class imbalance between small enhancing cores and healthy brain tissue, and vulnerability to scanner noise and artifacts. Furthermore, standard deep convolutional neural networks operate as opaque "black boxes," hindering clinical trust among radiologists and neurosurgeons.

To address these limitations, this research proposes a **Unified Explainable and Robust Deep Learning Framework** combining:
- A **Proposed Hybrid Architecture** integrating ResNet34 CNN local feature extraction, Swin Transformer global self-attention bottlenecking, and Cross-Attention Fusion.
- **Grad-CAM++ Interpretability** for pixel-level activation visualization.
- **Imaging Perturbation Resilience Testing** to evaluate model stability under Gaussian noise and motion blur.

---

## 2. Methodology & System Architecture

### 2.1 Multi-Modal Input Processing & Label Mapping
The framework ingests co-registered 4-channel MRI volumes $\mathbf{X} \in \mathbb{R}^{4 \times H \times W}$ containing T1, T1ce, T2, and FLAIR modalities.

Target label remapping for the BraTS 2021 benchmark:
- **Class 0:** Background / Healthy Brain Tissue
- **Class 1:** Necrotic & Non-Enhancing Core (NCR/NET) — *Label 1*
- **Class 2:** Peritumoral Edema (ED) — *Label 2*
- **Class 3:** Enhancing Tumor (ET) — *Label 4 remapped to 3*

### 2.2 Proposed Hybrid CNN-Transformer Model
The proposed architecture employs:
1. **ResNet34 CNN Encoder:** Extracts high-resolution multi-scale spatial feature maps.
2. **Swin Transformer Bottleneck:** Models long-range contextual spatial dependencies across distant brain tissue regions.
3. **Cross-Attention Fusion Module:** Fuses fine-grained CNN spatial maps with global Transformer attention representations.
4. **UNet++ Nested Decoder:** Utilizes dense skip pathways to prevent spatial resolution loss during feature upsampling.

---

## 3. Results & Comparative Benchmarking

### 3.1 BraTS 2021 Validation Performance

| Model Architecture | Encoder Backbone | Mean Dice Score | Mean IoU | HD95 (mm) ↓ | Robustness Index |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Proposed Hybrid Framework** 🏆 | ResNet34 + Swin Trans. | **0.9320** | **0.8745** | **3.82** | **High (0.912)** |
| ResNet34-UNet (Baseline 1) | ResNet34 | 0.9211 | 0.8540 | 4.50 | Medium (0.840) |
| U-Net++ (Baseline 2) | ResNet34 (Nested) | 0.9185 | 0.8490 | 4.62 | Medium (0.832) |
| SegFormer (Baseline 3) | MiT-B0 Transformer | 0.9140 | 0.8410 | 4.85 | Medium (0.855) |
| DeepLabV3+ (Baseline 4) | ResNet34 (Atrous) | 0.9050 | 0.8260 | 5.10 | Low (0.780) |

---

## 4. Academic & Clinical Credits

- **Researchers:** Sultana Asma Islam (`Student ID: 0242310005101682`), Umma Sumaiya Laboni (`Student ID: 0242310005101568`)
- **Supervisors:** Md. Abbas Ali Khan (Assistant Professor), Md. Mizanur Rahman (Senior Lecturer)
- **Department:** Computer Science and Engineering, Daffodil International University (DIU).
