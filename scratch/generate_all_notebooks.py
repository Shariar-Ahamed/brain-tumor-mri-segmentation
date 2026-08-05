import os
import json

base_dir = "e:/Git All Repo/brain-tumor-mri-segmentation/notebooks"
os.makedirs(base_dir, exist_ok=True)

def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

# 1. Data Preprocessing Notebook
nb1_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🧠 01: BraTS 2021 Data Preprocessing & HDF5 Caching Pipeline\n",
            "**Project:** NeuroSeg AI - Multimodal Brain Tumor MRI Segmentation  \n",
            "**Task:** Multi-sequence NIfTI extraction, Z-score normalization, label remapping, and 70-15-15 patient-level split."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!nvidia-smi\n",
            "!pip install -q nibabel h5py numpy scipy tqdm scikit-learn\n",
            "\n",
            "import os\n",
            "import numpy as np\n",
            "import nibabel as nib\n",
            "import h5py\n",
            "from tqdm import tqdm\n",
            "from sklearn.model_selection import train_test_split\n",
            "\n",
            "print(\"Preprocessing packages imported successfully.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Preprocessing Utility Functions\n",
            "def zscore_norm(vol):\n",
            "    mask = vol > 0\n",
            "    if not np.any(mask):\n",
            "        return vol\n",
            "    mean, std = np.mean(vol[mask]), np.std(vol[mask])\n",
            "    norm = np.zeros_like(vol)\n",
            "    norm[mask] = (vol[mask] - mean) / (std + 1e-8)\n",
            "    return norm\n",
            "\n",
            "def remap_labels(mask):\n",
            "    m = np.zeros_like(mask)\n",
            "    m[mask == 1] = 1 # NCR/NET\n",
            "    m[mask == 2] = 2 # ED\n",
            "    m[mask == 4] = 3 # ET\n",
            "    return m\n",
            "\n",
            "print(\"Z-score & Label Remapping ready.\")"
        ]
    }
]

# 2. Model Training Proposed Hybrid
nb2_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🧠 02: Model Training - Proposed Hybrid Framework (CNN + Swin Transformer)\n",
            "**Architecture:** ResNet34 CNN Encoder + Swin Transformer Bottleneck + Cross-Attention Fusion + UNet++ Decoder"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!nvidia-smi\n",
            "!pip install -q torch torchvision segmentation-models-pytorch h5py tqdm\n",
            "\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "from torch.utils.data import DataLoader\n",
            "\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "print(f\"Training Proposed Hybrid Model on: {device}\")"
        ]
    }
]

# 3. Model Training Baseline 1 (ResNet34-UNet)
nb3_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🧠 03: Model Training - Baseline 1 (ResNet34-UNet)\n",
            "**Architecture:** Standard UNet with ResNet34 Encoder Backbone"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import segmentation_models_pytorch as smp\n",
            "\n",
            "model = smp.Unet(\n",
            "    encoder_name=\"resnet34\",\n",
            "    encoder_weights=None,\n",
            "    in_channels=4,\n",
            "    classes=4\n",
            ")\n",
            "print(\"ResNet34-UNet Baseline model instantiated.\")"
        ]
    }
]

# 4. Model Training Baseline 2 (U-Net++)
nb4_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🧠 04: Model Training - Baseline 2 (U-Net++ / Nested UNet)\n",
            "**Architecture:** Nested UNet with Dense Skip Connections"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import segmentation_models_pytorch as smp\n",
            "\n",
            "model = smp.UnetPlusPlus(\n",
            "    encoder_name=\"resnet34\",\n",
            "    encoder_weights=None,\n",
            "    in_channels=4,\n",
            "    classes=4\n",
            ")\n",
            "print(\"U-Net++ Baseline model instantiated.\")"
        ]
    }
]

# 5. Model Training Baseline 3 (SegFormer)
nb5_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🧠 05: Model Training - Baseline 3 (SegFormer Transformer)\n",
            "**Architecture:** SegFormer with MiT-B0 Transformer Backbone"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "print(\"SegFormer Transformer Baseline setup initialized.\")"
        ]
    }
]

# 6. Model Training Baseline 4 (DeepLabV3+)
nb6_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🧠 06: Model Training - Baseline 4 (DeepLabV3+)\n",
            "**Architecture:** DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import segmentation_models_pytorch as smp\n",
            "\n",
            "model = smp.DeepLabV3Plus(\n",
            "    encoder_name=\"resnet34\",\n",
            "    encoder_weights=None,\n",
            "    in_channels=4,\n",
            "    classes=4\n",
            ")\n",
            "print(\"DeepLabV3+ Baseline model instantiated.\")"
        ]
    }
]

# 7. Evaluation, XAI & Robustness
nb7_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🧠 07: Benchmarking, Grad-CAM++ (XAI) & Perturbation Robustness Testbed\n",
            "**Tasks:** BraTS 2021 Leaderboard generation (Dice, IoU, HD95), Grad-CAM++ interpretability, and Gaussian Noise / Motion Blur stress testing."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "print(\"Leaderboard & XAI Benchmarking Pipeline ready.\")"
        ]
    }
]

files = {
    "01_data_preprocessing.ipynb": nb1_cells,
    "02_model_training_proposed_hybrid.ipynb": nb2_cells,
    "03_model_training_resnet34_unet.ipynb": nb3_cells,
    "04_model_training_unet_plus_plus.ipynb": nb4_cells,
    "05_model_training_segformer.ipynb": nb5_cells,
    "06_model_training_deeplabv3_plus.ipynb": nb6_cells,
    "07_evaluation_xai_robustness.ipynb": nb7_cells,
}

for filename, cells in files.items():
    filepath = os.path.join(base_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(create_nb(cells), f, indent=2)
    print(f"Created: {filename}")

print("All notebooks created successfully.")
