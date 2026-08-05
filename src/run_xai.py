import argparse
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models import get_model
from src.xai.grad_cam import GradCAMPlusPlus

def main():
    parser = argparse.ArgumentParser(description="Run XAI Grad-CAM++ Analysis")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="Proposed-Hybrid")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name=args.model, in_channels=config['dataset']['num_channels'], num_classes=config['dataset']['num_classes']).to(device)

    # Get target layer for Grad-CAM++
    target_layer = model.final_conv if hasattr(model, 'final_conv') else list(model.children())[-1]
    
    grad_cam = GradCAMPlusPlus(model=model, target_layer=target_layer)

    sample_input = torch.randn(1, 4, 192, 192).to(device)
    print("Generating Grad-CAM++ Heatmap for Enhancing Tumor (Class 3)...")

    heatmap = grad_cam.generate_heatmap(sample_input, target_class=3)
    
    output_dir = config['xai']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, "grad_cam_heatmap.png")
    
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="jet")
    plt.title(f"Grad-CAM++ Interpretability Map ({args.model})")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()

    print(f"XAI Heatmap successfully saved to: {heatmap_path}")

if __name__ == "__main__":
    main()
