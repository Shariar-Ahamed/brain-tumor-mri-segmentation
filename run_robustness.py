import argparse
import os
import yaml
import torch
import numpy as np

from src.models import get_model
from src.robustness.perturbations import apply_gaussian_noise, apply_motion_blur, apply_brightness_shift, apply_low_resolution
from src.utils.metrics import compute_dice_score

def main():
    parser = argparse.ArgumentParser(description="Run Robustness Perturbation Benchmarks")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="Proposed-Hybrid")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name=args.model, in_channels=config['dataset']['num_channels'], num_classes=config['dataset']['num_classes']).to(device)
    model.eval()

    print(f"==================================================")
    print(f"Robustness Benchmarking for Model: {args.model}")
    print(f"==================================================")

    # Clean Synthetic Sample
    clean_sample_np = np.random.randn(4, 192, 192).astype(np.float32)
    clean_target_tensor = torch.randint(0, 4, (192, 192)).to(device)

    # 1. Clean Evaluation
    with torch.no_grad():
        clean_input_tensor = torch.tensor(clean_sample_np).unsqueeze(0).to(device)
        clean_out = model(clean_input_tensor)
        clean_dice = compute_dice_score(clean_out, clean_target_tensor.unsqueeze(0))['Mean_Tumor_Dice']
        print(f"Clean Baseline Dice: {clean_dice:.4f}")

    # 2. Gaussian Noise Perturbation
    for level in config['robustness']['noise_levels']:
        noisy_sample_np = apply_gaussian_noise(clean_sample_np, noise_level=level)
        with torch.no_grad():
            noisy_input_tensor = torch.tensor(noisy_sample_np).unsqueeze(0).to(device)
            noisy_out = model(noisy_input_tensor)
            noisy_dice = compute_dice_score(noisy_out, clean_target_tensor.unsqueeze(0))['Mean_Tumor_Dice']
            drop = (clean_dice - noisy_dice)
            print(f"Noise Level {level:.2f} | Dice: {noisy_dice:.4f} (Drop: {drop:+.4f})")

    # 3. Motion Blur Perturbation
    blurred_sample_np = apply_motion_blur(clean_sample_np, kernel_size=5)
    with torch.no_grad():
        blurred_input_tensor = torch.tensor(blurred_sample_np).unsqueeze(0).to(device)
        blurred_out = model(blurred_input_tensor)
        blurred_dice = compute_dice_score(blurred_out, clean_target_tensor.unsqueeze(0))['Mean_Tumor_Dice']
        print(f"Motion Blur (k=5) | Dice: {blurred_dice:.4f} (Drop: {clean_dice - blurred_dice:+.4f})")

    # 4. Low Resolution Perturbation
    low_res_sample_np = apply_low_resolution(clean_sample_np, downscale_factor=2)
    with torch.no_grad():
        low_res_input_tensor = torch.tensor(low_res_sample_np).unsqueeze(0).to(device)
        low_res_out = model(low_res_input_tensor)
        low_res_dice = compute_dice_score(low_res_out, clean_target_tensor.unsqueeze(0))['Mean_Tumor_Dice']
        print(f"Low Resolution (2x) | Dice: {low_res_dice:.4f} (Drop: {clean_dice - low_res_dice:+.4f})")

    print("\nRobustness Benchmarking completed successfully!")

if __name__ == "__main__":
    main()
