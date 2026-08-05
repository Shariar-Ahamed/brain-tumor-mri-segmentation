import argparse
import yaml
import torch
import numpy as np

from src.models import get_model
from src.utils.metrics import compute_dice_score, compute_iou_score, compute_precision_recall_specificity, compute_hd95

def main():
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Segmentation Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="Proposed-Hybrid")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name=args.model, in_channels=config['dataset']['num_channels'], num_classes=config['dataset']['num_classes']).to(device)

    if args.checkpoint and torch.cuda.is_available():
        model.load_state_dict(torch.load(args.checkpoint))
        print(f"Loaded checkpoint from: {args.checkpoint}")

    model.eval()
    print(f"Running Evaluation for Model: {args.model} on Device: {device}")

    # Generate synthetic batch for testing pipeline
    sample_input = torch.randn(4, 4, 192, 192).to(device)
    sample_target = torch.randint(0, 4, (4, 192, 192)).to(device)

    with torch.no_grad():
        outputs = model(sample_input)

    dice = compute_dice_score(outputs, sample_target)
    iou = compute_iou_score(outputs, sample_target)
    pr_rec = compute_precision_recall_specificity(outputs, sample_target)

    print("\n--- Evaluation Metrics Summary ---")
    print(f"Mean Tumor Dice Score: {dice['Mean_Tumor_Dice']:.4f}")
    print(f"Mean Tumor IoU Score:  {iou['Mean_Tumor_IoU']:.4f}")
    print("Per Class Dice:")
    for k, v in dice.items():
        if k != 'Mean_Tumor_Dice':
            print(f"  - {k}: {v:.4f}")

if __name__ == "__main__":
    main()
