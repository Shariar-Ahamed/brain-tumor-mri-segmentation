import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.seed import seed_everything
from src.models import get_model
from src.training.losses import CombinedDiceCELoss
from src.training.trainer import Trainer
from src.dataset.brats_dataset import BraTSDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train Brain Tumor MRI Segmentation Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Override model name from config")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch with synthetic data for verification")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config['dataset'].get('seed', 42)
    seed_everything(seed)

    model_name = args.model if args.model else config['models']['selected_model']
    print(f"==================================================")
    print(f"Training Framework Initialized for Model: {model_name}")
    print(f"==================================================")

    # Initialize model
    model = get_model(
        model_name=model_name,
        in_channels=config['dataset']['num_channels'],
        num_classes=config['dataset']['num_classes']
    )

    # Device & Optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['lr']), weight_decay=float(config['training']['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    criterion = CombinedDiceCELoss()

    if args.dry_run or not os.path.exists(config['dataset']['h5_path']):
        print("Note: Running with synthetic data loader for testing/dry-run...")
        B_train, B_val = 16, 8
        C, H, W = config['dataset']['num_channels'], config['dataset']['image_size'][0], config['dataset']['image_size'][1]

        synthetic_train_imgs = torch.randn(B_train, C, H, W)
        synthetic_train_masks = torch.randint(0, config['dataset']['num_classes'], (B_train, H, W))
        synthetic_val_imgs = torch.randn(B_val, C, H, W)
        synthetic_val_masks = torch.randint(0, config['dataset']['num_classes'], (B_val, H, W))

        train_dataset = TensorDataset(synthetic_train_imgs, synthetic_train_masks)
        val_dataset = TensorDataset(synthetic_val_imgs, synthetic_val_masks)

        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        epochs = 1 if args.dry_run else 5
    else:
        # Load real dataset from HDF5 cache
        train_dataset = BraTSDataset(h5_filepath=config['dataset']['h5_path'])
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        epochs = config['training']['epochs']

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        amp=config['training']['amp'],
        patience=config['training']['early_stopping_patience'],
        checkpoint_dir=config['training']['checkpoint_dir']
    )

    best_dice = trainer.fit(num_epochs=epochs)
    print(f"\nTraining completed! Best Validation Dice Score: {best_dice:.4f}")

if __name__ == "__main__":
    main()
