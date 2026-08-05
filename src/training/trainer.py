import os
import time
import torch

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from ..utils.metrics import compute_dice_score, compute_iou_score

class Trainer:
    """
    Universal Deep Learning Training Engine for BraTS MRI Segmentation.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 amp=True, patience=10, checkpoint_dir="./checkpoints"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.amp = amp and (device != "cpu")
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir

        self.scaler = GradScaler('cuda', enabled=self.amp) if self.amp else None
        self.best_val_dice = 0.0
        self.patience_counter = 0

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for imgs, masks in tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]"):
            imgs, masks = imgs.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()

            if self.amp:
                with autocast():
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(imgs)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(len(self.train_loader), 1)
        return avg_loss, elapsed

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        count = 0

        with torch.no_grad():
            for imgs, masks in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                if self.amp:
                    with autocast():
                        outputs = self.model(imgs)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, masks)

                total_loss += loss.item()
                dice_metrics = compute_dice_score(outputs, masks)
                total_dice += dice_metrics['Mean_Tumor_Dice']
                count += 1

        avg_loss = total_loss / max(count, 1)
        avg_dice = total_dice / max(count, 1)
        return avg_loss, avg_dice

    def fit(self, num_epochs=30):
        print(f"Starting training for {num_epochs} epochs on device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            train_loss, train_time = self.train_epoch(epoch)
            val_loss, val_dice = self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mean Dice: {val_dice:.4f} | Time: {train_time:.1f}s")

            # Save latest checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_dice': val_dice,
            }, checkpoint_path)

            # Check best validation performance
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.patience_counter = 0
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                torch.save(self.model.state_dict(), best_path)
                print(f"--> Saved Best Model Checkpoint with Val Dice: {val_dice:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered at Epoch {epoch}. Best Val Dice: {self.best_val_dice:.4f}")
                    break

        return self.best_val_dice
