import json
import os
from tqdm import tqdm
import torch
import time

from utils.plot import plot_curves

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        dev_loader,
        evaluator,
        device,
        log_path,
        checkpoint_dir=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.evaluator = evaluator
        self.device = device
        self.log_path = log_path
        self.logs = []
        # Directory to store separate checkpoints for model / optimizer / scheduler
        if checkpoint_dir is not None:
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = os.path.join(os.path.dirname(self.log_path), "checkpoints")

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                images=batch["images"],
                labels=batch["labels"],
            )

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def save_checkpoint(self, epoch):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        for f in os.listdir(self.checkpoint_dir):
            fpath = os.path.join(self.checkpoint_dir, f)
            if not os.path.isfile(fpath):
                continue
            if (
                f.startswith("model_epoch_")
                or f.startswith("optimizer_epoch_")
                or f.startswith("scheduler_epoch_")
            ) and not f.endswith(f"epoch_{epoch}.pth"):
                try:
                    os.remove(fpath)
                except OSError:
                    pass

        model_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
        optimizer_path = os.path.join(self.checkpoint_dir, f"optimizer_epoch_{epoch}.pth")
        scheduler_path = os.path.join(self.checkpoint_dir, f"scheduler_epoch_{epoch}.pth")

        # Remove older checkpoints to save disk space
        for fname in os.listdir(self.checkpoint_dir):
            fpath = os.path.join(self.checkpoint_dir, fname)
            if not os.path.isfile(fpath):
                continue
            # Keep only current epoch's checkpoints
            if (
                fname.startswith("model_epoch_")
                or fname.startswith("optimizer_epoch_")
                or fname.startswith("scheduler_epoch_")
            ) and not fname.endswith(f"epoch_{epoch}.pth"):
                try:
                    os.remove(fpath)
                except OSError:
                    pass

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), scheduler_path)

        print(
            f"Checkpoint saved for epoch {epoch}:\n"
            f"  model -> {model_path}\n"
            f"  optimizer -> {optimizer_path}\n"
            f"  scheduler -> {scheduler_path if self.scheduler is not None else 'N/A'}"
        )

    def train(self, epochs, early_stopping=None, save_best_path=None):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            dev_metrics = self.evaluator.evaluate(self.dev_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]

            log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "learning_rate": current_lr,
                "dev_EM": dev_metrics["EM"],
                "dev_F1": dev_metrics["F1"]
            }

            self.logs.append(log)

            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "w") as f:
                json.dump(self.logs, f, indent=2)

            print(log)

            # Always save separate checkpoints for this epoch
            self.save_checkpoint(epoch + 1)

            if early_stopping is not None:
                stop, improved = early_stopping.step(dev_metrics["F1"])

                if improved and save_best_path is not None:
                    best_epoch = epoch + 1
                    self.save_model(save_best_path)
                    print(
                        f"New best F1: {early_stopping.best_f1:.4f} "
                        f"(epoch {best_epoch})"
                    )

                if stop:
                    print(
                        f"Early stopping at epoch {epoch+1} | "
                        f"Best F1: {early_stopping.best_f1:.4f} "
                        f"(epoch {best_epoch})"
                    )
                    break
        
        plot_curves(self.log_path, self.log_path.replace(".json", ".png"))
        print(f"Training logs and plots saved to {self.log_path}")

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        model_dir = os.path.dirname(save_path)
        self.model.config.save_pretrained(model_dir)
        print(f"Model configuration saved to {model_dir}")
        
    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}")

    def get_logs(self):
        return self.logs
