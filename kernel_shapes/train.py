import multiprocessing
import platform
import random
import time

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

import wandb
from .esc50_dataset import ESC50Dataset
from .sound_cnn import SoundCNN
from .spec_augment import SpecAugment


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, optimizer, criterion, device, spec_aug):
    model.train()  # set model to training mode
    running_loss = 0.0

    for batch in train_loader:
        features = batch["data"].to(device)
        if spec_aug:
            features = spec_aug(features)
        labels = batch["label"].to(device)
        # If features are spectrograms with shape (batch, n_mels, time), add channel dim:
        if features.ndim == 3:
            features = features.unsqueeze(1)  # shape: (batch, 1, n_mels, time)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Compute average training loss
    avg_train_loss = running_loss / len(train_loader)
    return avg_train_loss


def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            features = batch["data"].to(device)
            labels = batch["label"].to(device)
            if features.ndim == 3:
                features = features.unsqueeze(1)
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # Compute accuracy
            _, predicted = torch.max(outputs, dim=1)  # predicted class index
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    return avg_val_loss, val_acc


def run(kernel_params, max_epochs, min_epochs=20, patience=10, seed=0, use_spec_aug=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    set_seed(seed)
    fold_results = []

    for fold in range(1, 6):
        run = wandb.init(project="esc50", name=f"{kernel_params[0]}_fold_{fold}_seed_{seed}")
        wandb.define_metric("epoch")
        print(f"running {kernel_params[0]}, holdout fold {fold}")
        train_dataset = ESC50Dataset(
            root_dir="ESC-50-master/audio",
            meta_csv="ESC-50-master/meta/esc50.csv",
            fold=fold,
            exclude=True,
            precomputed=True,
        )

        val_dataset = ESC50Dataset(
            root_dir="ESC-50-master/audio",
            meta_csv="ESC-50-master/meta/esc50.csv",
            fold=fold,
            exclude=False,
            precomputed=True,
        )

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
        model = SoundCNN(num_classes=50, kernel_size=kernel_params[1]).to(device)
        if use_spec_aug:
            spec_aug = SpecAugment(time_mask_param=15, freq_mask_param=15, num_masks=2)
        else:
            spec_aug = None

        summary(model, input_size=(128, 1, 128, 862))

        # wandb.define_metric("train/*", step_metric="epoch")
        # wandb.define_metric("val/*", step_metric="epoch")

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        # Early stopping variables
        min_epochs = 30
        best_val_acc = 0.0
        best_epoch = 0
        no_improve_count = 0
        best_model_path = f"best_model_{kernel_params[0]}_{seed}.pt"
        epoch = -1 # make pyright happy
        # Training loop
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            avg_train_loss = train_epoch(model, train_loader, optimizer, criterion, device, spec_aug)
            avg_val_loss, val_acc = validate(model, val_loader, device, criterion)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{max_epochs} ({(time.time() - epoch_start_time):.2f}s) - "
                    f"Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}"
                )

            wandb.log(
                {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"]
                }
            )

            scheduler.step(val_acc)

            # Early stopping check

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                no_improve_count = 0
                # Save best model
                torch.save(model.state_dict(), best_model_path)
                # print(f"New best model saved with validation accuracy: {val_acc:.4f}")
            else:
                no_improve_count += 1
                # print(f"No improvement for {no_improve_count} epochs (best: {best_val_acc:.4f} at epoch {best_epoch+1})")

            if no_improve_count >= patience and epoch > min_epochs:
                # print(f"Early stopping triggered after {epoch+1} epochs")
                break

        fold_results.append(best_val_acc)
        print(f"Best val acc for fold {fold}: {best_val_acc:.4f} at epoch {best_epoch + 1}/{epoch + 1}")
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.summary["best_epoch"] = best_epoch + 1
        wandb.summary["epochs_run"] = epoch + 1
        run.finish()
        print(fold_results)
    final_results = sum(fold_results)/len(fold_results)

    return final_results

if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn", force=True)
    for seed in [13,43,55]:
        kernel_params_list = [
            ("3x3", (3,3)),
            ("3x5", (3,5)),
            ("3x7", (3,7)),
            ("3x9", (3,9)),
            ("3x11", (3,11)),
        ]
        max_epochs = 100
        patience = 100
        min_epochs = 100
        print("-" * 50)
        print("STARTING RUN WITH SEED:", seed)
        for kernel_params in kernel_params_list:
            print(f"Training with kernel: {kernel_params[0]}")
            avg_val_acc = run(kernel_params, max_epochs, min_epochs, patience, seed, use_spec_aug=False)
            # Add a separator between runs
            print("\n" + "="*50 + "\n")
            print(f"Final val accuracy for {kernel_params[0]} seed {seed}: {avg_val_acc:.4f}")
            print("\n" + "="*50 + "\n")
        print("END RUN SEED:", seed)
        print("-" * 50)
