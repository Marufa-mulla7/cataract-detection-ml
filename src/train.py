"""
Training loop. Designed to run in Google Colab (GPU) — import this module there,
or run as a script with the CLI args below.

Example (in Colab, after cloning/uploading the repo):

    from src.data_loader import get_dataloaders, compute_class_weights
    from src.models import build_model
    from src.train import train_model

    train_loader, val_loader, test_loader = get_dataloaders("data/cataract")
    class_weights = compute_class_weights("data/cataract")
    model = build_model("efficientnet_b0", num_classes=4)
    train_model(model, train_loader, val_loader, class_weights, epochs=20,
                save_path="models/efficientnet_b0_best.pt")
"""

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, val_loader, class_weights=None,
                 epochs=20, lr=1e-3, save_path="models/best_model.pt",
                 patience=5, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved new best model ({best_val_acc:.4f}) to {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
                break

    return history
