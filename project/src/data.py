import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from project_MLops.project.src.model import ConvNet

DATA_DIR = Path(__file__).parents[1] / "corruptmnist_v1"
NUM_TRAIN_FILES = 6


def load_data(data_dir: Path = DATA_DIR) -> tuple[TensorDataset, TensorDataset]:
    images = torch.cat([torch.load(data_dir / f"train_images_{i}.pt", weights_only=True) for i in range(NUM_TRAIN_FILES)])
    targets = torch.cat([torch.load(data_dir / f"train_target_{i}.pt", weights_only=True) for i in range(NUM_TRAIN_FILES)])
    test_images = torch.load(data_dir / "test_images.pt", weights_only=True)
    test_targets = torch.load(data_dir / "test_target.pt", weights_only=True)

    # Add channel dim: (N, 28, 28) -> (N, 1, 28, 28)
    images = images.unsqueeze(1)
    test_images = test_images.unsqueeze(1)

    return TensorDataset(images, targets), TensorDataset(test_images, test_targets)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def train(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    data_dir: Path = DATA_DIR,
    save_path: Path | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, test_ds = load_data(data_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr * 10, steps_per_epoch=len(train_loader), epochs=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * y.size(0)

        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch:>2}/{epochs}  loss={avg_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--save", type=Path, default=None, help="Path to save model weights")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
        save_path=args.save,
    )
