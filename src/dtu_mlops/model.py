import torch
from torch import nn


class ConvNet(nn.Module):
    """Small CNN for CorruptMNIST (28x28 grayscale, 10 classes)."""

    def __init__(self, dropout: float = 0.25, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = ConvNet()
    x = torch.randn(16, 1, 28, 28)  # batch of 16 random images
    print(f"Output shape of model: {model(x).shape}")
