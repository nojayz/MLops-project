import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from model import ConvNet  # noqa: I001
from omegaconf import DictConfig

from data import corrupt_mnist  # noqa: I001

ROOT = Path(__file__).parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(config_path="config", config_name="config")
def train(config: DictConfig) -> None:
    """Train a model on MNIST."""
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    hparams = config.train
    batch_size = hparams["batch_size"]
    lr = hparams["learning_rate"]
    epochs = hparams["epochs"]
    wandb.init(
        project="dtu-mlops", config={"batch_size": batch_size, "learning_rate": lr, "epochs": epochs}, job_type="train"
    )
    artifact = wandb.Artifact(name="mnist_dumb_model", type="model")

    model = ConvNet(
        dropout=config.model["dropout"],
        kernel_size=config.model["kernel_size"],
        padding=config.model["padding"],
    ).to(DEVICE)
    train_set, _ = corrupt_mnist(config.data["path"])

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

    print("Training complete")
    torch.save(model.state_dict(), ROOT / "models" / "model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(ROOT / "reports" / "figures" / "training_statistics.png")
    artifact.add_file(str(ROOT / "models" / "model.pth"))
    wandb.log_artifact(artifact)
    wandb.log({"training_statistics": wandb.Image(str(ROOT / "reports" / "figures" / "training_statistics.png"))})


if __name__ == "__main__":
    train()
