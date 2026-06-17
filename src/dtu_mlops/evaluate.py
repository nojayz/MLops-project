# ruff: noqa: I001
import torch
import typer
from data import corrupt_mnist
from model import ConvNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> float:
    model = ConvNet().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_dataloader:
            img, target = x.to(DEVICE), y.to(DEVICE)
            preds = model(img).argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    accuracy = correct / total
    print(f"Test accuracy: {accuracy}")
    return accuracy


if __name__ == "__main__":
    typer.run(evaluate)
