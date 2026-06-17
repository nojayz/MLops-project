import pytest
import torch
from torch.utils.data import TensorDataset


@pytest.fixture
def mock_datasets(tmp_path):
    processed = tmp_path / "corruptmnist_v1" / "processed"
    processed.mkdir(parents=True)

    train_images = torch.randn(30000, 1, 28, 28)
    train_targets = torch.randint(0, 10, (30000,)).long()
    test_images = torch.randn(5000, 1, 28, 28)
    test_targets = torch.randint(0, 10, (5000,)).long()

    torch.save(train_images.squeeze(1), processed / "train_images.pt")
    torch.save(train_targets, processed / "train_targets.pt")
    torch.save(test_images.squeeze(1), processed / "test_images.pt")
    torch.save(test_targets, processed / "test_targets.pt")

    return TensorDataset(train_images, train_targets), TensorDataset(test_images, test_targets)


def test_data(mock_datasets):
    train, test = mock_datasets
    assert len(train) == 30000, f"Expected 30000 training samples, but got {len(train)}"
    assert len(test) == 5000, f"Expected 5000 test samples, but got {len(test)}"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), f"Expected image shape (1, 28, 28), but got {x.shape}"
            assert y in range(10), f"Expected target in range [0, 9], but got {y}"