import torch
from dtu_mlops.data import corrupt_mnist


def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000, f"Expected 30000 training samples, but got {len(train)}"
    assert len(test) == 5000, f"Expected 5000 test samples, but got {len(test)}"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), f"Expected image shape (1, 28, 28), but got {x.shape}"
            assert y in range(10), f"Expected target in range [0, 9], but got {y}"
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all(), f"Expected training targets to be in range [0, 9], but got {train_targets}"
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all(), f"Expected test targets to be in range [0, 9], but got {test_targets}"