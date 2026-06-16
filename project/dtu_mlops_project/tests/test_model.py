import torch
from dtu_mlops.model import ConvNet



def test_my_model():
    """Test the MyModel class."""
    # Add your test cases here
    model = ConvNet()
    x = torch.randn(16, 1, 28, 28)  
    output = model(x)
    assert output.shape == (16, 10), f"Expected output shape (16, 10), but got {output.shape}"
    