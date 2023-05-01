
"""Residual Block implementation.
"""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Residual Block implementation class.
    """
    def __init__(self, channels: int, momentum: float=0.01):
        """Initialization process for each layer.

        Args:
            channels (int): Number of channels in the convolutional layer.
            momentum (float, optional): momentum parameter of the batch regularization layer. Defaults to 0.01.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, \
            kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, \
            kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=channels, eps=2e-5, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(num_features=channels, eps=2e-5, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, input_plane: torch.Tensor) -> torch.Tensor:
        """Perform forward propagation processing.

        Args:
            input_plane (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: the block's output tensor.
        """
        hidden_1 = self.relu(self.bn1(self.conv1(input_plane)))
        hidden_2 = self.bn2(self.conv2(hidden_1))

        return self.relu(input_plane + hidden_2)
