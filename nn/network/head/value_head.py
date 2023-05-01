"""Implementation of Value head.
"""
import torch
from torch import nn


class ValueHead(nn.Module):
    """Implementation class of Value head.
    """
    def __init__(self, board_size: int, channels: int, momentum: float=0.01):
        """Initialization processing of Value head.

        Args:
            board_size (int): Go board size.
            channels (int): Number of channels in the convolutional layer of the common block part.
            momentum (float, optional): momentum parameter of the batch regularization layer. Defaults to 0.01.
        """
        super().__init__()

        self.conv_layer = nn.Conv2d(in_channels=channels, out_channels=1, \
            kernel_size=1, padding=0, bias=False)
        self.bn_layer = nn.BatchNorm2d(num_features=1, eps=2e-5, momentum=momentum)
        self.fc_layer = nn.Linear(board_size ** 2, 3)
        self.relu = nn.ReLU()
    
    def forward(self, input_plane: torch.Tensor) -> torch.Tensor:
        """Perform forward propagation processing.

        Args:
            input_plane (torch.Tensor): Input tensor to Value head.

        Returns:
            torch.Tensor: Logit output of Policy
        """
        hidden = self.relu(self.bn_layer(self.conv_layer(input_plane)))
        batch_size, _, height, width = hidden.shape
        reshape = hidden.reshape(batch_size, height * width)
        value_out = self.fc_layer(reshape)

        return value_out
