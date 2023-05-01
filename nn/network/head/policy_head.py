"""Implementation of Policy head.
"""
import torch
from torch import nn


class PolicyHead(nn.Module):
    """Policy head implementation class.
    """
    def __init__(self, board_size: int, channels: int, momentum: float=0.01):
        """Policy head initialization process.

        Args:
            board_size (int): Go board size.
            channels (int): Number of channels in the convolutional layer of the common block part.
            momentum (float, optional):  momentum parameter of the batch regularization layer. Defaults to 0.01.
        """
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels=channels, out_channels=2, \
            kernel_size=1, padding=0, bias=False)
        self.bn_layer = nn.BatchNorm2d(num_features=2, eps=2e-5, momentum=momentum)
        self.fc_layer = nn.Linear(2 * board_size ** 2, board_size ** 2 + 1)
        self.relu = nn.ReLU()

    def forward(self, input_plane: torch.Tensor) -> torch.Tensor:
        """Perform forward propagation processing.

        Args:
            input_plane (torch.Tensor): Input tensor to the Policy head.

        Returns:
            torch.Tensor: Logit output of Policy
        """
        hidden = self.relu(self.bn_layer(self.conv_layer(input_plane)))
        batch_size, channels, height, witdh = hidden.shape
        reshape = hidden.reshape(batch_size, channels * height * witdh)

        policy_out = self.fc_layer(reshape)

        return policy_out
