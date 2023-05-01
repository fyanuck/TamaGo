
"""Dual Network implementation.
"""
from typing import Tuple
from torch import nn
import torch

from board.constant import BOARD_SIZE
from nn.network.res_block import ResidualBlock
from nn.network.head.policy_head import PolicyHead
from nn.network.head.value_head import ValueHead


class DualNet(nn.Module): # pylint: disable=R0902
    """Dual Network implementation class.
    """
    def __init__(self, device: torch.device, board_size: int=BOARD_SIZE):
        """Dual Network initialization process

        Args:
            device (torch.device): Inference execution device. Only used during inference in exploration, not during training.
            board_size (int, optional): Go board size. Default value is BOARD_SIZE.
        """
        super().__init__()
        filters = 64
        blocks = 6

        self.device = device

        self.conv_layer = nn.Conv2d(in_channels=6, out_channels=filters, \
            kernel_size=3, padding=1, bias=False)
        self.bn_layer = nn.BatchNorm2d(num_features=filters)
        self.relu = nn.ReLU()
        self.blocks = make_common_blocks(blocks, filters)
        self.policy_head = PolicyHead(board_size, filters)
        self.value_head = ValueHead(board_size, filters)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward propagation processing.

        Args:
            input_plane (torch.Tensor): Input feature tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: logit of Policy and Value.
        """
        blocks_out = self.blocks(self.relu(self.bn_layer(self.conv_layer(input_plane))))

        return self.policy_head(blocks_out), self.value_head(blocks_out)


    def forward_for_sl(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs forward propagation, used in supervised learning.

        Args:
            input_plane (torch.Tensor): Input feature tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:  Policy through Softmax, logit of Value
        """
        policy, value = self.forward(input_plane)
        return self.softmax(policy), value


    def forward_with_softmax(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward propagation processing.

        Args:
            input_plane (torch.Tensor): Input feature tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy, Value inference result.
        """
        policy, value = self.forward(input_plane)
        return self.softmax(policy), self.softmax(value)


    def inference(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs forward propagation processing. 
        Data transfer between devices is also internally processed because it is a method used for searching.

        Args:
            input_plane (torch.Tensor): Input feature tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy, Value inference result.
        """
        policy, value = self.forward(input_plane.to(self.device))
        return self.softmax(policy).cpu(), self.softmax(value).cpu()


    def inference_with_policy_logits(self, input_plane: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward propagation. Because of the search method used for Gumbel AlphaZero,
        Inter-device data transfer is also handled internally.

        Args:
            input_plane (torch.Tensor): Input feature tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy, Value inference result.
        """
        policy, value = self.forward(input_plane.to(self.device))
        return policy.cpu(), self.softmax(value).cpu()


def make_common_blocks(num_blocks: int, num_filters: int) -> torch.nn.Sequential:
    """Construct and return the common residual block of the DualNet.

    Args:
        num_blocks (int): Number of residual blocks to stack.
        num_filters (int): Number of convolutional layer filters in the residual block.

    Returns:
        torch.nn.Sequential: residual block sequence.
    """
    blocks = [ResidualBlock(num_filters) for _ in range(num_blocks)]
    return nn.Sequential(*blocks)
