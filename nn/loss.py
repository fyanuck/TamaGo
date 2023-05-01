"""Implementation of the loss function.
"""
import torch
import torch.nn.functional as F

cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
kld_loss = torch.nn.KLDivLoss(reduction="batchmean")

def calculate_policy_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate the loss function value of _Policy.

    Args:
        output (torch.Tensor): output value of policy of neural network.
        target (torch.Tensor): Target (distribution) of the Policy.

    Returns:
        torch.Tensor: Policy loss。
    """
    return torch.sum((-target * (output.float() + 1e-8).log()), dim=1)

def calculate_sl_policy_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate the policy loss function value for supervised learning.

    Args:
        output (torch.Tensor): output value of policy of neural network.
        target (torch.Tensor): Target class of the Policy.

    Returns:
        torch.Tensor: Policy loss。
    """
    return cross_entropy_loss(output, target)

def calculate_policy_kld_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the Kullback-Leibler divegence loss function value for the Policy.

    Args:
        output (torch.Tensor): output value of policy of neural network.
        target (torch.Tensor): Target (distribution) of the Policy.

    Returns:
        torch.Tensor: Kullback-Leibler divergence loss in Policy.
    """
    return kld_loss(F.log_softmax(output, -1), target)

def calculate_value_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate the loss function value of Value.

    Args:
        output (torch.Tensor): The output value of the neural network's Value.
        target (torch.Tensor): Target class of Value.

    Returns:
        torch.Tensor: _description_
    """
    return cross_entropy_loss(output, target)
