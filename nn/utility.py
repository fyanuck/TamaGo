"""Deep learning utilities.
"""
from typing import NoReturn, Dict, List, Tuple
import time
import torch
import numpy as np

from common.print_console import print_err
from nn.network.dual_net import DualNet


def get_torch_device(use_gpu: bool) -> torch.device:
    """to get torch.device.

    Args:
        use_gpu (bool): GPU usage flag.

    Returns:
        torch.device: device information.
    """
    if use_gpu:
        torch.cuda.set_device(0)
        return torch.device("cuda")
    return torch.device("cpu")


def _calculate_losses(loss: Dict[str, float], iteration: int) \
    -> Tuple[float, float, float]:
    """Calculate various loss function values.

    Args:
        loss (Dict[str, float]): Loss function value information.
        iteration (int): number of iterations.

    Returns:
        Tuple[float, float, float]: Total loss, Policy loss, Value loss。
    """
    return loss["loss"] / iteration, loss["policy"] / iteration, \
        loss["value"] / iteration



def print_learning_process(loss_data: Dict[str, float], epoch: int, index: int, \
    iteration: int, start_time: float) -> NoReturn:
    """Display learning progress information.

    Args:
        loss_data (Dict[str]): Information of loss function value.
        epoch (int): Number of training epochs.
        index (int): dataset index.
        iteration (int): number of training iterations for the batch size.
        start_time (float): Learning start time.
    """
    loss, policy_loss, value_loss = _calculate_losses(loss_data, iteration)
    training_time = time.time() - start_time

    print_err(f"epoch {epoch}, data-{index} : loss = {loss}, time = {training_time} sec.")
    print_err(f"\tpolicy loss : {policy_loss}")
    print_err(f"\tvalue loss  : {value_loss}")


def print_evaluation_information(loss_data: Dict[str, float], epoch: int, \
    iteration: int, start_time: float) -> NoReturn:
    """Display evaluation information for test data.

    Args:
        loss_data (Dict[str, float]): Loss function value information.
        epoch (int): Number of training epochs.
        iteration (int): Number of test iterations.
        start_time (float): Evaluation start time.
    """
    loss, policy_loss, value_loss = _calculate_losses(loss_data, iteration)
    testing_time = time.time() - start_time

    print_err(f"Test {epoch} : loss = {loss}, time = {testing_time} sec.")
    print_err(f"\tpolicy loss : {policy_loss}")
    print_err(f"\tvalue loss  : {value_loss}")


def save_model(network: torch.nn.Module, path: str) -> NoReturn:
    """Saves neural network parameters.

    Args:
        network (torch.nnModel): Neural network model.
        path (str): Parameter file path.
    """
    torch.save(network.to("cpu").state_dict(), path)


def load_data_set(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the training data set.

    Args:
        path (str): file path of the dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Input data, Policy, Value.
    """
    data = np.load(path)
    perm = np.random.permutation(len(data["value"]))
    return data["input"][perm], data["policy"][perm].astype(np.float32), \
        data["value"][perm].astype(np.int64)


def split_train_test_set(file_list: List[str], train_data_ratio: float) \
    -> Tuple[List[str], List[str]]:
    """Separate the data file used for training and the data file used for validation.

    Args:
        file_list (List[str]): npz file list to use for training.
        train_data_ratio (float): Ratio of data to use for training.

    Returns:
        Tuple[List[str], List[str]]: training and validation datasets.
    """
    train_data_set = file_list[:int(len(file_list) * train_data_ratio)]
    test_data_set = file_list[int(len(file_list) * train_data_ratio):]

    print(f"Training data set : {train_data_set}")
    print(f"Testing data set  : {test_data_set}")

    return train_data_set, test_data_set


def apply_softmax(logits: np.array) -> np.array:
    """Apply the Softmax function.

    Args:
        logits (np.array): Input values ​​for the Softmax function.

    Returns:
        np.array: Values ​​after applying the Softmax function.
    """
    shift_exp = np.exp(logits - np.max(logits))

    return shift_exp / np.sum(shift_exp)


def load_network(model_file_path: str, use_gpu: bool) -> DualNet:
    """Load and get the neural network.

    Args:
        model_file_path (str): Neural network parameter file path.
        use_gpu (bool): GPU usage flag.

    Returns:
        DualNet: パラメータロード済みのニューラルネットワーク。
    """
    device = get_torch_device(use_gpu=use_gpu)
    network = DualNet(device)
    print(f'Setting network to device: {device}')
    network.to(device)
    try:
        network.load_state_dict(torch.load(model_file_path))
    except: # pylint: disable=W0702
        print(f"Failed to load {model_file_path}.")
    network.eval()
    torch.set_grad_enabled(False)

    return network
