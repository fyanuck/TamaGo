"""Implementation of command processing for GoGui.
"""
import math

import torch

from board.go_board import GoBoard
from board.stone import Stone
from nn.feature import generate_input_planes
from nn.network.dual_net import DualNet

class GoguiAnalyzeCommand: # pylint: disable=R0903
    """Base information class for Gogui analysis commands.
    """
    def __init__(self, command_type, label, command):
        """constructor.

        Args:
            command_type (_type_): _description_
            label (_type_): _description_
            command (_type_): _description_
        """
        self.type = command_type
        self.label = label
        self.command = command

    def get_command_information(self) -> str:
        """Get command information, what gogui-analyze_command displays.

        Returns:
            str: Command information string.
        """
        return self.type + "/" + self.label + "/" + self.command


def display_policy_distribution(model: DualNet, board: GoBoard, color: Stone) -> str: # pylint: disable=R0914
    """ Generate a string to colorize and display Policy.(GoGui analysis command)
    Args:
        model (DualNet): Neural network that outputs Policy.
        board (GoBoard): Position information to evaluate.
        color (Stone): The color of the turn to evaluate.

    Returns:
        str: Display string.
    """
    board_size = board.get_board_size()
    input_plane = generate_input_planes(board, color)
    input_plane = torch.tensor(input_plane.reshape(1, 6, board_size, board_size)) #pylint: disable=E1121
    policy, _ = model.forward_with_softmax(input_plane)

    max_policy, min_policy = 0, 1
    log_policies = [math.log(policy[0][i]) for i in range(board_size * board_size)]

    for i, log_policy in enumerate(log_policies):
        pos = board.onboard_pos[i]
        if board.board[pos] is Stone.EMPTY and board.is_legal(pos, color):
            max_policy = max(max_policy, log_policy)
            min_policy = min(min_policy, log_policy)

    scale = max_policy - min_policy
    response = ""

    for i, log_policy in enumerate(log_policies):
        pos = board.onboard_pos[i]
        if board.board[pos] is Stone.EMPTY and board.is_legal(pos, color):
            color_value = int((log_policy - min_policy) / scale * 255)
            response += f"\"#{color_value:02x}{0:02x}{255-color_value:02x}\" "
        else:
            response += "\"\" "
        if (i + 1) % board_size == 0:
            response += "\n"

    return response


def display_policy_score(model: DualNet, board: GoBoard, color: Stone) -> str:
    """Generate a string to display Policy numerically. (GoGui analysis command)

    Args:
        model (DualNet): Neural network that outputs Policy.
        board (GoBoard): Position information to evaluate.
        color (Stone): The color of the turn to evaluate.

    Returns:
        str: Display string.
    """
    board_size = board.get_board_size()
    input_plane = generate_input_planes(board, color)
    input_plane = torch.tensor(input_plane.reshape(1, 6, board_size, board_size)) #pylint: disable=E1121
    policy_predict, _ = model.forward_with_softmax(input_plane)
    policies = [policy_predict[0][i] for i in range(board_size ** 2)]
    response = ""

    for i, policy in enumerate(policies):
        pos = board.onboard_pos[i]
        if board.is_legal(pos, color):
            response += f"\"{policy:.04f}\" "
        else:
            response += "\"\" "
        if (i + 1) % board_size == 0:
            response += "\n"

    return response
