"""Neural network input feature generation process
"""
import numpy as np

from board.constant import PASS
from board.go_board import GoBoard
from board.stone import Stone


def generate_input_planes(board: GoBoard, color: Stone, sym: int=0) -> np.ndarray:
    """Generate input data for the neural network.

    Args:
        board (GoBoard): Go board information.
        color (Stone): The color of the turn.
        sym (int, optional): symmetry specification. Defaults to 0.

    Returns:
        numpy.ndarray: Input data for the neural network.
    """
    board_data = board.get_board_data(sym)
    board_size = board.get_board_size()
    # If the turn is white, reverse the color of the stone.
    if color is Stone.WHITE:
        board_data = [datum if datum == 0 else (3 - datum) for datum in board_data]

    # The state of each intersection of the Go board
    #     Void : 1st input surface
    #     Own stone : 2nd input screen
    #     Opponent's stone : 3rd input side
    board_plane = np.identity(3)[board_data].transpose()

    # Get the previous move
    _, previous_move, _ = board.record.get(board.moves - 1)

    # Coordinates of previous move
    #     Start : 4th input screen
    #     path : 5th input screen
    if board.moves > 1 and previous_move == PASS:
        history_plane = np.zeros(shape=(1, board_size ** 2))
        pass_plane = np.ones(shape=(1, board_size ** 2))
    else:
        previous_move_data = [1 if previous_move == board.get_symmetrical_coordinate(pos, sym) \
            else 0 for pos in board.onboard_pos]
        history_plane = np.array(previous_move_data).reshape(1, board_size**2)
        pass_plane = np.zeros(shape=(1, board_size ** 2))

    # Color of turn (6th input side)
    # black is 1, white is -1
    color_plane = np.ones(shape=(1, board_size**2))
    if color == Stone.WHITE:
        color_plane = color_plane * -1

    input_data = np.concatenate([board_plane, history_plane, pass_plane, color_plane]) \
        .reshape(6, board_size, board_size).astype(np.float32) # pylint: disable=E1121

    return input_data


def generate_target_data(board:GoBoard, target_pos: int, sym: int=0) -> np.ndarray:
    """Generate target data for use in supervised learning.

    Args:
        board (GoBoard): Go board information.
        target_pos (int): Starting coordinates of training data.
        sym (int, optional): symmetry system specification. The value range is an integer from 0 to 7. Default is 0.

    Returns:
        np.ndarray: Target labels for the Policy.
    """
    target = [1 if target_pos == board.get_symmetrical_coordinate(pos, sym) else 0 \
        for pos in board.onboard_pos]
    # Insert at the end that is out of symmetry by the path.
    target.append(1 if target_pos == PASS else 0)
    #target_index = np.where(np.array(target) > 0)
    #return target_index[0]
    return np.array(target)


def generate_rl_target_data(board: GoBoard, improved_policy_data: str, sym: int=0) -> np.ndarray:
    """Select target data for Gumbel AlphaZero reinforcement learning.

    Args:
        board (GoBoard): Go board information.
        improved_policy_data (str): A string summarizing the Improved Policy data.
        sym (int, optional): symmetry system specification. The value range is an integer from 0 to 7. Default is 0.

    Returns:
        np.ndarray: Target data for the Policy.
    """
    split_data = improved_policy_data.split(" ")[1:]
    target_data = [1e-18] * len(board.board)

    for datum in split_data[1:]:
        pos, target = datum.split(":")
        coord = board.coordinate.convert_from_gtp_format(pos)
        target_data[coord] = float(target)

    target = [target_data[board.get_symmetrical_coordinate(pos, sym)] for pos in board.onboard_pos]
    target.append(target_data[PASS])

    return np.array(target)
