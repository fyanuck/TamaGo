"""Start generation process using only Policy Network
"""
import random

import torch

from board.constant import PASS
from board.go_board import GoBoard
from board.stone import Stone
from nn.feature import generate_input_planes
from nn.network.dual_net import DualNet

def generate_move_from_policy(network: DualNet, board: GoBoard, color: Stone) -> int:
    """Generate move using Policy Network.

    Args:
        network (DualNet): Neural network.
        board (GoBoard): Current Go board information.
        color (Stone): The color of the turn.

    Returns:
        int: Generated move coordinates.
    """
    board_size = board.get_board_size()
    input_plane = generate_input_planes(board, color)
    input_data = torch.tensor(input_plane.reshape(1, 6, board_size, board_size)) #pylint: disable=E1121
    policy, _ = network.inference(input_data)

    policy = policy[0].numpy().tolist()

    # Pick up only legal hands as candidates
    candidates = [{"pos": pos, "policy": policy[i]} \
        for i, pos in enumerate(board.onboard_pos) if board.is_legal(pos, color)]

    # The pass is confirmed as a candidate
    candidates.append({ "pos": PASS, "policy": policy[board_size ** 2] })

    max_policy = max([candidate["policy"] for candidate in candidates])

    sampled_candidates = [candidate for candidate in candidates \
        if candidate["policy"] > max_policy * 0.1]

    sampled_pos = [candidate["pos"] for candidate in sampled_candidates]
    sampled_policy = [candidate["policy"] for candidate in sampled_candidates]

    return random.choices(sampled_pos, weights=sampled_policy, k=1)[0]
