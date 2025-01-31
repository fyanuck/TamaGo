"""Implementation of a self-matching execution worker.
"""
import os
import random
from typing import List
import numpy as np

from board.constant import PASS, RESIGN
from board.go_board import GoBoard, copy_board
from board.stone import Stone

from sgf.selfplay_record import SelfPlayRecord
from mcts.tree import MCTSTree
from mcts.time_manager import TimeManager, TimeControl
from nn.utility import load_network
from learning_param import SELF_PLAY_VISITS
import torch

# pylint: disable=R0913,R0914
def selfplay_worker(save_dir: str, model_file_path: str, index_list: List[int], \
    size: int, visits: int, use_gpu: bool):
    """Self match execution worker.

    Args:
        save_dir (str): Directory path to save game record files.
        model_file_path (str): Neural network model file path to use.
        index_list (List[int]): Index list to use when saving the game record file.
        size (int): Go board size.
        visits (int): Number of visits during self-play.
        use_gpu (bool): GPU usage flag.
    """
    board = GoBoard(board_size=size, komi=7.0, check_superko=True)
    init_board = GoBoard(board_size=size, komi=7.0, check_superko=True)
    record = SelfPlayRecord(save_dir, board.coordinate)
    network = load_network(model_file_path=model_file_path, use_gpu=use_gpu)
    network.training = False

    print(f'Torch version: {torch.__version__}')
    print(f'Is CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA device count: {torch.cuda.device_count()}')

    np.random.seed(random.choice(index_list))

    mcts = MCTSTree(network, tree_size=SELF_PLAY_VISITS * 10)
    time_manager = TimeManager(TimeControl.CONSTANT_PLAYOUT, constant_visits=visits)

    max_moves = (board.get_board_size() ** 2) * 2

    for i, index in enumerate(index_list):
        print(f'Doing {i+1}/{len(index_list)}...')
        if os.path.isfile(os.path.join(save_dir, f"{index}.sgf")):
            continue
        copy_board(board, init_board)
        color = Stone.BLACK
        record.clear()
        pass_count = 0
        never_resign = True if random.randint(1, 10) == 1 else False # pylint: disable=R1719
        is_resign = False
        score = 0.0
        for _ in range(max_moves):
            pos = mcts.generate_move_with_sequential_halving(board=board, color=color, \
                time_manager=time_manager, never_resign=never_resign)

            if pos == RESIGN:
                winner = Stone.get_opponent_color(color)
                is_resign = True
                break

            board.put_stone(pos, color)

            if pos == PASS:
                pass_count += 1
            else:
                pass_count = 0

            record.save_record(mcts.get_root(), pos, color)

            color = Stone.get_opponent_color(color)

            if pass_count == 2:
                winner = Stone.EMPTY
                break

        if pass_count == 2:
            score = board.count_score() - board.get_komi()
            if score > 0.1:
                winner = Stone.BLACK
            elif score < -0.1:
                winner = Stone.WHITE
            else:
                winner = Stone.OUT_OF_BOARD

        record.set_index(index)
        record.write_record(winner, board.get_komi(), is_resign, score)
