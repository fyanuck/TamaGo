"""Monte Carlo tree search implementation.
"""
from typing import Dict, List, NoReturn, Tuple
import copy
import time
import numpy as np
import torch

from board.constant import PASS, RESIGN
from board.go_board import GoBoard, copy_board
from board.stone import Stone
from common.print_console import print_err
from nn.feature import generate_input_planes
from nn.network.dual_net import DualNet
from mcts.batch_data import BatchQueue
from mcts.constant import NOT_EXPANDED, PLAYOUTS, NN_BATCH_SIZE, \
    MAX_CONSIDERED_NODES, RESIGN_THRESHOLD
from mcts.sequential_halving import get_candidates_and_visit_pairs
from mcts.node import MCTSNode
from mcts.time_manager import TimeManager


class MCTSTree:
    """Monte Carlo tree search implementation class.
    """
    def __init__(self, network: DualNet, tree_size=65536, batch_size=NN_BATCH_SIZE):
        """MCTSTreeクラスのコンストラクタ。

        Args:
            network (DualNet): Neural network to use.
            tree_size (int, optional): Maximum number of nodes that make up the tree. Default is 65536.
            batch_size (int, optional): Mini-batch size for forward propagation of neural network. Default is NN_BATCH_SIZE.
        """
        self.node = [MCTSNode() for i in range(tree_size)]
        self.num_nodes = 0
        self.root = 0
        self.network = network
        self.batch_queue = BatchQueue()
        self.current_root = 0
        self.batch_size = batch_size


    def search_best_move(self, board: GoBoard, color: Stone, time_manager: TimeManager) -> int:
        """Performs a Monte Carlo tree search and returns the best move.

        Args:
            board (GoBoard): Position information to evaluate.
            color (Stone): The color of the turn of the position to be evaluated.
            time_manager (TimeManager):

        Returns:
            int: starting coordinates.
        """
        self.num_nodes = 0

        start_time = time.time()

        self.current_root = self.expand_node(board, color)
        input_plane = generate_input_planes(board, color, 0)
        self.batch_queue.push(input_plane, [], self.current_root)

        self.process_mini_batch(board)

        root = self.node[self.current_root]

        # Return PASS if there is only one candidate move
        if root.get_num_children() == 1:
            return PASS

        # run a search
        self.search(board, color, time_manager.get_num_visits_threshold(color))

        # get the best move
        next_move = root.get_best_move()
        next_index = root.get_best_move_index()

        # display search results and time taken
        root.print_search_result(board)
        search_time = time.time() - start_time
        po_per_sec = root.node_visits / search_time

        time_manager.set_search_speed(root.node_visits, search_time)

        print_err(f"{search_time:.2f} seconds, {po_per_sec:.2f}")

        value = root.calculate_value_evaluation(next_index)

        if value < RESIGN_THRESHOLD:
            return RESIGN

        return next_move


    def search(self, board: GoBoard, color: Stone, threshold: int) -> NoReturn:
        """Performs a specified number of searches.

        Args:
            board (GoBoard): Current position information.
            color (Stone): The color of the turn in the current position.
            threshold (int): The number of searches to perform in this search.
        """
        search_board = copy.deepcopy(board)
        for _ in range(threshold):
            copy_board(dst=search_board,src=board)
            start_color = color
            self.search_mcts(search_board, start_color, self.current_root, [])


    def search_mcts(self, board: GoBoard, color: Stone, current_index: int, \
        path: List[Tuple[int, int]]) -> NoReturn:
        """Perform a Monte Carlo tree search.

        Args:
            board (GoBoard): Current position information.
            color (Stone): The color of the turn in the current position.
            current_index (int): Index of the node to evaluate.
            path (List[Tuple[int, int]]): The path from the root to the node corresponding to current_index.
        """

        # Find the hand with the maximum UCB value
        next_index = self.node[current_index].select_next_action()
        next_move = self.node[current_index].get_child_move(next_index)

        path.append((current_index, next_index))

        # advance one step
        board.put_stone(pos=next_move, color=color)
        color = Stone.get_opponent_color(color)

        # add virtual loss
        self.node[current_index].add_virtual_loss(next_index)

        if self.node[current_index].children_visits[next_index] < 1:
            # Neural network computation

            input_plane = generate_input_planes(board, color, 0)
            next_node_index = self.node[current_index].get_child_index(next_index)
            self.batch_queue.push(input_plane, path, next_node_index)

            if len(self.batch_queue.node_index) >= self.batch_size:
                self.process_mini_batch(board)
        else:
            if self.node[current_index].get_child_index(next_index) == NOT_EXPANDED:
                child_index = self.expand_node(board, color)
                self.node[current_index].set_child_index(next_index, child_index)
            next_node_index = self.node[current_index].get_child_index(next_index)
            self.search_mcts(board, color, next_node_index, path)


    def expand_node(self, board: GoBoard, color: Stone) -> NoReturn:
        """Expand the node.

        Args:
            board (GoBoard): Current position information.
            color (Stone): The color of the current turn.
        """
        node_index = self.num_nodes

        candidates = board.get_all_legal_pos(color)
        candidates = [candidate for candidate in candidates \
            if (board.check_self_atari_stone(candidate, color) < 7) \
                and not board.is_complete_eye(candidate, color)]
        candidates.append(PASS)

        policy = get_tentative_policy(candidates)
        self.node[node_index].expand(policy)

        self.num_nodes += 1
        return node_index


    def process_mini_batch(self, board: GoBoard, use_logit: bool=False): # pylint: disable=R0914
        """Mini-batch process the input of the neural network and reflect the calculation result in the search result.

        Args:
            board (GoBoard): Go board information.
            use_logit (bool): Flag to logit output of policy
        """

        input_planes = torch.Tensor(np.array(self.batch_queue.input_plane))

        if use_logit:
            raw_policy, value_data = self.network.inference_with_policy_logits(input_planes)
        else:
            raw_policy, value_data = self.network.inference(input_planes)

        policy_data = []
        for policy in raw_policy:
            policy_dict = {}
            for i, pos in enumerate(board.onboard_pos):
                policy_dict[pos] = policy[i]
            policy_dict[PASS] = policy[board.get_board_size() ** 2] - 0.5
            policy_data.append(policy_dict)

        for policy, value_dist, path, node_index in zip(policy_data, \
            value_data, self.batch_queue.path, self.batch_queue.node_index):

            self.node[node_index].update_policy(policy)

            if path:
                value = value_dist[0] + value_dist[1] * 0.5

                reverse_path = list(reversed(path))
                leaf = reverse_path[0]

                self.node[leaf[0]].set_leaf_value(leaf[1], value)

                for index, child_index in reverse_path:
                    self.node[index].update_child_value(child_index, value)
                    self.node[index].update_node_value(value)
                    value = 1.0 - value

        self.batch_queue.clear()


    def generate_move_with_sequential_halving(self, board: GoBoard, color: Stone, \
        time_manager: TimeManager, never_resign: bool) -> int:
        """_summary_

        Args:
            board (GoBoard): _description_
            color (Stone): _description_
            time (TimeManager):

        Returns:
            int: _description_
        """
        self.num_nodes = 0
        start_time = time.time()
        self.current_root = self.expand_node(board, color)
        input_plane = generate_input_planes(board, color)
        self.batch_queue.push(input_plane, [], self.current_root)
        self.process_mini_batch(board, use_logit=True)
        self.node[self.current_root].set_gumbel_noise()

        # run search
        self.search_by_sequential_halving(board, color, \
            time_manager.get_num_visits_threshold(color))

        # get the best hand
        root = self.node[self.current_root]
        next_index = root.select_move_by_sequential_halving_for_root(PLAYOUTS)

        #root.print_search_result(board)

        # Decide whether to concede based on winning percentage
        value = root.calculate_value_evaluation(next_index)

        search_time = time.time() - start_time

        time_manager.set_search_speed(self.node[self.current_root].node_visits, search_time)

        #po_per_sec = self.node[self.current_root].node_visits / search_time
        #print_err(f"{search_time:.2f} seconds, {po_per_sec:.2f}")

        if not never_resign and value < 0.05:
            return RESIGN

        return root.get_child_move(next_index)


    def search_by_sequential_halving(self, board: GoBoard, color: Stone, \
        threshold: int) -> NoReturn:
        """Performs Sequential Halving searches for the specified number of searches.

        Args:
            board (GoBoard): The position you want to evaluate.
            color (Stone): The color of the turn of the position you want to evaluate.
            threshold (int): Number of searches to perform.
        """
        search_board = copy.deepcopy(board)

        num_root_children = self.node[self.current_root].get_num_children()
        base_num_considered = num_root_children \
            if num_root_children < MAX_CONSIDERED_NODES else MAX_CONSIDERED_NODES
        search_control_dict = get_candidates_and_visit_pairs(base_num_considered, threshold)

        for num_considered, max_count in search_control_dict.items():
            for count_threshold in range(max_count):
                for _ in range(num_considered):
                    copy_board(search_board, board)
                    start_color = color

                    # Explore
                    self.search_sequential_halving(search_board, start_color, \
                        self.current_root, [], count_threshold + 1)
            self.process_mini_batch(search_board, use_logit=True)


    def search_sequential_halving(self, board: GoBoard, color: Stone, current_index: int, \
        path: List[Tuple[int, int]], count_threshold: int) -> NoReturn: # pylint: disable=R0913
        """Perform a Sequential Halving search.

        Args:
            board (GoBoard): Current position.
            color (Stone): The color of the current turn.
            current_index (int): Index of the current node.
            path (List[Tuple[int, int]]): The index to traverse to the current node.
            count_threshold (int): Threshold for the number of searches to be evaluated.
        """
        current_node = self.node[current_index]
        if current_index == self.current_root:
            next_index = current_node.select_move_by_sequential_halving_for_root(count_threshold)
        else:
            next_index = current_node.select_move_by_sequential_halving_for_node()
        next_move = self.node[current_index].get_child_move(next_index)

        path.append((current_index, next_index))

        board.put_stone(pos=next_move, color=color)
        color = Stone.get_opponent_color(color)

        self.node[current_index].add_virtual_loss(next_index)

        if self.node[current_index].children_visits[next_index] < 1:
            # ニューラルネットワークの計算
            input_plane = generate_input_planes(board, color)
            next_node_index = self.node[current_index].get_child_index(next_index)
            self.batch_queue.push(input_plane, path, next_node_index)
        else:
            if self.node[current_index].get_child_index(next_index) == NOT_EXPANDED:
                child_index = self.expand_node(board, color)
                self.node[current_index].set_child_index(next_index, child_index)
            next_node_index = self.node[current_index].get_child_index(next_index)
            self.search_sequential_halving(board, color, next_node_index, path, count_threshold)

    def get_root(self) -> MCTSNode:
        """Returns the root of the tree.

        Returns:
            MCTSNode: The root of the tree to use in Monte Carlo tree search.
        """
        return self.node[self.current_root]

def get_tentative_policy(candidates: List[int]) -> Dict[int, float]:
    """Gets the policy to use until the neural network is calculated.

    Args:
        candidates (List[int]): List of candidate hands containing paths.

    Returns:
        Dict[int, float]: Map of candidate hand coordinates and Policy values.
    """
    score = np.random.dirichlet(alpha=np.ones(len(candidates)))
    return dict(zip(candidates, score))
