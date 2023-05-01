"""Node implementation used in Monte Carlo tree search.
"""
from typing import Dict, NoReturn

import numpy as np
from board.constant import BOARD_SIZE
from board.go_board import GoBoard
from common.print_console import print_err
from mcts.constant import NOT_EXPANDED, C_VISIT, C_SCALE
from mcts.pucb.pucb import calculate_pucb_value
from nn.utility import apply_softmax


MAX_ACTIONS = BOARD_SIZE ** 2 + 1
PUCT_WEIGHT = 1.0

class MCTSNode: # pylint: disable=R0902, R0904
    """Class of node information used in Monte Carlo tree search.
    """
    def __init__(self, num_actions: int=MAX_ACTIONS):
        """_MCTSNodeクラスのコンストラクタ

        Args:
            num_actions (int, optional): Maximum number of candidate moves. Defaults to MAX_ACTIONS.
        """
        self.node_visits = 0
        self.virtual_loss = 0
        self.node_value_sum = 0.0
        self.action = [0] * num_actions
        self.children_index = np.zeros(num_actions, dtype=np.int32)
        self.children_value = np.zeros(num_actions, dtype=np.float64)
        self.children_visits = np.zeros(num_actions, dtype=np.int32)
        self.children_policy = np.zeros(num_actions, dtype=np.float64)
        self.children_virtual_loss = np.zeros(num_actions, dtype=np.int32)
        self.children_value_sum = np.zeros(num_actions, dtype=np.float64)
        self.noise = np.zeros(num_actions, dtype=np.float64)
        self.num_children = 0

    def expand(self, policy: Dict[int, float]) -> NoReturn:
        """Expand the node and initialize it.

        Args:
            policy (Dict[int, float]): A map of policies corresponding to candidate moves.
        """
        self.node_visits = 0
        self.node_value_sum = 0.0
        self.virtual_loss = 0
        self.action = [0] * MAX_ACTIONS
        self.children_index.fill(NOT_EXPANDED)
        self.children_value.fill(0.0)
        self.children_visits.fill(0)
        self.children_virtual_loss.fill(0)
        self.children_value_sum.fill(0.0)
        self.noise.fill(0.0)

        self.set_policy(policy)


    def set_policy(self, policy_map: Dict[int, float]) -> NoReturn:
        """Set the coordinates of the start candidate and the value of Policy.

        Args:
            policy_map (Dict[int, float]): Key is start coordinates, Value is start policy.
        """
        index = 0
        for pos, policy in policy_map.items():
            self.action[index] = pos
            self.children_policy[index] = policy
            index += 1
        self.num_children = index


    def add_virtual_loss(self, index) -> NoReturn:
        """Add Virtual Loss.

        Args:
            index (_type_): The index of the child node to add.
        """
        self.virtual_loss += 1
        self.children_virtual_loss[index] += 1


    def update_policy(self, policy: Dict[int, float]) -> NoReturn:
        """Policyを更新する。

        Args:
            policy (Dict[int, float]): A map of candidate moves and corresponding policies.
        """
        for i in range(self.num_children):
            self.children_policy[i] = policy[self.action[i]]


    def set_leaf_value(self, index: int, value: float) -> NoReturn:
        """Sets the value at the end.

        Args:
            index (int): The index of the child node to set the Value for.
            value (float): The value of Value to set.

        Returns:
            NoReturn: _description_
        """
        self.children_value[index] = value


    def update_child_value(self, index: int, value: float) -> NoReturn:
        """Adds Value to child node and restores Virtual Loss.

        Args:
            index (int): The index of the child node to update.
            value (float): Value of Value to add.
        """
        self.children_value_sum[index] += value
        self.children_visits[index] += 1
        self.children_virtual_loss[index] -= 1


    def update_node_value(self, value: float) -> NoReturn:
        """Add Value to node and restore Virtual Loss.

        Args:
            value (float): Value of Value to add.
        """
        self.node_value_sum += value
        self.node_visits += 1
        self.virtual_loss -= 1


    def select_next_action(self) -> int:
        """Select next move based on PUCB value.

        Returns:
            int: The index of the child node to choose as the next move.
        """
        pucb_values = calculate_pucb_value(self.node_visits + self.virtual_loss, \
            self.children_visits + self.children_virtual_loss, \
            self.children_value_sum, self.children_policy + self.noise)

        return np.argmax(pucb_values[:self.num_children])


    def get_num_children(self) -> int:
        """Get the number of child nodes.

        Returns:
            int: Number of child nodes.
        """
        return self.num_children


    def get_best_move_index(self) -> int:
        """Get the index of the child node with the maximum number of searches.

        Returns:
            int: Index of the child node with the maximum number of searches.
        """
        return np.argmax(self.children_visits[:self.num_children])


    def get_best_move(self) -> int:
        """Acquire the move with the maximum number of searches.

        Returns:
            int: Start coordinate with maximum number of searches.
        """
        return self.action[self.get_best_move_index()]


    def get_child_move(self, index: int) -> int:
        """Gets the start coordinates corresponding to the specified child node.

        Args:
            index (int): The index of the specified child node.

        Returns:
            int: starting coordinates.
        """
        return self.action[index]


    def get_child_index(self, index: int) -> int:
        """Gets the transition destination index of the specified child node.

        Args:
            index (int): The index of the specified child node.

        Returns:
            int: Index to transition to.
        """
        return self.children_index[index]


    def set_child_index(self, index: int, child_index: int) -> NoReturn:
        """Set the transition destination index of the specified child node.

        Args:
            index (int): Index of the specified child node.
            child_index (int): Index of the node to transition to.
        """
        self.children_index[index] = child_index


    def print_search_result(self, board: GoBoard) -> NoReturn:
        """Displays the search result. Displays the number of searches for the searched hand and the average value of Value.

        Args:
            board (GoBoard): Current position information.
        """
        value = np.divide(self.children_value_sum, self.children_visits, \
            out=np.zeros_like(self.children_value_sum), where=(self.children_visits != 0))
        for i in range(self.num_children):
            if self.children_visits[i] > 0:
                pos = board.coordinate.convert_to_gtp_format(self.action[i])
                print_err(f"pos={pos}, visits={self.children_visits[i]}, value={value[i]:.4f}")


    def set_gumbel_noise(self) -> NoReturn:
        """Sets the Gumbel noise.
        """
        self.noise = np.random.gumbel(loc=0.0, scale=1.0, size=self.noise.size)


    def calculate_completed_q_value(self) -> np.array:
        """Calculate the Completed-Q value.

        Returns:
            np.array: Completed-Q value.
        """
        policy = apply_softmax(self.children_policy[:self.num_children])

        q_value = np.divide(self.children_value_sum, self.children_visits, \
            out=np.zeros_like(self.children_value_sum), \
            where=(self.children_visits > 0))[:self.num_children]

        sum_prob = np.sum(policy)
        v_pi = np.sum(policy * q_value)

        return np.where(self.children_visits[:self.num_children] > 0, q_value, v_pi / sum_prob)


    def calculate_improved_policy(self) -> np.array:
        """Compute Improved Policy.

        Returns:
            np.array: Improved Policy.
        """
        max_visit = np.max(self.children_visits)

        sigma_base = (C_VISIT + max_visit) * C_SCALE
        completed_q_value = self.calculate_completed_q_value()

        improved_logits = self.children_policy[:self.num_children] + sigma_base * completed_q_value

        return apply_softmax(improved_logits)


    def select_move_by_sequential_halving_for_root(self, count_threshold: int) -> int:
        """Uses Gumbel AlphaZero's search method to select the next move.Used on Root only.

        Args:
            count_threshold (int): Search count threshold.

        Returns:
            int: Index of the selected child node.
        """
        max_count = max(self.children_visits[:self.num_children])

        sigma_base = (C_VISIT + max_count) * C_SCALE

        counts = self.children_visits[:self.num_children] \
            + self.children_virtual_loss[:self.num_children]
        q_mean = np.divide(self.children_value_sum, self.children_visits, \
            out=np.zeros_like(self.children_value_sum), \
            where=(self.children_visits > 0))[:self.num_children]

        evaluation_value = np.where(counts >= count_threshold, -10000.0, \
            self.children_policy[:self.num_children] + self.noise[:self.num_children] \
            + sigma_base * q_mean)
        return np.argmax(evaluation_value)


    def select_move_by_sequential_halving_for_node(self) -> int:
        """Use Gumbel AlphaZero's search method to select the next move. Used outside the root.

        Returns:
            int: Index of the selected child node.
        """

        improved_policy = self.calculate_improved_policy()

        evaluation_value = improved_policy \
            - (self.children_visits[:self.num_children] / (1.0 + self.node_visits))

        return np.argmax(evaluation_value)


    def calculate_value_evaluation(self, index: int) -> float:
        """Calculates the value of the specified child node.

        Args:
            index (int): index of the child node.

        Returns:
            float: Value of the specified child node.
        """
        if self.children_visits[index] == 0:
            return 0.5
        return self.children_value_sum[index] / self.children_visits[index]
