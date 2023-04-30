"""Queue for neural network computation.
"""
from typing import List, Tuple
import numpy as np


class BatchQueue:
    """A queue that holds mini-batch data.
    """
    def __init__(self):
        """BatchQueueクラスのコンストラクタ。
        """
        self.input_plane = []
        self.path = []
        self.node_index = []

    def push(self, input_plane: np.array, path: List[Tuple[int, int]], node_index: int):
        """キューにデータをプッシュする。

        Args:
            input_plane (np.array): Input data to the neural network.
            path (List[Tuple[int, int]]): The path from the root to the evaluation node.
            node_index (int): Index of the node corresponding to the aspect that the neural network evaluates.
        """
        self.input_plane.append(input_plane)
        self.path.append(path)
        self.node_index.append(node_index)

    def clear(self):
        """キューのデータを全て削除する。
        """
        self.input_plane = []
        self.path = []
        self.node_index = []
