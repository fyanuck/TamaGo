"""着手の履歴の保持。
"""
from typing import NoReturn, Tuple
import numpy as np

from board.constant import PASS, MAX_RECORDS
from board.stone import Stone
from common.print_console import print_err


class Record:
    """Class that holds the history of moves.
    """
    def __init__(self):
        """Record class constructor.
        """
        self.color = [Stone.EMPTY] * MAX_RECORDS
        self.pos = [PASS] * MAX_RECORDS
        self.hash_value = np.zeros(shape=MAX_RECORDS, dtype=np.uint64)

    def clear(self) -> NoReturn:
        """Initialize the data.
        """
        self.color = [Stone.EMPTY] * MAX_RECORDS
        self.pos = [PASS] * MAX_RECORDS
        self.hash_value.fill(0)

    def save(self, moves: int, color: Stone, pos: int, hash_value: np.array) -> NoReturn:
        """Record the history of the move.

        Args:
            moves (int): Number of moves.
            color (Stone): The color of the stone to start with.
            pos (int): starting coordinates.
            hash_value (np.array): Hash value of the aspect.
        """
        if moves < MAX_RECORDS:
            self.color[moves] = color
            self.pos[moves] = pos
            self.hash_value[moves] = hash_value
        else:
            print_err("Cannot save move record.")

    def has_same_hash(self, hash_value: np.array) -> bool:
        """Check if they have the same hash value.

        Args:
            hash_value (np.array): hash value.

        Returns:
            bool: True if they have the same hash value, False otherwise.
        """
        return np.any(self.hash_value == hash_value)

    def get(self, moves: int) -> Tuple[Stone, int, np.array]:
        """Gets the specified move.

        Args:
            moves (int): Number of moves.

        Returns:
            (Stone, int, np.array): starting color, coordinates, hash value.
        """
        return (self.color[moves], self.pos[moves], self.hash_value[moves])


def copy_record(dst: Record, src: Record) -> NoReturn:
    """Копія історії ходів

    Args:
        dst (Record): Куди записати
        src (Record): Звідки записати
    """
    dst.color = src.color[:]
    dst.pos = src.pos[:]
    dst.hash_value = src.hash_value.copy()
