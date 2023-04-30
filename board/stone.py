"""constants and handling definitions for intersection states (colors).
"""
from enum import Enum

class Stone(Enum):
    """Class that represents the color of the stone.
    """
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    OUT_OF_BOARD = 3

    @classmethod
    def get_opponent_color(cls, color):
        """Get the color of the opponent's turn.

        Args:
            color (Stone): The color of the turn.

        Returns:
            Stone: Opponent's color.
        """
        if color == Stone.BLACK:
            return Stone.WHITE

        if color == Stone.WHITE:
            return Stone.BLACK

        return color

    @classmethod
    def get_char(cls, color) -> str:
        """Get the character corresponding to the color.

        Args:
            color (Stone): Колір камня

        Returns:
            str: the character corresponding to the color.
        """
        if color == Stone.EMPTY:
            return '+'

        if color == Stone.BLACK:
            return '@'

        if color == Stone.WHITE:
            return 'O'

        if color == Stone.OUT_OF_BOARD:
            return '#'

        return '!'
