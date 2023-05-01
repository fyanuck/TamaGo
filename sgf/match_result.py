"""Game result constant.
"""
from enum import Enum

class MatchResult(Enum):
    """ Class that represents the result of winning or losing.
    """
    DRAW = 0
    BLACK_WIN = 1
    WHITE_WIN = 2

    @classmethod
    def get_winner_string(cls, result):
        """Gets a string representing the game result.

        Args:
            result (MatchResult): match result.

        Returns:
            str: String of game result.
        """
        if result == MatchResult.DRAW:
            return "Draw"

        if result == MatchResult.BLACK_WIN:
            return "Black"

        if result == MatchResult.WHITE_WIN:
            return "White"

        return "Undefined"
