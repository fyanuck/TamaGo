"""Program internal format and GTP format coordinate conversion processing.
"""
from board.constant import PASS, RESIGN, OB_SIZE, GTP_X_COORDINATE


class Coordinate:
    """Coordinate transformation processing class
    """
    def __init__(self, board_size: int):
        """Initialization of coordinate transformation processing.

        Args:
            board_size (int): 碁盤の大きさ。
        """
        self.board_size = board_size
        self.board_size_with_ob = board_size + OB_SIZE * 2
        self.sgf_format = "abcdefghijklmnopqrstuvwxyz"

    def convert_from_gtp_format(self, pos: str) -> int:
        """Convert from coordinates in GTP format to coordinates in the internal representation of the program.

        Args:
            pos (str): Go Text Protocol format

        Returns:
            int: Coordinates in the internal representation of the program.
        """
        if pos.upper() == "PASS":
            return PASS

        if pos.upper() == "RESIGN":
            return RESIGN

        alphabet = pos.upper()[0]
        x_coord = 0
        for i in range(self.board_size):
            if GTP_X_COORDINATE[i + 1] is alphabet:
                x_coord = i
        y_coord = self.board_size - int(pos[1:])

        pos = x_coord + OB_SIZE + (y_coord + OB_SIZE) * self.board_size_with_ob

        return pos

    def convert_to_gtp_format(self, pos: int) -> str:
        """Convert from internal coordinates to GTP format.

        Args:
            pos (int): coordinates in the internal representation of the program.

        Returns:
            str: Go Text Protocol format
        """
        if pos == PASS:
            return "PASS"

        if pos == RESIGN:
            return "RESIGN"

        x_coord = pos % self.board_size_with_ob - OB_SIZE + 1
        y_coord = self.board_size - (pos // self.board_size_with_ob - OB_SIZE)

        return GTP_X_COORDINATE[x_coord] + str(y_coord)

    def convert_to_sgf_format(self, pos: int) -> str:
        """Convert from coordinates in the program to coordinates in SGF format.

        Args:
            pos (int): coordinates in the internal representation of the program.

        Returns:
            str: coordinate in SGF format
        """
        if pos == PASS:
            return "tt"

        if pos == RESIGN:
            return "tt"

        x_coord = pos % self.board_size_with_ob - OB_SIZE
        y_coord = pos // self.board_size_with_ob - OB_SIZE
        return self.sgf_format[x_coord] +  self.sgf_format[y_coord]
