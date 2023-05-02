"""SGF format file reading process.
"""
from typing import NoReturn
from board.coordinate import Coordinate
from board.constant import PASS, OB_SIZE
from board.stone import Stone
from common.print_console import print_err
from sgf.match_result import MatchResult

sgf_coord_map = {
    "a" : 1,
    "b" : 2,
    "c" : 3,
    "d" : 4,
    "e" : 5,
    "f" : 6,
    "g" : 7,
    "h" : 8,
    "i" : 9,
    "j" : 10,
    "k" : 11,
    "l" : 12,
    "m" : 13,
    "n" : 14,
    "o" : 15,
    "p" : 16,
    "q" : 17,
    "r" : 18,
    "s" : 19,
}


class SGFReader: # pylint: disable=R0902
    """Read SGF file.
    """
    def __init__(self, filename: str, board_size: int): # pylint: disable=R0912
        """constructor

        Args:
            filename (str): SGF file path
            board_size (int): Go board size
        """
        self.board_size = board_size
        self.board_size_with_ob = board_size + OB_SIZE * 2
        self.move = [0] * board_size * board_size * 3
        self.komi = 7.0
        self.result = MatchResult.DRAW
        self.comment = [""] * board_size * board_size * 3
        self.moves = 0
        self.size = board_size
        self.event = None
        self.black_player_name = None
        self.white_player_name = None
        self.application = None
        self.copyright = None

        with open(filename, mode='r', encoding='utf-8') as sgf_file:
            sgf_text = sgf_file.read()

        sgf_text = sgf_text.replace('\n', '')

        cursor, last = 0, len(sgf_text)

        while cursor < last:
            while cursor < last and _is_ignored_char(sgf_text[cursor]):
                cursor += 1
            if cursor == last:
                return

            if sgf_text[cursor:cursor+3] == "SZ[":
                cursor = self._get_size(sgf_text, cursor)
            elif sgf_text[cursor:cursor+3] == "RE[":
                cursor = self._get_result(sgf_text, cursor)
            elif sgf_text[cursor:cursor+3] == "KM[":
                cursor = self._get_komi(sgf_text, cursor)
            elif sgf_text[cursor:cursor+2] == "B[":
                cursor = self._get_move(sgf_text, cursor, Stone.BLACK)
            elif sgf_text[cursor:cursor+2] == "W[":
                cursor = self._get_move(sgf_text, cursor, Stone.WHITE)
            elif sgf_text[cursor:cursor+2] == "C[":
                cursor = self._get_comment(sgf_text, cursor)
            elif sgf_text[cursor:cursor+3] == "EV[":
                cursor = self._get_event(sgf_text, cursor)
            elif sgf_text[cursor:cursor+3] == "PB[":
                cursor = self._get_player_name(sgf_text, cursor, Stone.BLACK)
            elif sgf_text[cursor:cursor+3] == "PW[":
                cursor = self._get_player_name(sgf_text, cursor, Stone.WHITE)
            elif sgf_text[cursor:cursor+3] == "AP[":
                cursor = self._get_application(sgf_text, cursor)
            elif sgf_text[cursor:cursor+3] == "CP[":
                cursor = self._get_copyright(sgf_text, cursor)
            elif _is_ignored_tag(sgf_text, cursor):
                cursor = _skip_data(sgf_text, cursor)
            else:
                cursor += 1

    def _get_size(self, sgf_text: str, cursor: int) -> int:
        """Reads the board size from the SZ tag.

        Args:
            sgf_text (str): SGF text.
            cursor (int): Current cursor position.

        Returns:
            int: Position of cursor to look next.
        """
        tmp_cursor = 3

        while sgf_text[cursor+tmp_cursor] != ']':
            tmp_cursor += 1

        self.size = int(sgf_text[cursor+3:cursor+tmp_cursor])
        self.board_size = self.size
        self.board_size_with_ob = self.size + OB_SIZE * 2

        return cursor + tmp_cursor

    def _get_komi(self, sgf_text: str, cursor: int) -> int:
        """Read the Komi value from the KM tag.

        Args:
            sgf_text (str): SGF text.
            cursor (int): Current cursor position.

        Returns:
            int: Position of cursor to look next.
        """
        tmp_cursor = 3

        while sgf_text[cursor+tmp_cursor] != ']':
            tmp_cursor += 1

        self.komi = float(sgf_text[cursor+3:cursor+tmp_cursor])

        return cursor + tmp_cursor

    def _get_comment(self, sgf_text: str, cursor: int) -> int:
        """Reads comments from C tags.

        Args:
            sgf_text (str): SGF text.
            cursor (int): Current cursor position.

        Returns:
            int: Position of cursor to look next.
        """
        tmp_cursor = 2

        while sgf_text[cursor+tmp_cursor] != ']':
            tmp_cursor += 1

        self.comment[self.moves - 1] = sgf_text[cursor+2:cursor+tmp_cursor]

        return cursor + tmp_cursor

    def _get_event(self, sgf_text: str, cursor: int) -> int:
        """Read event name from EV tag.

        Args:
            sgf_text (str): SGF text.
            cursor (int): Current cursor position.

        Returns:
            int: Position of cursor to look next.
        """
        tmp_cursor = 3

        while sgf_text[cursor+tmp_cursor] != ']':
            tmp_cursor += 1

        self.event = sgf_text[cursor+3:cursor+tmp_cursor]

        return cursor + tmp_cursor

    def _get_player_name(self, sgf_text: str, cursor: int, color: Stone) -> int:
        """Read player name from PB tag and PW tag.

        Args:
            sgf_text (str): SGF text.
            cursor (int): Current cursor position.
            color (Stone): The color of the turn.

        Returns:
            int: Position of cursor to look next.
        """
        tmp_cursor = 3

        while sgf_text[cursor+tmp_cursor] != ']':
            tmp_cursor += 1

        if color is Stone.BLACK:
            self.black_player_name = sgf_text[cursor+3:cursor+tmp_cursor]
        elif color is Stone.WHITE:
            self.white_player_name = sgf_text[cursor+3:cursor+tmp_cursor]

        return cursor + tmp_cursor

    def _get_application(self, sgf_text: str, cursor: int) -> int:
        """Reads the application name from the AP tag.

        Args:
            sgf_text (str): SGF text.
            cursor (int): Current cursor position.

        Returns:
            int: Position of cursor to look next.
        """
        tmp_cursor = 3

        while sgf_text[cursor+tmp_cursor] != ']':
            tmp_cursor += 1

        self.application = sgf_text[cursor+3:cursor+tmp_cursor]

        return cursor + tmp_cursor

    def _get_copyright(self, sgf_text: str, cursor: int) -> int:
        """Reads the copyright from the CP tag.

        Args:
            sgf_text (str): SGF text.
            cursor (int): Current cursor position.

        Returns:
            int: Position of cursor to look next.
        """
        tmp_cursor = 3

        while sgf_text[cursor+tmp_cursor] != ']':
            tmp_cursor += 1

        self.copyright = sgf_text[cursor+3:cursor+tmp_cursor]

        return cursor + tmp_cursor

    def _get_result(self, sgf_text: str, cursor: int) -> int:
        """Read game result from RE tag.

        Args:
            sgf_text (str): SGF text.
            cursor (int): Current cursor position.

        Returns:
            int: Position of cursor to look next.
        """
        tmp_cursor = 3

        while sgf_text[cursor+tmp_cursor] != ']':
            tmp_cursor += 1

        result = sgf_text[cursor+3].upper()

        if result == 'B':
            self.result = MatchResult.BLACK_WIN
        elif result == 'W':
            self.result = MatchResult.WHITE_WIN
        else:
            self.result = MatchResult.DRAW

        return cursor + tmp_cursor

    def _get_move(self, sgf_text: str, cursor: int, color: Stone) -> int:
        """Read the start from B tag and W tag.

        Args:
            sgf_text (str): SGF text.
            cursor (int): Current cursor position.
            color (Stone): The starting color.

        Returns:
            int: Position of cursor to look next.
        """
        tmp_cursor = 0
        if sgf_text[cursor+2] == "]":
            x_coord, y_coord = 0, 0
            tmp_cursor = 2
        else:
            x_coord = _parse_coordinate(sgf_text[cursor+2])
            y_coord = _parse_coordinate(sgf_text[cursor+3])
            while sgf_text[cursor+tmp_cursor] != ']':
                tmp_cursor += 1
        self.move[self.moves] = (x_coord, y_coord, color)
        self.moves += 1

        return cursor + tmp_cursor

    def get_moves(self) -> int:
        """Get started one by one from the beginning.

        Yields:
            int: starting coordinates.
        """
        for i in range(self.moves):
            yield self.get_move_data(i)

    def get_n_moves(self) -> int:
        """Gets the number of moves in the game record.

        Returns:
            int: number of moves
        """
        return self.moves

    def get_move_data(self, index: int) -> int:
        """Get the coordinates of the move for the specified number of moves.

        Args:
            index (int): the move you want to get the coords for

        Returns:
            int: starting coordinates.
        """
        if index >= self.moves:
            print_err("overrun move")
            return PASS

        x_coord, y_coord, _ = self.move[index]
        if x_coord == 0 and y_coord == 0:
            return PASS

        return x_coord + (OB_SIZE - 1) + (y_coord + (OB_SIZE - 1)) * self.board_size_with_ob

    def get_color(self, index: int) -> Stone:
        """Get the turn of the specified number of moves.

        Args:
            index (int): the move you want to get the color for

        Returns:
            Stone: The color of the turn.
        """
        if index >= self.moves:
            print_err("overrun color")
            return Stone.EMPTY

        _, _, color = self.move[index]

        return color

    def get_value_label(self) -> int:
        """Get the learning label for Value.

        Returns:
            int: Value label. Black wins with 2, White wins with 0, and Go with 1.
        """
        if self.result is MatchResult.BLACK_WIN:
            return 2
        if self.result is MatchResult.WHITE_WIN:
            return 0
        if self.result is MatchResult.DRAW:
            return 1
        print_err(f"Invalid value label {self.result}")
        return 1

    def get_comment(self, index: int) -> str:
        """Gets the comment for the number of moves specified.

        Args:
            index (int): number of moves.

        Returns:
            str: Comment for the specified number of moves.
        """
        return self.comment[index]

    def display(self) -> NoReturn:
        """Display the information of the read SGF file.(for debugging)
        """
        message = ""
        message += f"Board size   : {self.size}\n"
        message += f"Komi         : {self.komi}\n"
        message += f"Winner       : {MatchResult.get_winner_string(self.result)}\n"
        if self.event is not None:
            message += "Event        : " + self.event + "\n"
        if self.black_player_name is not None:
            message += "Black player : " + self.black_player_name + "\n"
        if self.white_player_name is not None:
            message += "White player : " + self.white_player_name + "\n"
        if self.application is not None:
            message += "Application  : " + self.application + "\n"

        coordinate = Coordinate(self.board_size)
        for index in range(self.moves):
            pos = self.get_move_data(index)
            _, _, color = self.move[index]
            move_data = (index + 1, coordinate.convert_to_gtp_format(pos), color)
            message += f"\tMove {move_data[0]} : {move_data[1]} ({move_data[2]})\n"

        print_err(message)


def _is_ignored_char(char: str) -> bool:
    """Determines whether the character is ignored by SGF.

    Args:
        char (str): The character to be judged.

    Returns:
        bool: True if the character should be ignored, False otherwise.
    """
    return  (char in '') or (char in '\t') or (char in '\n') or \
            (char in '\r') or (char in ';') or (char in '(') or (char in ')')

def _is_ignored_tag(text: str, cursor: int) -> bool:
    return text[cursor:cursor+3] in ["GM[", "HA[", "AB[", "PL[", "RU[", \
        "CP[", "GM[", "FF[", "DT[", "PC[", "CA[", "TM[", "OT[", "TB[", \
        "TW[", "BR[", "WR["]

def _parse_coordinate(char: str) -> int:
    """Convert coordinates in SGF format to coordinates inside the program.

    Args:
        char (str): coordinates in SGF format.

    Returns:
        int: Coordinates inside the program.
    """
    if char in sgf_coord_map:
        return sgf_coord_map[char]

    return 0

def _skip_data(sgf_text: str, cursor: int) -> int:
    """Skip ignore tags.

    Args:
        sgf_text (str): SGF text.
        cursor (int): Current cursor position.

    Returns:
        int: Position of cursor to look next.
    """
    tmp_cursor = 2
    while sgf_text[cursor + tmp_cursor] != ']':
        tmp_cursor += 1

    return cursor + tmp_cursor
