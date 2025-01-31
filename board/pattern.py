"""Implementation of the stone arrangement pattern
"""
from typing import Callable, List, NoReturn
import numpy as np

from board.constant import OB_SIZE
from board.stone import Stone
from common.print_console import print_err

pattern_mask = np.array([
    [0x3fff, 0x00004000, 0x00008000],
    [0xcfff, 0x00001000, 0x00002000],
    [0xf3ff, 0x00000400, 0x00000800],
    [0xfcff, 0x00000100, 0x00000200],
    [0xff3f, 0x00000040, 0x00000080],
    [0xffcf, 0x00000010, 0x00000020],
    [0xfff3, 0x00000004, 0x00000008],
    [0xfffc, 0x00000001, 0x00000002],
], dtype=np.uint32)


class Pattern:
    """Stone arrangement pattern class.
    """
    def __init__(self, board_size: int, pos_func: Callable[[int], int]):
        """Constructor for the Pattern class.

        Args:
            board_size (int): Go board size.
            POS (Callable[[int], int]): function for coordinate transformation.
        """
        self.board_size = board_size
        board_size_with_ob = board_size + OB_SIZE * 2
        self.pat3 = np.empty(shape=board_size_with_ob ** 2, dtype=np.uint32)
        self.POS = pos_func # pylint: disable=C0103
        self.update_pos = [
            -board_size_with_ob - 1, -board_size_with_ob, -board_size_with_ob + 1,
            -1, 1, board_size_with_ob - 1, board_size_with_ob, board_size_with_ob + 1
        ]

        self.nb4_empty = [0] * 65536
        for i, _ in enumerate(self.nb4_empty):
            if ((i >> 2) & 0x3) == 0:
                self.nb4_empty[i] += 1
            if ((i >> 6) & 0x3) == 0:
                self.nb4_empty[i] += 1
            if ((i >> 8) & 0x3) == 0:
                self.nb4_empty[i] += 1
            if ((i >> 12) & 0x3) == 0:
                self.nb4_empty[i] += 1

        # eye pattern
        eye_pat3 = [
            # +OO     XOO     +O+     XO+
            # O*O     O*O     O*O     O*O
            # OOO     OOO     OOO     OOO
            0x5554, 0x5556, 0x5544, 0x5546,

            # +OO     XOO     +O+     XO+
            # O*O     O*O     O*O     O*O
            # OO+     OO+     OO+     OO+
            0x1554, 0x1556, 0x1544, 0x1546,

            # +OX     XO+     +OO     OOO
            # O*O     O*O     O*O     O*O
            # OO+     +O+     ###     ###
            0x1564, 0x1146, 0xFD54, 0xFD55,

            # +O#     OO#     XOX     XOX
            # O*#     O*#     O+O     O+O
            # ###     ###     OOO     ###
            0xFF74, 0xFF75, 0x5566, 0xFD66,

            # OOX     OOO     XOO     XO#
            # O*O     O*O     O*O     O*#
            # XOO     XOX     ###     ###
            0x5965, 0x9955, 0xFD56, 0xFF76,
        ]

        self.eye = [Stone.EMPTY] * 65536

        # OOO
        # O*O
        # OOO
        self.eye[0x5555] = Stone.BLACK
        self.eye[pat3_reverse(0x5555)] = Stone.WHITE

        # +O+
        # O*O
        # +O+
        self.eye[0x1144] = Stone.BLACK
        self.eye[pat3_reverse(0x1144)] = Stone.WHITE

        for eye_pat in eye_pat3:
            sym_eye_pat = get_pat3_symmetry8(eye_pat)
            for pat3 in sym_eye_pat:
                self.eye[pat3] = Stone.BLACK
                self.eye[pat3_reverse(pat3)] = Stone.WHITE

        self.clear()

    def clear(self) -> NoReturn:
        """Initialize the surrounding stone pattern.
        """
        board_start = OB_SIZE
        board_end = self.board_size + OB_SIZE - 1
        self.pat3.fill(0)

        for y_pos in range(board_start, board_end + 1):
            self.pat3[self.POS(y_pos, board_start)] = \
                self.pat3[self.POS(y_pos, board_start)] | 0x003f
            self.pat3[self.POS(board_end, y_pos)] = \
                self.pat3[self.POS(board_end, y_pos)] | 0xc330
            self.pat3[self.POS(y_pos, board_end)] = \
                self.pat3[self.POS(y_pos, board_end)] | 0xfc00
            self.pat3[self.POS(board_start, y_pos)] = \
                self.pat3[self.POS(board_start, y_pos)] | 0x0cc3

    def remove_stone(self, pos: int) -> NoReturn:
        """Remove stones from the surrounding stone pattern.

        Args:
            pos (int): the coordinates to remove the stone from.
        """
        for i, shift in enumerate(self.update_pos):
            self.pat3[pos + shift] = self.pat3[pos + shift] & pattern_mask[i][0]

    def put_stone(self, pos: int, color: Stone) -> NoReturn:
        """Add stones with a pattern of surrounding stones.

        Args:
            pos (int): 打つ石の座標。
            color (Stone): 打つ石の色。
        """
        if color in (Stone.BLACK, Stone.WHITE):
            color_index = color.value
        else:
            return
        for i, shift in enumerate(self.update_pos):
            self.pat3[pos + shift] = self.pat3[pos + shift] | pattern_mask[i][color_index]

    def get_n_neighbors_empty(self, pos: int) -> int:
        """Gets the number of vacancies on the top, bottom, left, and right of the specified coordinates.

        Args:
            pos (int): the specified coordinates.
        Returns:
            int: number of vacancies on top, bottom, left, and right (maximum 4)
        """
        return self.nb4_empty[self.pat3[pos]]

    def get_eye_color(self, pos: int) -> Stone:
        """Gets the eye color at the specified coordinates.

        Args:
            pos (int): the specified coordinates.

        Returns:
            Stone: eye color. Stone.EMPTY if not eye.
        """
        return self.eye[self.pat3[pos]]

    def display(self, pos: int) -> NoReturn:
        """Display the stone pattern around the specified coordinates (for debugging).

        Args:
            pos (int): 表示する対象の座標。
        """
        print_err(self.pat3[pos])
        print_err(get_pat3_string(self.pat3[pos]))


def get_pat3_string(pat3: int) -> str:
    """Generates a string of 3x3 stone patterns.

    Args:
        pat3 (int): bit string of stone arrangement pattern.

    Returns:
        str: A string representing the stone arrangement pattern.
    """
    stone = ["+", "@", "O", "#"]

    message = ""
    message += stone[np.bitwise_and(pat3, 0x3)[0]]
    message += stone[np.bitwise_and(np.right_shift(pat3, 2), 0x3)[0]]
    message += stone[np.bitwise_and(np.right_shift(pat3, 4), 0x3)[0]] + '\n'
    message += stone[np.bitwise_and(np.right_shift(pat3, 6), 0x3)[0]]
    message += "*"
    message += stone[np.bitwise_and(np.right_shift(pat3, 8), 0x3)[0]] +'\n'
    message += stone[np.bitwise_and(np.right_shift(pat3, 10), 0x3)[0]]
    message += stone[np.bitwise_and(np.right_shift(pat3, 12), 0x3)[0]]
    message += stone[np.bitwise_and(np.right_shift(pat3, 14), 0x3)[0]] + '\n'

    return message


def rev(bit: int) -> int:
    """bitstring operations.

    Args:
        bit (int): Bit string.

    Returns:
        int: bit string.
    """
    return (bit >> 2) | ((bit & 0x3) << 2)


def rev3(bit: int) -> int:
    """ビット列操作。

    Args:
        bit (int): ビット列。

    Returns:
        int: ビット列。
    """
    return (bit >> 4) | (bit & 0xC) | ((bit & 0x3) << 4)

def pat3_reverse(pat3: int) -> int:
    """Generates a pattern that swaps the colors of the stones.

    Args:
        pat3 (int): 配石パターンのビット列。

    Returns:
        int: 石の色を入れ替えたパターンのビット列。
    """
    return (pat3 >> 1) & 0x5555 | ((pat3 & 0x5555) << 1)


def pat3_vertical_mirror(pat3: int) -> int:
    """Generates a symmetrical stone arrangement pattern.

    Args:
        pat3 (int): 配石パターンのビット列。

    Returns:
        int: 上下対象の配石パターンのビット列。
    """
    return ((pat3 & 0xfc00) >> 10) | (pat3 & 0x03C0) | ((pat3 & 0x003f) << 10)


def pat3_horizontal_mirror(pat3: int) -> int:
    """Generates a vertically symmetrical stone arrangement pattern.

    Args:
        pat3 (int): 配石パターンのビット列。

    Returns:
        int: 上下対象の配石パターンのビット列。
    """
    return (rev3((pat3 & 0xFC00) >> 10) << 10) \
      | (rev((pat3 & 0x03C0) >> 6) << 6) \
      | rev3((pat3 & 0x003F))


def pat3_rotate_90(pat3: int) -> int:
    """Generates a 90 degree rotated stone pattern.
    1 2 3    3 5 8
    4 * 5 -> 2 * 7
    6 7 8    1 4 6

    Args:
        pat3 (int): 配石パターンのビット列。

    Returns:
        int: 90度回転した配石パターンのビット列。
    """
    return ((pat3 & 0x0003) << 10) \
        | ((pat3 & 0x0c0c) << 4) \
        | ((pat3 & 0x3030) >> 4) \
        | ((pat3 & 0x00c0) << 6) \
        | ((pat3 & 0x0300) >> 6) \
        | ((pat3 & 0xc000) >> 10)


def get_pat3_symmetry8(pat3: int) -> List[int]:
    """Generates 8 symmetric stone placement patterns.

    Args:
        pat3 (int): 配石パターンのビット列。

    Returns:
        List[int]:  8対象の配石パターンのビット列リスト。
    """
    symmetries = [0] * 8
    symmetries[0] = pat3
    symmetries[1] = pat3_vertical_mirror(pat3)
    symmetries[2] = pat3_horizontal_mirror(pat3)
    symmetries[3] = pat3_vertical_mirror(symmetries[2])
    symmetries[4] = pat3_rotate_90(pat3)
    symmetries[5] = pat3_rotate_90(symmetries[1])
    symmetries[6] = pat3_rotate_90(symmetries[2])
    symmetries[7] = pat3_rotate_90(symmetries[3])

    return symmetries


def copy_pattern(dst: Pattern, src: Pattern) -> NoReturn:
    """opy the stone arrangement pattern data.

    Args:
        dst (Pattern): Destination stone arrangement pattern data.
        src (Pattern): Stone arrangement pattern data to copy from.
    """
    dst.pat3 = src.pat3.copy()
