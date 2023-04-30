"""Go board constants
"""
# Go board size
BOARD_SIZE = 9
# size outside the board
OB_SIZE = 1
# maximum number of runs
STRING_MAX = int(0.8 * BOARD_SIZE * (BOARD_SIZE - 1) + 5)
# maximum number of adjacent runs
NEIGHBOR_MAX = STRING_MAX
# Maximum number of stones that make up a run
STRING_POS_MAX = (BOARD_SIZE + OB_SIZE * 2) ** 2
# maximum number of breathing points
STRING_LIB_MAX = (BOARD_SIZE + OB_SIZE * 2) ** 2
# Stone Coordinate limit
STRING_END = STRING_POS_MAX - 1
# Liberty point limit
LIBERTY_END = STRING_LIB_MAX - 1
# Adjacent enemy limits=
NEIGHBOR_END = NEIGHBOR_MAX - 1

# 着手に関する定数
# パスに対応する座標
PASS = 0
# 投了に対応する座標
RESIGN = -1
# Go Text Protocol X-coordinate character
GTP_X_COORDINATE = 'IABCDEFGHJKLMNOPQRSTUVWXYZ'

# maximum number of moves history
MAX_RECORDS = (BOARD_SIZE ** 2) * 3
