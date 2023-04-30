"""Parameter setting for search
"""

# index of unexpanded child node
NOT_EXPANDED = -1

# PUCB second term weight parameter
PUCB_SECOND_TERM_WEIGHT = 1.0

# number of searches per move
PLAYOUTS = 100

# Mini-batch size when searching
NN_BATCH_SIZE = 1

# Parameter for Gumbel AlphaZero (C_visit)
C_VISIT = 50

# Parameter for Gumbel AlphaZero (C_scale)
C_SCALE = 1.0

# Maximum number of moves to consider in Sequential Halving
MAX_CONSIDERED_NODES = 16

# default number of searches per move
CONST_VISITS = 1000

# Default value for search time per move
CONST_TIME = 5.0

# default value of duration
REMAINING_TIME = 60.0

# default search speed
VISITS_PER_SEC = 200

# threshold for conceding
RESIGN_THRESHOLD = 0.05
