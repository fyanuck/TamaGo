"""program information.
"""
PROGRAM_NAME="TamaGo"
PROTOCOL_VERSION="2"

# Version 0.0.0 : Random player implementation.
# Version 0.1.0 : Implementation of reading process of SGF file. Supported load_sgf command.
#                 Added data structure for stone arrangement pattern. Improved eye detection, upper/lower/left/right vacancy detection, etc.
#                 Implementation of move history, Zobrist Hash, and judgment of transcendence.
# Version 0.2.0 : Implementation of supervised learning for neural networks.
#                 Implementation of move generation logic using Policy Network.
# Version 0.2.1 : Fixed structure of Residual Block. Rerun training.
# Version 0.3.0 : Implementation of Monte Carlo tree search.
# Version 0.3.1 : Bug fix for value update processing of Monte Carlo tree search. Support for komi and get_komi commands.
# Version 0.4.0 : Implementation of Sequential Halving Applied to Trees (SHOT).
# Version 0.5.0 : Support for search time control, time_left and time_settings commands.
# Version 0.6.0 : Implementation of Gumbel AlphaZero method of reinforcement learning. Improved network structure.
# Version 0.6.1 : Added --batch-size option.
VERSION="0.6.1"
