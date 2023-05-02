"""Set various hyperparameters for training.
"""

# learning rate during supervised learning
SL_LEARNING_RATE = 0.01

# Learning rate when executing reinforcement learning
RL_LEARNING_RATE = 0.01

# mini batch size
BATCH_SIZE = 256

# learner momentum parameter
MOMENTUM=0.9

# L2 regularization weights
WEIGHT_DECAY = 1e-4

EPOCHS = 15

# Number of epochs to change learning rate and learning rate after change
LEARNING_SCHEDULE = {
    "learning_rate": {
        5: 0.001,
        8: 0.0001,
        10: 0.00001,
    }
}

# Number of data to store in one npz file
DATA_SET_SIZE = BATCH_SIZE * 4000

# Weight ratio of loss of Value to loss of Policy
SL_VALUE_WEIGHT = 0.02

# Weight ratio of loss of Value to loss of Policy
RL_VALUE_WEIGHT = 1.0

# Number of searches during self-play
SELF_PLAY_VISITS = 16

# Number of self-matching workers
NUM_SELF_PLAY_WORKERS = 4

# Number of game records to generate per learning
NUM_SELF_PLAY_GAMES = 10000
