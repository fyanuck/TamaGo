"""GTP client entry point.
"""
import os
import click

from gtp.client import GtpClient
from board.constant import BOARD_SIZE
from mcts.constant import NN_BATCH_SIZE
from mcts.time_manager import TimeControl

default_model_path = os.path.join("model", "model.bin")

@click.command()
@click.option('--size', type=click.IntRange(2, BOARD_SIZE), default=BOARD_SIZE, \
    help=f"Specify the board size. Default is {BOARD_SIZE}.")
@click.option('--superko', type=click.BOOL, default=False, help="superko enable flag. Default is False.")
@click.option('--model', type=click.STRING, default=default_model_path, \
    help=f"Specify the model path of the neural network to use. Specify the path relative to the home directory of the program. \
    Default is {default_model_path}.")
@click.option('--use-gpu', type=click.BOOL, default=False, \
    help="Flag to use GPU for neural network computation. Default is False.")
@click.option('--policy-move', type=click.BOOL, default=False, \
    help="Start generation process flag according to Policy distribution. Default is False.")
@click.option('--sequential-halving', type=click.BOOL, default=False, \
    help="Flag to generate start with Gumbel AlphaZero search method. Default is False.")
@click.option('--komi', type=click.FLOAT, default=7.0, \
    help="Set Komi value. Default is 7.0.")
@click.option('--visits', type=click.IntRange(min=1), default=1000, \
    help="Specify the number of searches per move. Default is 1000. \
    Ignored when --const-time option or --time option is specified.")
@click.option('--const-time', type=click.FLOAT, \
    help="Specify search time per move. Ignore when --time option is specified.")
@click.option('--time', type=click.FLOAT, \
    help="Specify time limit.")
@click.option('--batch-size', type=click.IntRange(min=1), default=NN_BATCH_SIZE, \
    help="Mini-batch size when searching. Default is NN_BATCH_SIZE.")
def gtp_main(size: int, superko: bool, model:str, use_gpu: bool, sequential_halving: bool, \
    policy_move: bool, komi: float, visits: int, const_time: float, time: float, \
    batch_size: int): # pylint: disable=R0913
    """Starting the GTP client.

    Args:
        size (int): Go board size.
        superko (bool): superko enable flag.
        model (str): The path of the model file relative to the program's home directory.
        use_gpu (bool):  GPU usage flag for neural network. Default is False.
        policy_move (bool): Move generation process flag according to policy distribution. Default is False.
        sequential_halving (bool): Flag to generate halving in Gumbel AlphaZero's halving method. Default is False.
        komi (float): Komi value. Default is 7.0.
        visits (int): Number of visits per move. Default is 1000.
        const_time (float): Exploration time per move.
        time (float): The duration of the game.
        batch_size (int): Neural network mini-batch size when performing search. Default is NN_BATCH_SIZE.
    """
    mode = TimeControl.CONSTANT_PLAYOUT

    if const_time is not None:
        mode = TimeControl.CONSTANT_TIME
    if time is not None:
        mode = TimeControl.TIME_CONTROL

    program_dir = os.path.dirname(__file__)
    client = GtpClient(size, superko, os.path.join(program_dir, model), use_gpu, policy_move, \
        sequential_halving, komi, mode, visits, const_time, time, batch_size)
    client.run()


if __name__ == "__main__":
    gtp_main() # pylint: disable=E1120
