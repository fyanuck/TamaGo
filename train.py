"""The entry point for supervised learning.
"""
import glob
import os
import click
from learning_param import BATCH_SIZE, EPOCHS
from board.constant import BOARD_SIZE
from nn.learn import train_on_cpu, train_on_gpu, train_with_gumbel_alphazero_on_gpu, \
    train_with_gumbel_alphazero_on_cpu
from nn.data_generator import generate_supervised_learning_data, \
    generate_reinforcement_learning_data


@click.command()
@click.option('--kifu-dir', type=click.STRING, \
    help="The path of the directory where the learning data game record file is stored. If not specified, data generation will not be executed.")
@click.option('--size', type=click.IntRange(2, BOARD_SIZE), default=BOARD_SIZE, \
    help=f"Size of Go board. Minimum 2, maximum {BOARD_SIZE}")
@click.option('--use-gpu', type=click.BOOL, default=True, \
    help="Flag to use GPU during training. If not specified, GPU will be used.")
@click.option('--rl', type=click.BOOL, default=False, help="")
@click.option('--window-size', type=click.INT, default=300000, help="")
def train_main(kifu_dir: str, size: int, use_gpu: bool, rl: bool, window_size: int): # pylint: disable=C0103
    """Perform supervised learning, or reinforcement learning data generation and training.

    Args:
        kifu_dir (str): Directory path that stores game record files to learn.
        size (int): Go board size.
        use_gpu (bool): GPU usage flag.
        rl (bool): Reinforcement learning execution flag.
        window_size (int): Window size used in reinforcement learning.
    """
    program_dir = os.path.dirname(__file__)
    # Generate data if training data is specified
    if kifu_dir is not None:
        if rl:
            kifu_index_list = [int(os.path.split(dir_path)[-1]) \
                for dir_path in glob.glob(os.path.join(kifu_dir, "*"))]
            num_kifu = 0
            kifu_dir_list = []
            for index in sorted(kifu_index_list, reverse=True):
                kifu_dir_path = os.path.join(kifu_dir, str(index))
                num_kifu += len(glob.glob(kifu_dir_path))
                kifu_dir_list.append(kifu_dir_path)
                if num_kifu >= window_size:
                    break

            generate_reinforcement_learning_data(program_dir=program_dir, \
                kifu_dir_list=kifu_dir_list, board_size=size)
        else:
            generate_supervised_learning_data(program_dir=program_dir, \
                kifu_dir=kifu_dir, board_size=size)

    if rl:
        if use_gpu:
            train_with_gumbel_alphazero_on_gpu(program_dir=program_dir, \
                board_size=size, batch_size=BATCH_SIZE)
        else:
            train_with_gumbel_alphazero_on_cpu(program_dir=program_dir, \
                board_size=size, batch_size=BATCH_SIZE)
    else:
        if use_gpu:
            train_on_gpu(program_dir=program_dir,board_size=size, \
                batch_size=BATCH_SIZE, epochs=EPOCHS)
        else:
            train_on_cpu(program_dir=program_dir,board_size=size, \
                batch_size=BATCH_SIZE, epochs=EPOCHS)


if __name__ == "__main__":
    train_main() # pylint: disable=E1120
