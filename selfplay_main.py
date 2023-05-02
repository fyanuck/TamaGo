"""The entry point for self-matching.
"""
import glob
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
import click
from board.constant import BOARD_SIZE
from selfplay.worker import selfplay_worker
from learning_param import SELF_PLAY_VISITS, NUM_SELF_PLAY_WORKERS, \
    NUM_SELF_PLAY_GAMES

# pylint: disable=R0913, R0914
@click.command()
@click.option('--save-dir', type=click.STRING, default="archive", \
    help="Directory to save game record files. Default is archive.")
@click.option('--process', type=click.IntRange(min=1), default=NUM_SELF_PLAY_WORKERS, \
    help=f"Number of self-playing workers. Default is {NUM_SELF_PLAY_WORKERS}.")
@click.option('--num-data', type=click.IntRange(min=1), default=NUM_SELF_PLAY_GAMES, \
    help="Number of data (game records) to generate. Default is 10000.")
@click.option('--size', type=click.IntRange(2, BOARD_SIZE), default=BOARD_SIZE, \
    help=f"Size of board. Default is {BOARD_SIZE}.")
@click.option('--use-gpu', type=click.BOOL, default=True, \
    help="GPU usage flag. Default is True.")
@click.option('--visits', type=click.IntRange(min=2), default=SELF_PLAY_VISITS, \
    help=f"Number of searches during self-match. Default is {SELF_PLAY_VISITS}.")
@click.option('--model', type=click.STRING, default=os.path.join("model", "rl-model.bin"), \
    help="Neural network model file path. Default is rl-model.bin in the model directory.")
def selfplay_main(save_dir: str, process: int, num_data: int, size: int, \
    use_gpu: bool, visits: int, model: str):
    """Perform a self-match.

    Args:
        save_dir (str): Directory to save game record files. Default is archive.
        process (int): Number of self-playing processes to run. Default is 4.
        num_data (int): Number of data to generate. Default is 10000.
        size (int): Go board size. Default is BOARD_SIZE.
        use_gpu (bool): GPU usage flag. Default is true
        visits (int): Number of visits during self-play. Default is SELF_PLAY_VISITS.
        model (str): the path of the model file to use. Default is model/model.bin.
    """
    file_index_list = list(range(1, num_data + 1))
    split_size = math.ceil(num_data / process)
    file_indice = [file_index_list[i:i+split_size] \
        for i in range(0, len(file_index_list), split_size)]
    kifu_dir_index_list = [int(os.path.split(dir_path)[-1]) \
        for dir_path in glob.glob(os.path.join(save_dir, "*"))]
    kifu_dir_index_list.append(0)
    kifu_dir_index = max(kifu_dir_index_list) + 1

    start_time = time.time()
    os.mkdir(os.path.join(save_dir, str(kifu_dir_index)))

    print(f"Self play visits : {visits}")

    with ProcessPoolExecutor(max_workers=process) as executor:
        futures = [executor.submit(selfplay_worker, os.path.join(save_dir, str(kifu_dir_index)), \
            model, file_list, size, visits, use_gpu) for file_list in file_indice]
        for future in futures:
            future.result()

    finish_time = time.time() - start_time

    print(f"{finish_time}sec, {3600.0 * num_data / finish_time} games/hour")


if __name__ == "__main__":
    selfplay_main() # pylint: disable=E1120
