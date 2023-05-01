"""Sequential Halving
"""
from typing import Dict, Tuple
import math


def get_sequence_of_considered_visits(max_num_considered_actions: int, \
    num_simulations: int) -> Tuple[int]:
    """Gets the search count threshold column corresponding to the search count.

    Args:
        max_num_considered_actions (int): Maximum search width.
        num_simulations (int): Number of searches to perform in one thought.

    Returns:
        Tuple[int]: column of search count threshold.
    """
    if max_num_considered_actions <= 1:
        return tuple(range(num_simulations))
    log2max = int(math.ceil(math.log2(max_num_considered_actions)))
    sequence = []
    visits = [0] * max_num_considered_actions
    num_considered = max_num_considered_actions

    while len(sequence) < num_simulations:
        num_extra_visits = max(1, int(num_simulations / (log2max * num_considered)))
        for _ in range(num_extra_visits):
            sequence.extend(visits[:num_considered])
            for i in range(num_considered):
                visits[i] += 1
        num_considered = max(2, num_considered // 2)

    return tuple(sequence[:num_simulations])


def get_candidates_and_visit_pairs(max_num_considered_actions: int, \
    num_simulations: int) -> Dict[int, int]:
    """Get a pair of search width and search count.

    Args:
        max_num_considered_actions (int): Maximum search width.
        num_simulations (int): Number of searches to perform in one thought.

    Returns:
        Dict[int, int]: Dictionary with search width as key and search count as value.
    """
    visit_dict = {}
    visit_list = get_sequence_of_considered_visits(max_num_considered_actions, num_simulations)
    max_count = max(visit_list)
    count_list = [0] * (max_count + 1)
    for visit in visit_list:
        count_list[visit] += 1

    for count in count_list:
        if count in visit_dict:
            visit_dict[count] += 1
        else:
            visit_dict[count] = 1

    return visit_dict
