"""he process that controls the search time.
"""
from enum import Enum
from typing import NoReturn

from board.stone import Stone
from mcts.constant import CONST_VISITS, CONST_TIME, REMAINING_TIME, VISITS_PER_SEC


class TimeControl(Enum):
    """A class that represents the thinking time management mode.
    """
    CONSTANT_PLAYOUT = 0
    CONSTANT_TIME = 1
    TIME_CONTROL = 2


class TimeManager:
    """Time management class.
    """
    def __init__(self, mode: TimeControl, constant_visits: int=CONST_VISITS, constant_time: \
        float=CONST_TIME, remaining_time: float=REMAINING_TIME):
        """Constructor for the TimeManager class.

        Args:
            mode (TimeControl): search time management mode
            constant_visits (int, optional): Number of visits per move. The default value is CONST_VISITS.
            constant_time (float, optional): Exploration time per move. The default value is CONST_TIME.
            remaining_time (float, optional): Remaining time. Default value REMAINING_TIME.
        """
        self.mode = mode
        self.constant_visits = constant_visits
        self.constant_time = constant_time
        self.default_time = remaining_time
        self.search_speed = 200
        self.remaining_time = [remaining_time] * 2


    def initialize(self):
        """Initialize time limit.
        """
        self.remaining_time = [self.default_time] * 2


    def set_search_speed(self, visits: int, time: float) -> NoReturn:
        """Sets the search speed.

        Args:
            visits (int): Number of visits performed.
            time (float): Time spent searching in seconds.
        """
        self.search_speed = visits / time if visits > 0 else VISITS_PER_SEC


    def get_num_visits_threshold(self, color: Stone) -> int:
        """_summary_

        Args:
            color (Stone): _description_

        Returns:
            int: _description_
        """
        if self.mode == TimeControl.CONSTANT_PLAYOUT:
            return int(self.constant_visits)
        if self.mode == TimeControl.CONSTANT_TIME:
            return int(self.search_speed * self.constant_time)
        if self.mode == TimeControl.TIME_CONTROL:
            remaining_time = self.remaining_time[0] \
                if color is Stone.BLACK else self.remaining_time[1]
            return int(self.search_speed * remaining_time / 8.0)
        return int(self.constant_visits)


    def set_remaining_time(self, color: Stone, time: float) -> NoReturn:
        """Sets the remaining time.

        Args:
            color (Stone): The color of the turn that sets the remaining time.
            time (float): Remaining time to set.
        """
        if color is Stone.BLACK:
            self.remaining_time[0] = time
        if color is Stone.WHITE:
            self.remaining_time[1] = time


    def set_mode(self, mode:TimeControl) -> NoReturn:
        """Change the settings for thinking time management.

        Args:
            mode (TimeControl): Specified thinking mode
        """
        self.mode = mode
