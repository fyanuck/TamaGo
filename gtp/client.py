"""Імплементація клієнта Go Text Protocol
"""
import os
import random
import sys
from typing import List, NoReturn

from program import PROGRAM_NAME, VERSION, PROTOCOL_VERSION
from board.constant import PASS, RESIGN
from board.coordinate import Coordinate
from board.go_board import GoBoard
from board.stone import Stone
from common.print_console import print_err
from gtp.gogui import GoguiAnalyzeCommand, display_policy_distribution, \
    display_policy_score
from mcts.time_manager import TimeControl, TimeManager
from mcts.tree import MCTSTree
from nn.policy_player import generate_move_from_policy
from nn.utility import load_network
from sgf.reader import SGFReader



class GtpClient: # pylint: disable=R0902,R0903
    """_Go Text Protocol client implementation class
    """
    # pylint: disable=R0913
    def __init__(self, board_size: int, superko: bool, model_file_path: str, \
        use_gpu: bool, policy_move: bool, use_sequential_halving: bool, \
        komi: float, mode: TimeControl, visits: int, const_time: float, \
        time: float, batch_size: int): # pylint: disable=R0913
        """Initialize the Go Text Protocol client.

        Args:
            board_size (int): Go board size.
            superko (bool): Enable superko detection.
            model_file_path (str): Network parameter file path.
            use_gpu (bool): GPU usage flag.
            policy_move (bool): Flag to move according to policy distribution.
            use_sequential_halving (bool): Flag to generate start with Gumbel AlphaZero's halving method.
            komi (float): Komi value.
            mode (TimeControl): Think time control mode.
            visits (int): Number of visits per move.
            const_time (float): Exploration time per move.
            time (float): time limit.
            batch_size (int): Neural network mini-batch size when searching.
        """
        self.gtp_commands = [
            "version",
            "protocol_version",
            "name",
            "quit",
            "known_command",
            "list_commands",
            "play",
            "genmove",
            "clear_board",
            "boardsize",
            "time_left",
            "time_settings",
            "get_komi",
            "komi",
            "showboard",
            "load_sgf",
            "gogui-analyze_commands"
        ]
        self.superko = superko
        self.board = GoBoard(board_size=board_size, komi=komi, check_superko=superko)
        self.coordinate = Coordinate(board_size=board_size)
        self.gogui_analyze_command = [
            GoguiAnalyzeCommand("cboard", "Display policy distribution (Black)", \
                "display_policy_black_color"),
            GoguiAnalyzeCommand("cboard", "Display policy distribution (White)", \
                "display_policy_white_color"),
            GoguiAnalyzeCommand("sboard", "Display policy score (Black)", \
                "display_policy_black"),
            GoguiAnalyzeCommand("sboard", "Display policy score (White)", \
                "display_policy_white"),
        ]
        self.policy_move = policy_move
        self.use_sequential_halving = use_sequential_halving
        self.use_network = False

        if mode is TimeControl.CONSTANT_PLAYOUT:
            self.time_manager = TimeManager(mode=mode, constant_visits=visits)
        if mode is TimeControl.CONSTANT_TIME:
            self.time_manager = TimeManager(mode=mode, constant_time=const_time)
        if mode is TimeControl.TIME_CONTROL:
            self.time_manager = TimeManager(mode=mode, remaining_time=time)

        try:
            self.network = load_network(model_file_path, use_gpu)
            self.use_network = True
            self.mcts = MCTSTree(network=self.network, batch_size=batch_size)
        except FileNotFoundError:
            print_err(f"Model file {model_file_path} is not found")
        except RuntimeError:
            print_err(f"Failed to load {model_file_path}")


    def _known_command(self, command: str) -> NoReturn:
        """Process known_command commands.
        Show 'true' for supported commands, and 'unknown command' for unsupported commands

        Args:
            command (str): GTP command for which you want to check compatibility.
        """
        if command in self.gtp_commands:
            respond_success("true")
        else:
            respond_failure("unknown command")

    def _list_commands(self) -> NoReturn:
        """Process the list_commands command.
        Show all supported commands.
        """
        response = ""
        for command in self.gtp_commands:
            response += '\n' + command
        respond_success(response)

    def _komi(self, s_komi: str) -> NoReturn:
        """Handle the komi command.
        Set the input Komi.

        Args:
            s_komi (str): Komi to set.
        """
        komi = float(s_komi)
        self.board.set_komi(komi)
        respond_success("")

    def _play(self, color: str, pos: str) -> NoReturn:
        """Handle the play command.
        Place a stone of the specified color at the entered coordinates.

        Args:
            color (str): The turn color.
            pos (str): starting coordinates.
        """
        if color.lower()[0] == 'b':
            play_color = Stone.BLACK
        elif color.lower()[0] == 'w':
            play_color = Stone.WHITE
        else:
            respond_failure("play color pos")
            return

        coord = self.coordinate.convert_from_gtp_format(pos)

        if coord != PASS and not self.board.is_legal(coord, play_color):
            print(f"illigal {color} {pos}")

        if pos.upper != "RESIGN":
            self.board.put_stone(coord, play_color)

        respond_success("")

    def _genmove(self, color: str) -> NoReturn:
        """Handle the genmove command.
        Think in the entered turn and generate a move.

        Args:
            color (str): The turn color.
        """
        if color.lower()[0] == 'b':
            genmove_color = Stone.BLACK
        elif color.lower()[0] == 'w':
            genmove_color = Stone.WHITE
        else:
            respond_failure("genmove color")
            return

        if self.use_network:
            if self.policy_move:
                # Start generation from Policy Network
                pos = generate_move_from_policy(self.network, self.board, genmove_color)
                _, previous_move, _ = self.board.record.get(self.board.moves - 1)
                if self.board.moves > 1 and previous_move == PASS:
                    pos = PASS
            else:
                # Start generation by Monte Carlo tree search
                if self.use_sequential_halving:
                    pos = self.mcts.generate_move_with_sequential_halving(self.board, \
                        genmove_color, self.time_manager, False)
                else:
                    pos = self.mcts.search_best_move(self.board, genmove_color, self.time_manager)
        else:
            # Random start generation
            legal_pos = [pos for pos in self.board.onboard_pos \
                if self.board.is_legal_not_eye(pos, genmove_color)]
            if legal_pos:
                pos = random.choice(legal_pos)
            else:
                pos = PASS

        if pos != RESIGN:
            self.board.put_stone(pos, genmove_color)

        respond_success(self.coordinate.convert_to_gtp_format(pos))

    def _boardsize(self, size: str) -> NoReturn:
        """Handle the boardsize command.
        Set to Go board of specified size.

        Args:
            size (str): The board size to set.
        """
        board_size = int(size)
        self.board = GoBoard(board_size=board_size, check_superko=self.superko)
        self.coordinate = Coordinate(board_size=board_size)
        self.time_manager.initialize()
        respond_success("")

    def _clear_board(self) -> NoReturn:
        """Process the clear_board command.
        Initialize the board.
        """
        self.board.clear()
        self.time_manager.initialize()
        respond_success("")

    def _time_settings(self, arg_list: List[str]) -> NoReturn:
        """Process the time_settings command.
        Set only the holding time.

        Args:
            arg_list (List[str]): Argument list of the command (duration, countdown, countdown count).
        """
        time = float(arg_list[0])
        self.time_manager.set_remaining_time(Stone.BLACK, time)
        self.time_manager.set_remaining_time(Stone.WHITE, time)
        respond_success("")

    def _time_left(self, arg_list: List[str]) -> NoReturn:
        """Process the time_left command.
        Sets the remaining time for the specified turn.

        Args:
            arg_list (List[str]): Command argument list (hand color, remaining time).
        """
        if arg_list[0][0] in ['B', 'b']:
            color = Stone.BLACK
        elif arg_list[0][0] in ['W', 'b']:
            color = Stone.WHITE
        else:
            respond_failure("invalid color")

        self.time_manager.set_remaining_time(color, float(arg_list[1]))
        respond_success("")

    def _get_komi(self) -> NoReturn:
        """Handle the get_komi command.
        """
        respond_success(str(self.board.get_komi()))

    def _showboard(self) -> NoReturn:
        """Process the showboard command.
        Display the current board.
        """
        self.board.display()
        respond_success("")

    def _load_sgf(self, arg_list: List[str]) -> NoReturn:
        """Process the load_sgf command.
        Go to the stage where you have advanced to the specified turn of the specified SGF file.

        Args:
            arg_list (List[str]): Command argument list (file name (required), number (optional))
        """
        if not os.path.exists(arg_list[0]):
            respond_failure(f"cannot load {arg_list[0]}")

        sgf_data = SGFReader(arg_list[0], board_size=self.board.get_board_size())

        if len(arg_list) < 2:
            moves = sgf_data.get_n_moves()
        else:
            moves = int(arg_list[1])

        self.board.clear()

        for i in range(moves):
            pos = sgf_data.get_move_data(i)
            color = sgf_data.get_color(i)
            self.board.put_stone(pos, color)

        respond_success("")

    def run(self) -> NoReturn: # pylint: disable=R0912,R0915
        """Go Text Protocol client execution process.
        Execute the processing corresponding to the input command and display the response message.
        """
        while True:
            command = input()

            command_list = command.split(' ')

            input_gtp_command = command_list[0]

            if input_gtp_command == "version":
                _version()
            elif input_gtp_command == "protocol_version":
                _protocol_version()
            elif input_gtp_command == "name":
                _name()
            elif input_gtp_command == "quit":
                _quit()
            elif input_gtp_command == "known_command":
                self._known_command(command_list[1])
            elif input_gtp_command == "list_commands":
                self._list_commands()
            elif input_gtp_command == "komi":
                self._komi(command_list[1])
            elif input_gtp_command == "play":
                self._play(command_list[1], command_list[2])
            elif input_gtp_command == "genmove":
                self._genmove(command_list[1])
            elif input_gtp_command == "boardsize":
                self._boardsize(command_list[1])
            elif input_gtp_command == "clear_board":
                self._clear_board()
            elif input_gtp_command == "time_settings":
                self._time_settings(command_list[1:])
            elif input_gtp_command == "time_left":
                self._time_left(command_list[1:])
            elif input_gtp_command == "get_komi":
                self._get_komi()
            elif input_gtp_command == "showboard":
                self._showboard()
            elif input_gtp_command == "load_sgf":
                self._load_sgf(command_list[1:])
            elif input_gtp_command == "final_score":
                respond_success("?")
            elif input_gtp_command == "showstring":
                self.board.strings.display()
                respond_success("")
            elif input_gtp_command == "showpattern":
                coordinate = Coordinate(self.board.get_board_size())
                self.board.pattern.display(coordinate.convert_from_gtp_format(command_list[1]))
                respond_success("")
            elif input_gtp_command == "eye":
                coordinate = Coordinate(self.board.get_board_size())
                coord = coordinate.convert_from_gtp_format(command_list[1])
                print_err(self.board.pattern.get_eye_color(coord))
            elif input_gtp_command == "gogui-analyze_commands":
                response = ""
                for cmd in self.gogui_analyze_command:
                    response += cmd.get_command_information() + '\n'
                respond_success(response)
            elif input_gtp_command == "display_policy_black_color":
                respond_success(display_policy_distribution(
                    self.network, self.board, Stone.BLACK))
            elif input_gtp_command == "display_policy_white_color":
                respond_success(display_policy_distribution(
                    self.network, self.board, Stone.WHITE))
            elif input_gtp_command == "display_policy_black":
                respond_success(display_policy_score(
                    self.network, self.board, Stone.BLACK
                ))
            elif input_gtp_command == "display_policy_white":
                respond_success(display_policy_score(
                    self.network, self.board, Stone.WHITE
                ))
            elif input_gtp_command == "self-atari":
                self.board.display_self_atari(Stone.BLACK)
                self.board.display_self_atari(Stone.WHITE)
                respond_success("")
            else:
                respond_failure("unknown_command")


def respond_success(response: str) -> NoReturn:
    """Displays a response message when command processing is successful.

    Args:
        response (str): Response message to display.
    """
    print("= " + response + '\n')

def respond_failure(response: str) -> NoReturn:
    """Displays a response message when command processing fails.

    Args:
        response (str): Response message to display.
    """
    print("= ? " + response + '\n')

def _version() -> NoReturn:
    """Handle the version command.
    Show program version.
    """
    respond_success(VERSION)

def _protocol_version() -> NoReturn:
    """Handle the protocol_version command.
    Print the protocol version of GTP.
    """
    respond_success(PROTOCOL_VERSION)

def _name() -> NoReturn:
    """Handle the name command.
    Show program name.
    """
    respond_success(PROGRAM_NAME)

def _quit() -> NoReturn:
    """Handle the quit command.
    Exit the program.
    """
    respond_success("")
    sys.exit(0)
