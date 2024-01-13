import argparse
import numpy
import os
import time

from minesweeper_game.game_interface import GameState, Mode
from minesweeper_game.game_field import MinesweeperGame, MinesweeperFieldPseudoGraphicsVisualizer

from minesweeper_cnn_solver import MinesweeperSolver, MinesweeperSolverModel


class ConsoleVisualizerMode:
    DEMO = 1
    LOG = 2
    STATISTICS_ONLY = 3


class ConsoleVisualizer:
    def __init__(self, mode):
        self._frame_presenting_time = 0.7
        self._mode = mode
        self._pseudographicsvisualizer = MinesweeperFieldPseudoGraphicsVisualizer()

    def draw(self, field):
        if self._mode == ConsoleVisualizerMode.DEMO:
            os.system('cls' if os.name == 'nt' else 'clear')

        if self._mode != ConsoleVisualizerMode.STATISTICS_ONLY:
            print(self._pseudographicsvisualizer.draw(field))

        if self._mode == ConsoleVisualizerMode.DEMO:
            time.sleep(self._frame_presenting_time)


parser = argparse.ArgumentParser(description='Play Minesweeper game simulation using pretrained model.')
parser.add_argument('-g', '--game-mode', help='The Minesweeper game mode to play.',
                    default='classic', choices=['classic', 'easy', 'medium', 'expert', 'custom'])
parser.add_argument('-c', '--custom-mode', help='The configuration of the custom game mode in the following format:'
                                                ' {field width}x{field height}x{number of mines}, e.g.: 8x8x8.',
                    default=None)
parser.add_argument('-m', '--model', help='The path to pretrained model.',
                    default=None)
parser.add_argument('-n', '--number-of-games', help='The number of time the games is played.',
                    default=1, type=int)
parser.add_argument('-o', '--output-mode', help='The output mode.',
                    default='demo', choices=['demo', 'log', 'statistics-only'])

args = parser.parse_args()

if args.game_mode == 'classic':
    game_mode = Mode.CLASSIC
elif args.game_mode == 'easy':
    game_mode = Mode.EASY
elif args.game_mode == 'medium':
    game_mode = Mode.MEDIUM
elif args.game_mode == 'expert':
    game_mode = Mode.EXPERT
else:
    if not args.custom_mode:
        raise ValueError('--custom-mode option must be specified.')

    mode_options = [int(x) for x in args.custom_mode.split('x')]
    game_mode = Mode(*mode_options)

if args.output_mode == 'demo':
    console_visualizer = ConsoleVisualizer(ConsoleVisualizerMode.DEMO)
elif args.output_mode == 'log':
    console_visualizer = ConsoleVisualizer(ConsoleVisualizerMode.LOG)
elif args.output_mode == 'statistics-only':
    console_visualizer = ConsoleVisualizer(ConsoleVisualizerMode.STATISTICS_ONLY)
else:
    raise ValueError('Unexpected output mode is specified.')

if args.model:
    model = MinesweeperSolverModel.fromfile(args.model)
elif game_mode == Mode.CLASSIC:
    model = MinesweeperSolverModel.fromfile('trained_models/classic_minesweeper_model.pt')
elif game_mode == Mode.EASY:
    model = MinesweeperSolverModel.fromfile('trained_models/easy_minesweeper_model.pt')
elif game_mode == Mode.MEDIUM:
    model = MinesweeperSolverModel.fromfile('trained_models/medium_minesweeper_model.pt')
elif game_mode == Mode.EXPERT:
    model = MinesweeperSolverModel.fromfile('trained_models/expert_minesweeper_model.pt')
else:
    raise ValueError('The model cannot be selected.')

solver = MinesweeperSolver(model)

games_won = 0

for game_idx in range(args.number_of_games):
    game = MinesweeperGame(game_mode)
    console_visualizer.draw(game.field())

    cell_idx = numpy.ravel_multi_index((game.field().shape[0] // 2, game.field().shape[1] // 2), game.field().shape)

    while game.open(cell_idx) == GameState.IN_PROGRESS:
        cell_idx, _ = solver(game.field())
        console_visualizer.draw(game.field())

    console_visualizer.draw(game.field())

    if game.state() == GameState.GAME_OVER:
        print(f'Game {game_idx}: Game Over!')
    else:
        print(f'Game {game_idx}: Win!')
        games_won += 1

print('Statistics:')
print(f' Games played: {args.number_of_games}')
print(f' Games won: {games_won}')
print(f' Win percentage: {games_won/args.number_of_games}')
