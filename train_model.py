import argparse

from minesweeper_game.game_interface import Mode
from minesweeper_cnn_solver import MinesweeperSolver, MinesweeperSolverTrainer, MinesweeperSolverModel

parser = argparse.ArgumentParser(description='Train CNN model.')
parser.add_argument('-g', '--game-mode', help='The Minesweeper game mode to train the model.',
                    default='classic', choices=['classic', 'easy', 'medium', 'expert', 'custom'])
parser.add_argument('-c', '--custom-mode', help='The configuration of the custom game mode in the following format:'
                                                ' {field width}x{field height}x{number of mines}, e.g.: 8x8x8.',
                    default=None)
parser.add_argument('-t', '--training-iterations', help='The number of training iterations.',
                    default=1000, type=int)
parser.add_argument('-e', '--epochs', help='The number of epochs during one training iteration.',
                    default=4, type=int)
parser.add_argument('-b', '--batches', help='The number of batches in one epoch.',
                    default=5, type=int)
parser.add_argument('-s', '--batch-size', help='The number of samples in one batch.',
                    default=200, type=int)
parser.add_argument('-i', '--input', help='The path to pretrained model.',
                    default=None)
parser.add_argument('-o', '--output', help='The path to keep trained model.',
                    default=None)

args = parser.parse_args()

if args.game_mode == 'classic':
    selected_game_mode = Mode.CLASSIC
elif args.game_mode == 'easy':
    selected_game_mode = Mode.EASY
elif args.game_mode == 'medium':
    selected_game_mode = Mode.MEDIUM
elif args.game_mode == 'expert':
    selected_game_mode = Mode.EXPERT
else:
    if not args.custom_mode:
        raise ValueError('--custom-mode option must be specified.')

    mode_options = [int(x) for x in args.custom_mode.split('x')]
    selected_game_mode = Mode(*mode_options)

model = MinesweeperSolverModel.fromfile(args.input) if args.input else None
solver = MinesweeperSolver(model)
trainer = MinesweeperSolverTrainer(selected_game_mode, solver)
trainer.train(args.training_iterations, args.epochs, args.batches, args.batch_size)

output_file = args.output if args.output else \
    f'{selected_game_mode}_{args.training_iterations}ti_{args.epochs}e_{args.batches}b_{args.batch_size}.pt'
solver.model().save(output_file)
