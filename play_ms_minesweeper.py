import numpy

from minesweeper_game.game_interface import GameState, Mode

from ms_minesweeper_game import MsMinesweeperClassicField, MsMinesweeperWindowManager
from minesweeper_cnn_solver import MinesweeperSolver, MinesweeperSolverModel


window_manager = MsMinesweeperWindowManager()
game = MsMinesweeperClassicField(window_manager)

if game.mode() == Mode.CLASSIC:
    model = MinesweeperSolverModel.fromfile('classic_minesweeper_model.pt')
elif game.mode() == Mode.EASY:
    model = MinesweeperSolverModel.fromfile('easy_1000ti_4e_5b_200.pt')
elif game.mode() == Mode.MEDIUM:
    model = MinesweeperSolverModel.fromfile('medium_minesweeper_model_4000ti.pt')
elif game.mode() == Mode.EXPERT:
    model = MinesweeperSolverModel.fromfile('expert_minesweeper_model_4000ti.pt')
else:
    raise NotImplementedError('Only standard modes are supported right now.')

solver = MinesweeperSolver(model)

cell_idx = numpy.ravel_multi_index((game.field().shape[0] // 2, game.field().shape[1] // 2), game.field().shape)

while game.open(cell_idx) == GameState.IN_PROGRESS:
    cell_idx, _ = solver(game.field())

if game.state() == GameState.GAME_OVER:
    print('game over!')
else:
    print('win!')
