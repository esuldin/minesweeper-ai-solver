import numpy
import unittest

from minesweeper_game.game_field import MinesweeperGame
from minesweeper_game.game_interface import Mode, CellState, GameState


class TestMinesweeper(unittest.TestCase):
    def test_flood_open(self):
        game = MinesweeperGame(Mode.CLASSIC, 0)
        self.assertEqual(numpy.count_nonzero(game.field() == CellState.CLOSED),
                         game.mode().height() * game.mode().width())

        game.open(0)
        expected_field = numpy.array([[ 0,  0,  0,  0,  1, -1,  1,  0],
                                      [ 0,  0,  0,  0,  1,  1,  1,  0],
                                      [ 0,  0,  0,  0,  0,  0,  0,  0],
                                      [ 1,  1,  1,  0,  0,  1,  1,  1],
                                      [-1, -1,  1,  0,  1,  2, -1, -1],
                                      [-1, -1,  3,  1,  3, -1, -1, -1],
                                      [-1, -1, -1, -1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1, -1, -1, -1]])
        self.assertTrue(numpy.array_equiv(game.field(), expected_field))

    def test_open_near_mine(self):
        game = MinesweeperGame(Mode.CLASSIC, 0)
        self.assertEqual(numpy.count_nonzero(game.field() == CellState.CLOSED),
                         game.mode().height() * game.mode().width())

        game.open(4)
        expected_field = numpy.array([[ 0,  0,  0,  0,  0,  0,  0,  0],
                                      [ 0,  0,  0,  0,  0,  0,  0,  0],
                                      [ 0,  0,  1,  1,  1,  0,  0,  0],
                                      [ 1,  1,  2, -1,  1,  1,  1,  1],
                                      [-1, -1, -1, -1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1, -1, -1, -1]])
        self.assertTrue(numpy.array_equiv(game.field(), expected_field))

    def test_classic_win(self):
        game = MinesweeperGame(Mode.CLASSIC, 0)
        self.assertEqual(numpy.count_nonzero(game.field() == CellState.CLOSED),
                         game.mode().height() * game.mode().width())

        game.open(0)
        self.assertEqual(game.state(), GameState.IN_PROGRESS)

        cells_to_open = [24, 25, 26, 29, 30, 31,
                         32, 34, 36, 37, 39,
                         40, 41, 42, 43, 44, 46, 47,
                         48, 50, 52, 54, 55,
                         56, 57, 58, 59, 60]
        for idx in cells_to_open:
            game.open(idx)
            self.assertEqual(game.state(), GameState.IN_PROGRESS)

        last_cell = 63
        game.open(last_cell)
        self.assertEqual(game.state(), GameState.WIN)

        expected_field = numpy.array([[0,  0,  0,  0,  1, -1,  1, 0],
                                      [0,  0,  0,  0,  1,  1,  1, 0],
                                      [0,  0,  0,  0,  0,  0,  0, 0],
                                      [1,  1,  1,  0,  0,  1,  1, 1],
                                      [1, -1,  1,  0,  1,  2, -1, 1],
                                      [2,  2,  3,  1,  3, -1,  3, 1],
                                      [1, -1,  2, -1,  4, -1,  4, 1],
                                      [1,  1,  2,  1,  3, -1, -1, 1]])
        self.assertTrue(numpy.array_equiv(game.field(), expected_field))

        cell_with_mine = 5
        game.open(cell_with_mine)
        self.assertEqual(game.state(), GameState.WIN)
        self.assertTrue(numpy.array_equiv(game.field(), expected_field))

    def test_classic_game_over(self):
        game = MinesweeperGame(Mode.CLASSIC, 0)
        self.assertEqual(numpy.count_nonzero(game.field() == CellState.CLOSED),
                         game.mode().height() * game.mode().width())

        first_cell_idx = 3
        game.open(first_cell_idx)
        self.assertEqual(game.state(), GameState.IN_PROGRESS)

        cell_with_mine_idx = 5
        game.open(cell_with_mine_idx)
        self.assertEqual(game.state(), GameState.GAME_OVER)

        expected_field = numpy.array([[ 0,  0,  0,  0,  1, -2,  1,  0],
                                      [ 0,  0,  0,  0,  1,  1,  1,  0],
                                      [ 0,  0,  0,  0,  0,  0,  0,  0],
                                      [ 1,  1,  1,  0,  0,  1,  1,  1],
                                      [-1, -1,  1,  0,  1,  2, -1, -1],
                                      [-1, -1,  3,  1,  3, -1, -1, -1],
                                      [-1, -1, -1, -1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1, -1, -1, -1]])
        self.assertTrue(numpy.array_equiv(game.field(), expected_field))

        closed_cell_idx = 3
        game.open(closed_cell_idx)
        self.assertEqual(game.state(), GameState.GAME_OVER)
        self.assertTrue(numpy.array_equiv(game.field(), expected_field))

    def test_open_mine_as_first_step(self):
        game = MinesweeperGame(Mode.CLASSIC, 0)

        cell_with_mine_idx = 5
        game.open(cell_with_mine_idx)
        self.assertEqual(game.state(), GameState.IN_PROGRESS)

        expected_field = numpy.array([[ 0,  0,  0,  0,  0,  0,  0,  0],
                                      [ 0,  0,  0,  0,  0,  0,  0,  0],
                                      [ 0,  0,  1,  1,  1,  0,  0,  0],
                                      [ 1,  1,  2, -1,  1,  1,  1,  1],
                                      [-1, -1, -1, -1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1, -1, -1, -1]])
        self.assertTrue(numpy.array_equiv(game.field(), expected_field))

    def test_init_state(self):
        game = MinesweeperGame(Mode.CLASSIC, 0)
        self.assertEqual(game.state(), GameState.IN_PROGRESS)


if __name__ == '__main__':
    unittest.main()
