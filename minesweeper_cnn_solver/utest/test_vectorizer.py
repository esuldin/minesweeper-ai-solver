import numpy
import unittest

from ..vectorizer import CellState
from ..vectorizer import MinesweeperFieldVectorizer


class TestMinesweeper(unittest.TestCase):
    def test_classic_field_vectorization(self):
        vect = MinesweeperFieldVectorizer()

        cell_states = (CellState.CLOSED, CellState.NO_MINES_NEARBY, CellState.ONE_MINE_NEARBY,
                       CellState.TWO_MINES_NEARBY, CellState.THREE_MINES_NEARBY, CellState.FOUR_MINES_NEARBY,
                       CellState.FIVE_MINES_NEARBY, CellState.SIX_MINES_NEARBY, CellState.SEVEN_MINES_NEARBY,
                       CellState.EIGHT_MINES_NEARBY)

        sc, sn, s1, s2, s3, s4, s5, s6, s7, s8 = cell_states
        game_field = numpy.array([[sn, sn, sn, sn, sn, sn, sn, sn],
                                  [s1, s2, s3, s3, s3, s3, s2, s1],
                                  [s2, sc, sc, sc, sc, sc, sc, s2],
                                  [s2, sc, s6, s7, sc, s8, sc, s3],
                                  [s1, s2, sc, sc, sc, sc, sc, s2],
                                  [sn, s1, s4, sc, s5, s3, s2, s1],
                                  [sn, sn, s2, sc, s2, sn, sn, sn],
                                  [sn, sn, s1, s1, s1, sn, sn, sn]])

        vect_res = vect(game_field)
        self.assertEqual(vect_res.shape, (11,) + game_field.shape)

        # Check the matrix that represents the field
        self.assertTrue(numpy.array_equiv(vect_res[0], numpy.ones(game_field.shape)))

        # Check the matrix that represents revealed cells
        self.assertTrue(numpy.array_equiv(vect_res[1], game_field != sc))

        for idx, state in enumerate(cell_states[1:]):
            self.assertTrue(numpy.array_equiv(vect_res[idx + 2], game_field == state))


if __name__ == '__main__':
    unittest.main()
