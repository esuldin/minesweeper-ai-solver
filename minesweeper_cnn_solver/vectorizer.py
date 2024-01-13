import numpy

from minesweeper_game.game_interface import CellState


class MinesweeperFieldVectorizer:
    def __call__(self, field, dtype=numpy.float32):
        v = numpy.zeros((11,) + field.shape, dtype=dtype)

        v[0] = numpy.ones(field.shape)
        v[1] = (field != CellState.CLOSED)

        cell_states = [CellState.NO_MINES_NEARBY, CellState.ONE_MINE_NEARBY, CellState.TWO_MINES_NEARBY,
                       CellState.THREE_MINES_NEARBY, CellState.FOUR_MINES_NEARBY, CellState.FIVE_MINES_NEARBY,
                       CellState.SIX_MINES_NEARBY, CellState.SEVEN_MINES_NEARBY, CellState.EIGHT_MINES_NEARBY]
        for idx, state in enumerate(cell_states):
            v[idx + 2] = (field == state)

        return v
