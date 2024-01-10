import random
import numpy

from queue import SimpleQueue
from minesweeper import CellState, Mode, GameState


class MinesweeperGame:
    def __init__(self, mode: Mode, seed=None):
        self._mode = mode
        self._seed = seed
        self._field = None
        self._revealed_field = numpy.full((self._mode.height(), self._mode.width()),
                                          CellState.CLOSED, dtype=numpy.int8)
        self._state = None
        self._update_state()

    def _idx_to_check(self, current_idx, max_idx_value):
        idx_to_check = [current_idx]
        if current_idx > 0:
            idx_to_check.append(current_idx - 1)
        if current_idx < max_idx_value:
            idx_to_check.append(current_idx + 1)
        return idx_to_check

    def _count_nearest_mines(self, row_idx, column_idx):
        rows_to_check = self._idx_to_check(row_idx, self._mode.height() - 1)
        columns_to_check = self._idx_to_check(column_idx, self._mode.width() - 1)

        mines_count = 0
        for check_row_idx in rows_to_check:
            for check_column_idx in columns_to_check:
                if self._field[check_row_idx, check_column_idx] == CellState.MINE:
                    mines_count += 1
        return mines_count

    def _generate_mines_idx(self, first_opened_cell_idx):
        row_idx, column_idx = numpy.unravel_index(first_opened_cell_idx, self._mode.shape())
        rows_to_exclude = self._idx_to_check(row_idx, self._mode.height() - 1)
        columns_to_exclude = self._idx_to_check(column_idx, self._mode.width() - 1)

        idx_to_exclude = {numpy.ravel_multi_index((row_idx, column_idx), self._mode.shape())
                          for row_idx in rows_to_exclude for column_idx in columns_to_exclude}

        random.seed(self._seed)

        mine_idx_range = (0, self._mode.width() * self._mode.height(), 1)
        mines = set()

        while len(mines) < self._mode.mines():
            mine_idx = random.randrange(*mine_idx_range)
            if mine_idx not in idx_to_exclude:
                mines.add(mine_idx)

        return mines

    def _create_field(self, first_opened_cell_idx):
        self._field = numpy.full((self._mode.height(), self._mode.width()),
                                 CellState.NO_MINES_NEARBY, dtype=numpy.int8)

        mines = self._generate_mines_idx(first_opened_cell_idx)
        for mine_idx in mines:
            self._field[numpy.unravel_index(mine_idx, self._mode.shape())] = CellState.MINE

        for row_idx in range(self._mode.height()):
            for column_idx in range(self._mode.width()):
                if self._field[row_idx, column_idx] == CellState.MINE:
                    continue

                mines_count = self._count_nearest_mines(row_idx, column_idx)

                if mines_count == 1:
                    self._field[row_idx, column_idx] = CellState.ONE_MINE_NEARBY
                elif mines_count == 2:
                    self._field[row_idx, column_idx] = CellState.TWO_MINES_NEARBY
                elif mines_count == 3:
                    self._field[row_idx, column_idx] = CellState.THREE_MINES_NEARBY
                elif mines_count == 4:
                    self._field[row_idx, column_idx] = CellState.FOUR_MINES_NEARBY
                elif mines_count == 5:
                    self._field[row_idx, column_idx] = CellState.FIVE_MINES_NEARBY
                elif mines_count == 6:
                    self._field[row_idx, column_idx] = CellState.SIX_MINES_NEARBY
                elif mines_count == 7:
                    self._field[row_idx, column_idx] = CellState.SEVEN_MINES_NEARBY
                elif mines_count == 8:
                    self._field[row_idx, column_idx] = CellState.EIGHT_MINES_NEARBY

    def field(self):
        return self._revealed_field

    def mode(self):
        return self._mode

    def open(self, idx):
        if self._field is None:
            self._create_field(first_opened_cell_idx=idx)

        if self._state != GameState.IN_PROGRESS:
            return self._state

        row_idx = idx // self._mode.width()
        column_idx = idx % self._mode.width()

        cells_to_process = SimpleQueue()
        cells_to_process.put((row_idx, column_idx))

        while not cells_to_process.empty():
            row_idx, column_idx = cells_to_process.get()

            self._revealed_field[row_idx, column_idx] = self._field[row_idx, column_idx]
            if self._revealed_field[row_idx, column_idx] == CellState.NO_MINES_NEARBY:
                rows_to_check = self._idx_to_check(row_idx, self._mode.height() - 1)
                columns_to_check = self._idx_to_check(column_idx, self._mode.width() - 1)

                for check_row_idx in rows_to_check:
                    for check_column_idx in columns_to_check:
                        if self._revealed_field[check_row_idx, check_column_idx] == CellState.CLOSED:
                            cells_to_process.put((check_row_idx, check_column_idx))

        return self._update_state()

    def _update_state(self):
        if self._state is None or self._state == GameState.IN_PROGRESS:
            if numpy.count_nonzero(self._revealed_field == CellState.MINE):
                self._state = GameState.GAME_OVER
            elif numpy.count_nonzero(self._revealed_field == CellState.CLOSED) == self._mode.mines():
                self._state = GameState.WIN
            else:
                self._state = GameState.IN_PROGRESS
        return self._state

    def state(self):
        return self._state
