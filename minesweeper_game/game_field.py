import random

from queue import SimpleQueue
from minesweeper import CellState, Mode


class MinesweeperGame:
    def __init__(self, mode: Mode, seed=None):
        self._mode = mode
        self._seed = seed
        self._is_field_created = False

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
                if self._field[check_row_idx][check_column_idx] == CellState.MINE:
                    mines_count += 1
        return mines_count

    def _create_field(self, first_cell_idx):
        random.seed(self._seed)

        mine_idx_range = (0, self._mode.width() * self._mode.height(), 1)
        mines = set()

        while len(mines) < self._mode.mines():
            mine_idx = random.randrange(*mine_idx_range)
            if mine_idx != first_cell_idx:
                mines.add(mine_idx)

        self._field = [[CellState.NO_MINES_NEARBY for _ in range(self._mode.width())] for _ in range(self._mode.height())]
        self._revealed_field = [[CellState.CLOSED for _ in range(self._mode.width())] for _ in range(self._mode.height())]

        for mine_idx in mines:
            self._field[mine_idx // self._mode.width()][mine_idx % self._mode.width()] = CellState.MINE

        for row_idx in range(self._mode.height()):
            for column_idx in range(self._mode.width()):
                if self._field[row_idx][column_idx] == CellState.MINE:
                    continue

                mines_count = self._count_nearest_mines(row_idx, column_idx)

                if mines_count == 1:
                    self._field[row_idx][column_idx] = CellState.ONE_MINE_NEARBY
                elif mines_count == 2:
                    self._field[row_idx][column_idx] = CellState.TWO_MINES_NEARBY
                elif mines_count == 3:
                    self._field[row_idx][column_idx] = CellState.THREE_MINES_NEARBY
                elif mines_count == 4:
                    self._field[row_idx][column_idx] = CellState.FOUR_MINES_NEARBY
                elif mines_count == 5:
                    self._field[row_idx][column_idx] = CellState.FIVE_MINES_NEARBY
                elif mines_count == 6:
                    self._field[row_idx][column_idx] = CellState.SIX_MINES_NEARBY
                elif mines_count == 7:
                    self._field[row_idx][column_idx] = CellState.SEVEN_MINES_NEARBY
                elif mines_count == 8:
                    self._field[row_idx][column_idx] = CellState.EIGHT_MINES_NEARBY

    def field(self):
        return self._revealed_field

    def open(self, idx):
        if not self._is_field_created:
            self._create_field(idx)

        row_idx = idx // self._mode.width()
        column_idx = idx % self._mode.width()

        cells_to_process = SimpleQueue()
        cells_to_process.put((row_idx, column_idx))

        while not cells_to_process.empty():
            row_idx, column_idx = cells_to_process.get()

            self._revealed_field[row_idx][column_idx] = self._field[row_idx][column_idx]
            if self._revealed_field[row_idx][column_idx] == CellState.NO_MINES_NEARBY:
                rows_to_check = self._idx_to_check(row_idx, self._mode.height() - 1)
                columns_to_check = self._idx_to_check(column_idx, self._mode.width() - 1)

                for check_row_idx in rows_to_check:
                    for check_column_idx in columns_to_check:
                        if self._revealed_field[check_row_idx][check_column_idx] == CellState.CLOSED:
                            cells_to_process.put((check_row_idx, check_column_idx))
