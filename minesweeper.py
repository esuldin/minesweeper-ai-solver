class Mode:
    def __init__(self, width, height, mines):
        self._height = height
        self._width = width
        self._mines = mines

    def height(self):
        return self._height

    def width(self):
        return self._width

    def mines(self):
        return self._mines


Mode.CLASSIC = Mode(8, 8, 9)
Mode.EASY = Mode(9, 9, 10)
Mode.MEDIUM = Mode(16, 16, 40)
Mode.EXPERT = Mode(30, 16, 99)


class CellState:
    MINE = -2
    CLOSED = -1
    NO_MINES_NEARBY = 0
    ONE_MINE_NEARBY = 1
    TWO_MINES_NEARBY = 2
    THREE_MINES_NEARBY = 3
    FOUR_MINES_NEARBY = 4
    FIVE_MINES_NEARBY = 5
    SIX_MINES_NEARBY = 6
    SEVEN_MINES_NEARBY = 7
    EIGHT_MINES_NEARBY = 8


class GameState:
    IN_PROGRESS = 1
    GAME_OVER = 2
    WIN = 3
