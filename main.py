from game_window_manager import MsMinesweeperWindowManager
from game_field import MsMinesweeperClassicField

window_manager = MsMinesweeperWindowManager()
field = MsMinesweeperClassicField(window_manager)

print(field._field)
