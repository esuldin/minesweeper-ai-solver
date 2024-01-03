from game_window_manager import GameWindowManager
from game_field import GameField

window_manager = GameWindowManager()
field = GameField(window_manager)

print(field._field)
