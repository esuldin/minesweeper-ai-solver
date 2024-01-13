import cv2
import numpy
import os
import time

from minesweeper_game.game_interface import CellState, GameState, Mode


class TrimMode:
    VERTICAL_LINES, HORIZONTAL_LINES = (0, 1)


class ColorComponent:
    B = 0


class PatternLibrary:
    def __init__(self, directory=None):
        pattern_images = {
            CellState.MINE: 'mine.png',
            CellState.CLOSED: 'closed.png',
            CellState.NO_MINES_NEARBY: 'empty.png',
            CellState.ONE_MINE_NEARBY: '1.png',
            CellState.TWO_MINES_NEARBY: '2.png',
            CellState.THREE_MINES_NEARBY: '3.png',
            CellState.FOUR_MINES_NEARBY: '4.png',
            CellState.FIVE_MINES_NEARBY: '5.png',
            CellState.SIX_MINES_NEARBY: '6.png',
            CellState.SEVEN_MINES_NEARBY: '7.png',
            CellState.EIGHT_MINES_NEARBY: '8.png',
        }

        if directory is None:
            directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'patterns')

        self._image_size = (52, 51)

        self._patterns = {
            key: cv2.imread(os.path.join(directory, value))
            for key, value in pattern_images.items()}

    def match(self, img):
        scaled_img = cv2.resize(img, self._image_size)
        cv2.imwrite('scaled.png', scaled_img)

        min_distance = None
        matched_state = None

        for state, pattern in self._patterns.items():
            match_result = cv2.matchTemplate(scaled_img, pattern, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, _, _ = cv2.minMaxLoc(match_result)

            if min_distance is None or min_val < min_distance:
                min_distance = min_val
                matched_state = state

        return matched_state


class MsMinesweeperClassicField:
    def __init__(self, window_manager, mode=None):
        self._window_manager = window_manager
        self._pattern_library = PatternLibrary()
        self._horizontal_lines = []
        self._vertical_lines = []
        self._min_distance_between_lines = 20
        self._field_header_height = 68
        self._mode = mode
        self._state = None

        self._create_field()

        self._enable_debug_dump_img = os.environ.get('MSAIS_DEBUG', False)
        self._debug_dump_img_idx = 0

    def _dump_debug_img(self, name, img):
        if self._enable_debug_dump_img:
            self._enable_debug_dump_img += 1
            cv2.imwrite(f'{self._enable_debug_dump_img}_{name}', img)

    def _dump_detected_lines(self, img):
        _field_left_top_corner = (self._vertical_lines[0], self._horizontal_lines[0])
        _field_right_bottom_corner = (self._vertical_lines[-1], self._horizontal_lines[-1])

        # debug image
        for line in self._vertical_lines:
            x = line
            cv2.line(img, (x, _field_left_top_corner[1]), (x, _field_right_bottom_corner[1]), (0, 0, 255), 1)

        for line in self._horizontal_lines:
            y = line
            cv2.line(img, (_field_left_top_corner[0], y), (_field_right_bottom_corner[0], y), (0, 0, 255), 1)

        self._dump_debug_img('lines.png', img)

    def _merge_nearby_lines(self, lines, threshold=None):
        original_lines = sorted(lines)
        if not original_lines:
            return list()

        if threshold is None:
            threshold = self._min_distance_between_lines

        prev_line = original_lines[0]
        merged_lines = []
        for line in original_lines[1:]:
            if (line - prev_line) < threshold:
                prev_line = (line + prev_line) // 2
            else:
                merged_lines.append(prev_line)
                prev_line = line

        merged_lines.append(prev_line)
        return merged_lines

    def _trim_lines(self, img, lines, direction, coordinate, color_component, color_value, threshold):
        if not lines:
            return []

        trimmed_lines = []
        prev_line = lines[0]
        for line in lines[1:]:
            if line < self._field_header_height:
                prev_line = line
                continue

            second_coordinate = (prev_line + line) // 2
            y, x = (coordinate, second_coordinate) if direction == 0 else (second_coordinate, coordinate)
            color = img[y, x, color_component]
            if abs(color - color_value) < color_value * threshold:
                trimmed_lines.append(prev_line)
                prev_line = line
            else:
                if trimmed_lines:
                    return trimmed_lines
                else:
                    prev_line = line
        trimmed_lines.append(prev_line)
        return trimmed_lines

    def _create_field(self):
        img = self._window_manager.get_picture()
        self._dump_debug_img('window.png', img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._dump_debug_img('gray.png', gray)

        edges = cv2.Canny(gray, 250, 300)
        self._dump_debug_img('edges.png', edges)

        detected_lines = cv2.HoughLines(edges, 0.1, numpy.pi / 180, 170)

        for line in detected_lines:
            rho, theta = line[0]
            a = int(numpy.cos(theta))
            b = int(numpy.sin(theta))

            if a == 0 and b == 1:
                self._horizontal_lines.append(int(rho))
            elif a == 1 and b == 0:
                self._vertical_lines.append(int(rho))

        self._horizontal_lines = self._merge_nearby_lines(self._horizontal_lines)
        self._vertical_lines = self._merge_nearby_lines(self._vertical_lines)

        x_coordinate = (self._vertical_lines[0] + self._vertical_lines[1]) // 2
        color_value = 255
        color_value_threshold = 0.15
        self._horizontal_lines = self._trim_lines(img, self._horizontal_lines, TrimMode.HORIZONTAL_LINES,
                                                  x_coordinate, ColorComponent.B, color_value, color_value_threshold)
        self._dump_detected_lines(img)

        self._field = numpy.full((len(self._horizontal_lines) - 1, len(self._vertical_lines) - 1), CellState.CLOSED)
        self._update_game_mode()
        self._update_state()

    def _game_mode_by_field_shape(self, shape):
        if shape == (8, 8):
            return Mode.CLASSIC
        elif shape == (9, 9):
            return Mode.EASY
        elif shape == (16, 16):
            return Mode.MEDIUM
        elif shape == (16, 30):
            return Mode.EXPERT
        else:
            raise RuntimeError('Cannot determine the game mode.')

    def _update_game_mode(self):
        if self._mode is None:
            self._mode = self._game_mode_by_field_shape(self._field.shape)
        elif self._field.shape != (self._mode.height(), self._mode.width()):
            raise RuntimeError('The field shape does not correspond the specified game mode.')

    def _update_field(self):
        img = self._window_manager.get_picture()
        self._dump_debug_img('updated_field.png', img)

        for horizontal_line_idx in range(len(self._horizontal_lines) - 1):
            for vertical_line_idx in range(len(self._vertical_lines) - 1):
                top = self._horizontal_lines[horizontal_line_idx]
                left = self._vertical_lines[vertical_line_idx]
                bottom = self._horizontal_lines[horizontal_line_idx + 1]
                right = self._vertical_lines[vertical_line_idx + 1]

                cell_img = img[top:bottom, left:right]
                self._field[horizontal_line_idx, vertical_line_idx] = self._pattern_library.match(cell_img)
        self._update_state()

    def _update_state(self):
        if self._state is None or self._state == GameState.IN_PROGRESS:
            if numpy.count_nonzero(self._field == CellState.MINE):
                self._state = GameState.GAME_OVER
            elif numpy.count_nonzero(self._field == CellState.CLOSED) == self._mode.mines():
                self._state = GameState.WIN
            else:
                self._state = GameState.IN_PROGRESS
        return self._state

    def field(self):
        return self._field

    def state(self):
        return self._state

    def mode(self):
        return self._mode

    def _cell_idx_to_window_coordinates(self, idx):
        y_idx = idx // (len(self._vertical_lines) - 1)
        x_idx = idx % (len(self._vertical_lines) - 1)

        x = (self._vertical_lines[x_idx] + self._vertical_lines[x_idx + 1]) // 2
        y = (self._horizontal_lines[y_idx] + self._horizontal_lines[y_idx + 1]) // 2

        return x, y

    def open(self, idx):
        x, y = self._cell_idx_to_window_coordinates(idx)
        self._window_manager.click(x, y)

        # skip animation before updating the field
        time.sleep(0.3)
        self._update_field()

        return self._state
