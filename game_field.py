import cv2 as cv
import numpy as np


class TrimMode:
    VERTICAL_LINES, HORIZONTAL_LINES = (0, 1)


class ColorComponent:
    B = 0


class CellState:
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


class GameField:
    def __init__(self, window_manager):
        self._window_manager = window_manager
        self._horizontal_lines = []
        self._vertical_lines = []
        self._min_distance_between_lines = 20
        self._field_header_height = 68
        self._field_left_top_corner = (0, 0)
        self._field_right_bottom_corner = (0, 0)

        self._create_field()

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
        cv.imwrite('window.png', img)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imwrite('gray.png', gray)

        edges = cv.Canny(gray, 250, 300)
        cv.imwrite('edges.png', edges)

        detected_lines = cv.HoughLines(edges, 0.1, np.pi / 180, 200)

        for line in detected_lines:
            rho, theta = line[0]
            a = int(np.cos(theta))
            b = int(np.sin(theta))

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

        self._field = [[CellState.CLOSED for _ in self._vertical_lines[:-1]] for _ in self._horizontal_lines[:-1]]

        self._field_left_top_corner = (self._vertical_lines[0], self._horizontal_lines[0])
        self._field_right_bottom_corner = (self._vertical_lines[-1], self._horizontal_lines[-1])

        # debug image
        for line in self._vertical_lines:
            x = line
            cv.line(img, (x, self._field_left_top_corner[1]), (x, self._field_right_bottom_corner[1]), (0, 0, 255), 1)

        for line in self._horizontal_lines:
            y = line
            cv.line(img, (self._field_left_top_corner[0], y), (self._field_right_bottom_corner[0], y), (0, 0, 255), 1)

        cv.imwrite('lines.png', img)

