import cv2
import numpy
import os

from minesweeper import CellState


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
    def __init__(self, window_manager):
        self._window_manager = window_manager
        self._pattern_library = PatternLibrary()
        self._horizontal_lines = []
        self._vertical_lines = []
        self._min_distance_between_lines = 20
        self._field_header_height = 68

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
        cv2.imwrite('../window.png', img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('../gray.png', gray)

        edges = cv2.Canny(gray, 250, 300)
        cv2.imwrite('../edges.png', edges)

        detected_lines = cv2.HoughLines(edges, 0.1, numpy.pi / 180, 200)

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

        self._field = [[CellState.CLOSED for _ in self._vertical_lines[:-1]] for _ in self._horizontal_lines[:-1]]

        _field_left_top_corner = (self._vertical_lines[0], self._horizontal_lines[0])
        _field_right_bottom_corner = (self._vertical_lines[-1], self._horizontal_lines[-1])

        # debug image
        for line in self._vertical_lines:
            x = line
            cv2.line(img, (x, _field_left_top_corner[1]), (x, _field_right_bottom_corner[1]), (0, 0, 255), 1)

        for line in self._horizontal_lines:
            y = line
            cv2.line(img, (_field_left_top_corner[0], y), (_field_right_bottom_corner[0], y), (0, 0, 255), 1)

        cv2.imwrite('../lines.png', img)

    def _update_field(self):
        img = self._window_manager.get_picture()
        cv2.imwrite('../update_field.png', img)

        for horizontal_line_idx in range(len(self._horizontal_lines) - 1):
            for vertical_line_idx in range(len(self._vertical_lines) - 1):
                top = self._horizontal_lines[horizontal_line_idx]
                left = self._vertical_lines[vertical_line_idx]
                bottom = self._horizontal_lines[horizontal_line_idx + 1]
                right = self._vertical_lines[vertical_line_idx + 1]

                cell_img = img[top:bottom, left:right]

                #if horizontal_line_idx == 1 and vertical_line_idx == 29:
                #    cv2.imwrite('%d-%d.png' % (horizontal_line_idx, vertical_line_idx), cell_img)

                self._field[horizontal_line_idx][vertical_line_idx] = self._pattern_library.match(cell_img)

    def field(self):
        return self._field

    def open(self, idx):
        y_idx = idx // (len(self._vertical_lines) - 1)
        x_idx = idx % (len(self._vertical_lines) - 1)

        x = (self._vertical_lines[x_idx] + self._vertical_lines[x_idx + 1]) // 2
        y = (self._horizontal_lines[y_idx] + self._horizontal_lines[y_idx + 1]) // 2

        self._window_manager.click(x, y)
        self._update_field()