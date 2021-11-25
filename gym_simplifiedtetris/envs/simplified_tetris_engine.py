import random
import time
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import cv2.cv2 as cv
import numpy as np
from matplotlib import colors
from PIL import Image

from ..utils.piece import Piece


class SimplifiedTetrisEngine(object):
    """
    A class representing a simplified Tetris engine containing methods
    that retrieve the actions available for each of the pieces in use, drop
    pieces vertically downwards having identified the correct location to
    drop them, clear full rows, and render a game of Tetris.

    :param grid_dims: the grid dimensions (height and width).
    :param piece_size: the size of the pieces in use.
    :param num_pieces: the number of pieces in use.
    :param num_actions: the number of available actions in each state.
    """

    @staticmethod
    def _get_bgr_code(colour_name: str) -> Tuple[float, float, float]:
        """
        Gets the BGR code corresponding to the arg provided.

        :param colour_name: a string of the colour name,
        :return: an inverted RGB code of the inputted colour name.
        """
        return tuple(np.array([255, 255, 255]) * colors.to_rgb(colour_name))[::-1]

    @staticmethod
    def _close() -> None:
        """Closes the open windows."""
        cv.waitKey(1)
        cv.destroyAllWindows()
        cv.waitKey(1)

    def __init__(
        self,
        grid_dims: Sequence[int],
        piece_size: int,
        num_pieces: int,
        num_actions: int,
    ):
        self._height, self._width = grid_dims
        self._piece_size = piece_size
        self._num_pieces = num_pieces
        self._num_actions = num_actions

        self._grid = np.zeros((grid_dims[1], grid_dims[0]), dtype="bool")
        self._colour_grid = np.zeros((grid_dims[1], grid_dims[0]), dtype="int")
        self._anchor = [grid_dims[1] / 2 - 1, piece_size - 1]

        self._final_scores = np.array([], dtype=int)
        self._sleep_time = 500
        self._show_agent_playing = True
        self._cell_size = int(min(0.8 * 1000 / grid_dims[0], 0.8 * 2000 / grid_dims[1]))
        self._img = np.array([])
        self._last_move_info = {}

        self._black = self._get_bgr_code("black")
        self._white = self._get_bgr_code("white")
        self._grid_colours = [
            self._white,  # Empty.
            self._get_bgr_code("cyan"),  # 'I'.
            self._get_bgr_code("orange"),  # 'L'.
            self._get_bgr_code("yellow"),  #  'O'.
            self._get_bgr_code("purple"),  # 'T'.
            self._get_bgr_code("blue"),  # 'J'.
            self._get_bgr_code("green"),  # 'S'.
            self._get_bgr_code("red"),  # 'Z'.
        ]

        self._initialise_pieces()
        self._update_coords_and_anchor()
        self._get_all_available_actions()
        self._reset()

    def _generate_id_randomly(self) -> int:
        """Randomly generates an id."""
        return random.randint(0, self._num_pieces - 1)

    def _initialise_pieces(self) -> None:
        """Creates a dictionary containing the pieces."""
        self._pieces = {}
        for idx in range(self._num_pieces):
            self._pieces[idx] = Piece(self._piece_size, idx)

    def _reset(self) -> None:
        """Resets the score, grid, piece coords, piece id and anchor."""
        self._score = 0
        self._grid = np.zeros_like(self._grid, dtype="bool")
        self._colour_grid = np.zeros_like(self._colour_grid, dtype="int")
        self._update_coords_and_anchor()

    def _render(self, mode: Optional[str] = "human") -> np.ndarray:
        """
        Shows an image of the current grid, having dropped the current piece.
        The human has the option to pause (SPACEBAR), speed up (RIGHT key),
        slow down (LEFT key) or quit (ESC) the window.

        :param mode: the render mode.
        :return: the image pixel values.
        """

        grid = self._get_grid()
        self._resize_grid(grid)
        self._draw_separating_lines()
        self._add_img_left()
        self._draw_boundary()

        if mode == "human":
            if self._show_agent_playing:

                cv.imshow("Simplified Tetris", self._img)
                k = cv.waitKey(self._sleep_time)

                if k == 3:  # right arrow
                    self._sleep_time -= 100

                    if self._sleep_time < 100:
                        self._sleep_time = 1

                    time.sleep(self._sleep_time / 1000)
                elif k == 2:  # Left arrow.
                    self._sleep_time += 100
                    time.sleep(self._sleep_time / 1000)
                elif k == 27:  # Esc.
                    self._show_agent_playing = False
                    self._close()
                elif k == 32:  # Spacebar.
                    while True:
                        j = cv.waitKey(30)

                        if j == 32:  # Spacebar.
                            break

                        if j == 27:  # Esc.
                            self._show_agent_playing = False
                            self._close()
                            break
        elif mode == "rgb_array":
            return self._img
        else:
            raise ValueError('Mode should be "human".')

    def _draw_boundary(self) -> None:
        """Draws a horizontal red line to indicate the cut off point."""
        vertical_position = self._piece_size * self._cell_size
        self._img[
            vertical_position
            - int(self._cell_size / 40) : vertical_position
            + int(self._cell_size / 40)
            + 1,
            400:,
            :,
        ] = self._get_bgr_code("red")

    def _get_grid(self) -> np.ndarray:
        """
        Gets the array of the current grid containing the colour tuples.

        :return: the array of the current grid.
        """
        grid = [
            [self._grid_colours[self._colour_grid[j][i]] for j in range(self._width)]
            for i in range(self._height)
        ]
        return np.array(grid)

    def _resize_grid(self, grid: np.ndarray) -> None:
        """
        Reshape the grid, convert it to an Image and resize it, then convert it
        to an array.

        :param grid: the grid to be resized.
        """
        self._img = grid.reshape((self._height, self._width, 3)).astype(np.uint8)
        self._img = Image.fromarray(self._img, "RGB")
        self._img = self._img.resize(
            (self._width * self._cell_size, self._height * self._cell_size)
        )
        self._img = np.array(self._img)

    def _draw_separating_lines(self) -> None:
        """
        Draws the horizontal and vertical _black lines to separate the grid's cells.
        """
        for j in range(-int(self._cell_size / 40), int(self._cell_size / 40) + 1):
            self._img[
                [i * self._cell_size + j for i in range(self._height)], :, :
            ] = self._black
            self._img[
                :, [i * self._cell_size + j for i in range(self._width)], :
            ] = self._black

    def _add_img_left(self) -> None:
        """
        Adds the image that will appear to the left of the grid.
        """
        img_array = np.zeros((self._height * self._cell_size, 400, 3)).astype(np.uint8)
        mean_score = (
            0.0 if len(self._final_scores) == 0 else np.mean(self._final_scores)
        )

        self._add_statistics(
            img_array=img_array,
            items=[
                [
                    "Height",
                    "Width",
                    "",
                    "Current score",
                    "Mean score",
                    "",
                    "Current piece",
                ],
                [
                    f"{self._height}",
                    f"{self._width}",
                    "",
                    f"{self._score}",
                    f"{mean_score:.1f}",
                    "",
                    f"{self._piece._name}",
                ],
            ],
            x_offsets=[50, 300],
        )
        self._img = np.concatenate((img_array, self._img), axis=1)

    def _add_statistics(
        self, img_array: np.ndarray, items: List[List[str]], x_offsets: List[int]
    ) -> None:
        """
        Adds statistics to the array provided.

        :param img_array: the array to be edited.
        :param items: the lists to be added to the array.
        :param x_offsets: the horizontal positions where the statistics should be added.
        """
        for i, item in enumerate(items):
            for count, j in enumerate(item):
                cv.putText(
                    img_array,
                    j,
                    (x_offsets[i], 60 * (count + 1)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    self._white,
                    2,
                    cv.LINE_AA,
                )

    def _update_coords_and_anchor(self) -> None:
        """Updates the current piece, and resets the anchor."""
        self._piece = self._pieces[self._generate_id_randomly()]
        self._anchor = [self._width / 2 - 1, self._piece_size - 1]

    def _is_illegal(self) -> bool:
        """
        Checks if the piece's current position is illegal by looping over each
        of its square blocks.

        Author: Andrean Lay
        Source: https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/engine.py

        :return: whether the piece's current position is illegal.
        """

        # Loop over each of the piece's blocks.
        for i, j in self._piece._coords:
            x_pos, y_pos = self._anchor[0] + i, self._anchor[1] + j

            # Don't check if the move is illegal when the block is too high.
            if y_pos < 0:
                continue

            # Check if the move is illegal. The last condition must come after the previous conditions.
            if (
                x_pos < 0
                or x_pos >= self._width
                or y_pos >= self._height
                or self._grid[x_pos, y_pos] > 0
            ):
                return True

        return False

    def _hard_drop(self) -> None:
        """
        Finds the position to place the piece (the anchor) by hard dropping the current piece.

        Author: Andrean Lay
        Source: https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/engine.py
        """
        while True:
            # Keep going until current piece occupies a full cell, then backtrack once.
            if not self._is_illegal():
                self._anchor[1] += 1
            else:
                self._anchor[1] -= 1
                break

    def _clear_rows(self) -> int:
        """
        Removes blocks from every full row.

        Author: Andrean Lay
        Source: https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/engine.py

        :return: the number of rows cleared.
        """

        can_clear = [
            (self._grid[:, i + self._piece_size] != 0).all()
            for i in range(self._height - self._piece_size)
        ]
        new_grid = np.zeros_like(self._grid)
        new_colour_grid = np.zeros_like(self._colour_grid)
        col = self._height - 1

        self._last_move_info["eliminated_num_blocks"] = 0

        for row_num in range(self._height - 1, self._piece_size - 1, -1):

            if not can_clear[row_num - self._piece_size]:
                new_grid[:, col] = self._grid[:, row_num]
                new_colour_grid[:, col] = self._colour_grid[:, row_num]
                col -= 1
            else:
                self._last_move_info["eliminated_num_blocks"] += self._last_move_info[
                    "rows_added_to"
                ][row_num]

        self._grid = new_grid
        self._colour_grid = new_colour_grid
        num_rows_cleared = sum(can_clear)
        self._last_move_info["num_rows_cleared"] = num_rows_cleared

        return num_rows_cleared

    def _update_grid(self, set_piece: bool) -> None:
        """
        Sets the current piece using the anchor.

        Author: Andrean Lay
        Source: https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/engine.py

        :param set_piece: whether to set the piece.
        """
        self._last_move_info["rows_added_to"] = {
            row_num: 0 for row_num in range(self._height)
        }
        # Loop over each block.
        for i, j in self._piece._coords:
            y_coord = j + self._anchor[1]
            if set_piece:
                self._last_move_info["rows_added_to"][y_coord] += 1
                self._grid[self._anchor[0] + i, self._anchor[1] + j] = 1
                self._colour_grid[self._anchor[0] + i, self._anchor[1] + j] = (
                    self._piece._idx + 1
                )
            else:
                self._grid[self._anchor[0] + i, self._anchor[1] + j] = 0
                self._colour_grid[self._anchor[0] + i, self._anchor[1] + j] = 0

        anchor_height = self._height - self._anchor[1]
        max_y_coord = self._piece._max_y_coord[self._piece._rotation]
        min_y_coord = self._piece._min_y_coord[self._piece._rotation]
        self._last_move_info["landing_height"] = anchor_height - 0.5 * (
            min_y_coord + max_y_coord
        )

    def _get_reward(self) -> Tuple[float, int]:
        """
        Returns the reward, which is the number of rows cleared.

        :return: the reward and the number of rows cleared.
        """
        num_rows_cleared = self._clear_rows()
        return float(num_rows_cleared), num_rows_cleared

    def _get_all_available_actions(self) -> None:
        """Gets the actions available for each of the pieces in use."""
        self._all_available_actions = {}
        for idx, piece in self._pieces.items():
            self._piece = piece
            self._all_available_actions[idx] = self._compute_available_actions()

    def _compute_available_actions(self) -> Dict[int, Tuple[int, int]]:
        """
        Computes the actions that are available with the current piece.

        Author: Andrean Lay
        Source: https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/engine.py

        :return: the available actions.
        """

        available_actions = {}
        count = 0

        for rotation in self._piece._all_coords.keys():
            self._rotate_piece(rotation)

            max_x_coord = self._piece._max_x_coord[rotation]
            min_x_coord = self._piece._min_x_coord[rotation]

            for translation in range(abs(min_x_coord), self._width - max_x_coord):

                if count == self._num_actions:
                    return available_actions

                self._anchor = [translation, 0]
                self._hard_drop()

                self._update_grid(True)
                available_actions[count] = (translation, rotation)
                self._update_grid(False)

                count += 1

        return available_actions

    def _get_dellacherie_scores(self) -> np.array:
        """
        Gets the Dellacherie feature values.

        :return: a list of the Dellacherie feature values.
        """
        weights = np.array([-1, 1, -1, -1, -4, -1], dtype="double")
        dellacherie_scores = np.empty((self._num_actions), dtype="double")

        for action, (translation, rotation) in self._all_available_actions[
            self._piece._idx
        ].items():
            old_grid = deepcopy(self._grid)
            old_anchor = deepcopy(self._anchor)

            self._rotate_piece(rotation)

            self._anchor = [translation, 0]

            self._hard_drop()
            self._update_grid(True)

            scores = np.empty((6), dtype="double")
            for count, feature_func in enumerate(self._get_dellacherie_funcs()):
                scores[count] = feature_func()

            dellacherie_scores[action] = np.dot(scores, weights)

            self._update_grid(False)
            self._anchor = deepcopy(old_anchor)
            self._grid = deepcopy(old_grid)

        return dellacherie_scores

    def _get_dellacherie_funcs(self) -> list:
        """
        Gets the Dellacherie feature functions.

        :return: a list of the Dellacherie feature functions.
        """
        return [
            self._get_landing_height,
            self._get_eroded_cells,
            self._get_row_transitions,
            self._get_column_transitions,
            self._get_holes,
            self._get_cumulative_wells,
        ]

    def _get_landing_height(self) -> int:
        """
        Landing height = the midpoint of the last piece to be placed.

        :return: landing height.
        """
        if "landing_height" in self._last_move_info:
            return self._last_move_info["landing_height"]
        return 0

    def _get_eroded_cells(self) -> int:
        """
        Num. eroded cells = # rows cleared * # blocks removed that were
        added to the grid by the last action.

        :return: eroded cells.
        """
        if "num_rows_cleared" in self._last_move_info:
            return (
                self._last_move_info["num_rows_cleared"]
                * self._last_move_info["eliminated_num_blocks"]
            )
        return 0

    def _get_row_transitions(self) -> float:
        """
        Row transitions = # transitions from empty to full (or vice versa), examining
        each row one at a time.

        Source: https://github.com/Benjscho/gym-mdptetris/blob/main/gym_mdptetris/envs/feature_functions.py

        :return: row transitions.
        """
        # A full column should be added either side.
        grid = np.ones((self._width + 2, self._height), dtype="bool")
        grid[1:-1, :] = self._grid
        return np.diff(grid.T).sum()

    def _get_column_transitions(self) -> float:
        """
        Column transitions = # transitions from empty to full (or vice versa), examining
        each column one at a time.

        Source: https://github.com/Benjscho/gym-mdptetris/blob/main/gym_mdptetris/envs/feature_functions.py

        :return: column transitions.
        """
        # A full row should be added to the bottom.
        grid = np.ones((self._width, self._height + 1), dtype="bool")
        grid[:, :-1] = self._grid
        return np.diff(grid).sum()

    def _get_holes(self) -> int:
        """
        Gets the number of holes present in the current grid.

        Author: andreanlay
        Source: https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/engine.py

        :return: holes.
        """
        holes = 0

        for col in zip(*self._grid.T):
            # Find the first full cell, starting from the top row below the cut off.
            row = self._piece_size
            while row < self._height and col[row] == 0:
                row += 1

            # Count the number of empty cells below the first full cell.
            holes += len([x for x in col[row + 1 :] if x == 0])

        return holes

    def _get_cumulative_wells(self) -> int:
        """
        Cumulative wells is defined here: https://arxiv.org/abs/1905.01652.

        For each well, find the depth of the well, d(w), then calculate the sum from i=1 to d(w) of i. Lastly, sum the well sums.

        A block is part of a well if the cells directly on either side are full, and the block can be reached from above (there are no full cells directly above it).

        :return: cumulative wells.
        """
        cumulative_wells = 0

        new_grid = np.ones((self._width + 2, self._height + 1), dtype="bool")
        new_grid[1:-1, :-1] = self._grid

        for col in range(1, self._width + 1):

            depth = 1
            well_complete = False

            for row in range(self._height):

                cell_mid = new_grid[col][row]
                cell_right = new_grid[col + 1][row]
                cell_left = new_grid[col - 1][row]

                if cell_mid >= 1:
                    well_complete = True

                # Checks either side to see if the cells are occupied.
                if not well_complete and cell_left > 0 and cell_right > 0:
                    cumulative_wells += depth
                    depth += 1

        return cumulative_wells

    def _rotate_piece(self, rotation: int) -> None:
        """Sets the piece's rotation and rotates the current piece."""
        self._piece._rotation = rotation
        self._piece._coords = self._piece._all_coords[self._piece._rotation]
