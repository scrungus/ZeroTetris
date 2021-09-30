import time
from typing import Dict, Tuple

import cv2.cv2 as cv
import imageio
import numpy as np
from gym_simplifiedtetris.utils.pieces import PieceCoords
from matplotlib import colors
from PIL import Image

PIECES_DICT = {
    1: {
        1: {
            'coords': [
                [(0, 0)],
            ],
            'name': 'O',
        }
    },
    2: {
        1: {
            'coords': [
                [(0, 0), (0, -1)],
                [(0, 0), (1, 0)],
            ],
            'name': 'I',
        }
    },
    3: {
        1: {
            'coords': [
                [(0, 0), (0, -1), (0, -2)],
                [(0, 0), (1, 0), (2, 0)],
                [(0, 0), (0, -1), (0, -2)],
                [(0, 0), (1, 0), (2, 0)],
            ],
            'name': 'I',
        },
        2: {
            'coords': [
                [(0, 0), (1, 0), (0, -1)],
                [(0, 0), (0, 1), (1, 0)],
                [(0, 0), (-1, 0), (0, 1)],
                [(0, 0), (0, -1), (-1, 0)],
            ],
            'name': 'L',
        }
    },
    4: {
        1: {
            'coords': [
                [(0, 0), (0, -1), (0, -2), (0, -3)],
                [(0, 0), (1, 0), (2, 0), (3, 0)],
                [(0, 0), (0, -1), (0, -2), (0, -3)],
                [(0, 0), (1, 0), (2, 0), (3, 0)],
            ],
            'name': 'I',
        },
        2: {
            'coords': [
                [(0, 0), (1, 0), (0, -1), (0, -2)],
                [(0, 0), (0, 1), (1, 0), (2, 0)],
                [(0, 0), (-1, 0), (0, 1), (0, 2)],
                [(0, 0), (0, -1), (-1, 0), (-2, 0)],
            ],
            'name': 'L',
        },
        3: {
            'coords': [
                [(0, 0), (0, -1), (-1, 0), (-1, -1)],
                [(0, 0), (0, -1), (-1, 0), (-1, -1)],
                [(0, 0), (0, -1), (-1, 0), (-1, -1)],
                [(0, 0), (0, -1), (-1, 0), (-1, -1)],
            ],
            'name': 'O',
        },
        4: {
            'coords': [
                [(0, 0), (-1, 0), (1, 0), (0, -1)],
                [(0, 0), (0, -1), (0, 1), (1, 0)],
                [(0, 0), (1, 0), (-1, 0), (0, 1)],
                [(0, 0), (0, 1), (0, -1), (-1, 0)],
            ],
            'name': 'T',
        },
        5: {
            'coords': [
                [(0, 0), (-1, 0), (0, -1), (0, -2)],
                [(0, 0), (0, -1), (1, 0), (2, 0)],
                [(0, 0), (1, 0), (0, 1), (0, 2)],
                [(0, 0), (0, 1), (-1, 0), (-2, 0)],
            ],
            'name': 'J',
        },
        6: {
            'coords': [
                [(0, 0), (-1, 0), (0, -1), (1, -1)],
                [(0, 0), (0, -1), (1, 0), (1, 1)],
                [(0, 0), (-1, 0), (0, -1), (1, -1)],
                [(0, 0), (0, -1), (1, 0), (1, 1)],
            ],
            'name': 'S',
        },
        7: {
            'coords': [
                [(0, 0), (-1, -1), (0, -1), (1, 0)],
                [(0, 0), (1, -1), (1, 0), (0, 1)],
                [(0, 0), (-1, -1), (0, -1), (1, 0)],
                [(0, 0), (1, -1), (1, 0), (0, 1)],
            ],
            'name': 'Z',
        },
    }
}


class SimplifiedTetrisEngine:
    """
    A class representing a simplified Tetris engine containing methods
    that retrieve the actions available for each of the pieces in use, drop
    pieces vertically downwards, having identified the correct location to
    drop them, clear full rows, and render a game of Tetris.

    :param grid_dims: the grid dimensions; height, width.
    :param piece_size: the size of the pieces in use.
    :param num_pieces: the number of pieces in use.
    :param num_actions: the number of available actions in each state.
    """

    def __init__(
            self,
            grid_dims: Tuple[int, int],
            piece_size: int,
            num_pieces: int,
            num_actions: int,
    ):
        self.height, self.width = grid_dims
        self.piece_size = piece_size
        self.num_pieces = num_pieces
        self.num_actions = num_actions

        # Create two empty grids.
        self.grid = np.zeros((grid_dims[1], grid_dims[0]), dtype=int)
        self.old_grid = np.zeros((grid_dims[1], grid_dims[0]), dtype=int)

        # Initialise two anchors.
        self.anchor = [grid_dims[1] / 2 - 1, piece_size - 1]
        self.old_anchor = [grid_dims[1] / 2 - 1, piece_size - 1]

        # Initialise render attributes.
        self.final_scores = np.array([], dtype=int)
        self.sleep_time = 500
        self.show_agent_playing = True
        self.cell_size = int(min(0.8 * 1000 / grid_dims[0],
                                 0.8 * 2000 / grid_dims[1]))
        self.LEFT_SPACE = 400
        self.BLACK: tuple = self.get_bgr_code('black')
        self.WHITE: tuple = self.get_bgr_code('white')
        self.RED: tuple = self.get_bgr_code('red')
        self.GRID_COLOURS: list = [
            self.WHITE,  # Empty.
            self.get_bgr_code('cyan'),  # 'I'.
            self.get_bgr_code('orange'),  # 'L'.
            self.get_bgr_code('yellow'),  #  'O'.
            self.get_bgr_code('purple'),  # 'T'.
            self.get_bgr_code('blue'),  # 'J'.
            self.get_bgr_code('green'),  # 'S'.
            self.RED,  # 'Z'.
        ]

        # Initialise an empty img array.
        self.img = np.array([])

        # Initialise the piece coordinates.
        self.all_piece_coords = PieceCoords(PIECES_DICT[piece_size])
        self.current_piece_coords, self.current_piece_id = \
            self.all_piece_coords.get_piece_at_random()

        self.get_all_available_actions()
        self.reset()

        # Initialise attributes for saving GIFs.
        self.image_lst = []
        self.save_frame = True

    @staticmethod
    def get_bgr_code(colour_name: str) -> Tuple[float, float, float]:
        """
        Gets the BGR code corresponding to the arg provided.

        :param colour_name: a string of the colour name,
        :return: an inverted RGB code of the inputted colour name.
        """
        return tuple(np.array([255, 255, 255]) * colors.to_rgb(colour_name))[::-1]

    @staticmethod
    def close():
        """Closes the open windows."""
        cv.waitKey(1)
        cv.destroyAllWindows()
        cv.waitKey(1)

    def reset(self):
        """Resets the score, grid, piece coords, piece id and anchor."""
        self.score = 0
        self.grid = np.zeros_like(self.grid)
        self.update_coords_and_anchor()

    def render(self, mode: str = 'human') -> np.ndarray:
        """
        Shows an image of the current grid, having dropped the current piece.
        The human has the option to pause (SPACEBAR), speed up (RIGHT key),
        slow down (LEFT key) or quit (ESC) the window.

        :param mode: the render mode.
        :return: the image pixel values.
        """

        # Get the grid after dropping the current piece.
        grid = self.get_grid()

        # Convert grid into Image object then into an array.
        self.resize_grid(grid)

        # Draw horizontal and vertical lines between the cells.
        self.draw_separating_lines()

        # Add extra space to the left.
        img_left = self.get_img_left()
        self.img = np.concatenate((img_left, self.img), axis=1)

        # Draws a horizontal red line to indicate the top of the playfield.
        self.draw_boundary()

        if mode == 'human':
            if self.show_agent_playing:

                if self.save_frame:
                    frame_rgb = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
                    self.image_lst.append(frame_rgb)

                if len(self.final_scores) == 5 and self.save_frame:
                    imageio.mimsave(f'assets/{self.height}x{self.width}_{self.piece_size}.gif',
                                    self.image_lst, fps=60, duration=0.5)
                    self.save_frame = False

                cv.imshow('Simplified Tetris', self.img)
                k = cv.waitKey(self.sleep_time)

                # Escape to exit, spacebar to pause and resume.
                if k == 3:  # right arrow
                    self.sleep_time -= 100

                    if self.sleep_time < 100:
                        self.sleep_time = 1

                    time.sleep(self.sleep_time/1000)
                elif k == 2:  # Left arrow.
                    self.sleep_time += 100
                    time.sleep(self.sleep_time/1000)
                elif k == 27:  # Esc.
                    self.show_agent_playing = False
                    self.close()
                elif k == 32:  # Spacebar.
                    while True:
                        j = cv.waitKey(30)

                        if j == 32:  # Spacebar.
                            break

                        if j == 27:  # Esc.
                            self.show_agent_playing = False
                            self.close()
                            break

            return self.img

        raise ValueError('Mode should be "human".')

    def draw_boundary(self):
        """Draws a horizontal red line to indicate the cut off point."""
        vertical_position = self.piece_size * self.cell_size
        self.img[vertical_position - int(self.cell_size/40): vertical_position
                 + int(self.cell_size/40) + 1, self.LEFT_SPACE:, :] = self.RED

    def get_grid(self) -> np.ndarray:
        """
        Gets the array of the current grid containing the colour tuples.

        :return: the array of the current grid.
        """
        grid = [[self.GRID_COLOURS[self.grid[j][i]] for j in range(
            self.width)] for i in range(self.height)]
        return np.array(grid)

    def resize_grid(self, grid: np.ndarray):
        """
        Reshape the grid, convert it to an Image and resize it, then convert it
        to an array.

        :param grid: the grid to be resized.
        """
        self.img = grid.reshape(
            (self.height, self.width, 3)).astype(np.uint8)
        self.img = Image.fromarray(self.img, 'RGB')
        self.img = self.img.resize((self.width * self.cell_size,
                                    self.height * self.cell_size))
        self.img = np.array(self.img)

    def draw_separating_lines(self):
        """Draws the horizontal and vertical black lines to separate the grid's cells."""
        for j in range(-int(self.cell_size / 40), int(self.cell_size / 40) + 1):
            self.img[[i * self.cell_size +
                      j for i in range(self.height)], :, :] = self.BLACK
            self.img[:, [i * self.cell_size +
                         j for i in range(self.width)], :] = self.BLACK

    def get_img_left(self) -> np.ndarray:
        """
        Gets the image that will appear to the left of the grid.

        :return: the image that will appear to the left of the grid.
        """
        img_array = np.zeros((self.height * self.cell_size,
                              self.LEFT_SPACE, 3)).astype(np.uint8)

        # Calculate the mean score.
        mean_score = 0.0 if len(
            self.final_scores) == 0 else np.mean(self.final_scores)

        # Add statistics.
        self.add_statistics(
            img_array=img_array,
            items=[
                [
                    "Height",
                    "Width",
                    "",
                    "Current score",
                    "Mean score",
                    "",
                    "Current piece"
                ],
                [
                    f"{self.height}",
                    f"{self.width}",
                    '',
                    f"{self.score}",
                    f"{mean_score:.1f}",
                    "",
                    f"{PIECES_DICT[self.piece_size][self.current_piece_id]['name']}"
                ]
            ],
            x_offsets=[50, 300],
        )
        return img_array

    def add_statistics(
            self,
            img_array: np.ndarray,
            items: list,
            x_offsets: list,
    ):
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
                    self.WHITE,
                    2,
                    cv.LINE_AA
                )

    def update_coords_and_anchor(self):
        """Updates the current piece's coords and ID, and resets the anchor."""
        self.current_piece_coords, self.current_piece_id = \
            self.all_piece_coords.get_piece_at_random()
        self.anchor = [self.width / 2 - 1, self.piece_size - 1]

    def is_legal(self) -> bool:
        """
        Checks if the piece's current position is legal by looping over each
        of its square blocks.

        Author:
        > Andrean Lay
        Source:
        > https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/
        engine.py

        :return: whether the piece's current position is legal.
        """

        # Loop over each of the piece's blocks.
        for i, j in self.piece:
            x_pos, y_pos = self.anchor[0] + i, self.anchor[1] + j

            # Doesn't matter if a block is too high.
            if y_pos < 0:
                continue

            cond1 = x_pos < 0
            cond2 = x_pos >= self.width
            cond3 = y_pos >= self.height

            # Check if legal move.
            if cond1 or cond2 or cond3 or self.grid[x_pos, y_pos] > 0:
                return True

        return False

    def hard_drop(self):
        """
        Finds the position to place the piece (the anchor) by hard dropping the current piece.

        Author:
        > Andrean Lay
        Source:
        > https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/
        engine.py
        """
        while True:
            # Keep going until current piece occupies a full cell, then backtrack once.
            if not self.is_legal():
                self.anchor[1] += 1
            else:
                self.anchor[1] -= 1
                break

    def clear_rows(self) -> int:
        """
        Removes blocks from all rows that are full.

        Author:
        > Andrean Lay
        Source:
        > https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/
        engine.py

        :return: the number of rows cleared.
        """

        can_clear = [(self.grid[:, i + self.piece_size] != 0).all()
                     for i in range(self.height - self.piece_size)]
        new_grid = np.zeros_like(self.grid)
        j = self.height - 1

        # Starts from bottom row.
        for i in range(self.height - 1, self.piece_size - 1, -1):

            if not can_clear[i - self.piece_size]:  # Unable to clear.
                new_grid[:, j] = self.grid[:, i]
                j -= 1

        self.grid = new_grid
        return sum(can_clear)

    def update_grid(self, set_piece: bool):
        """
        Sets the current piece using the anchor.

        Author:
        > Andrean Lay
        Source:
        > https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/
        engine.py

        :param set_piece: whether to set the piece.
        """
        # Loop over each block.
        for i, j in self.piece:
            self.grid[int(i + self.anchor[0]), int(j + self.anchor[1])
                      ] = self.current_piece_id if set_piece else 0

    def get_reward(self) -> Tuple[float, int]:
        """
        Returns the reward, which is the number of rows cleared.

        :return: the reward and the number of rows cleared.
        """
        num_rows_cleared = self.clear_rows()
        return float(num_rows_cleared), num_rows_cleared

    def get_all_available_actions(self):
        """Gets the actions available for each of the pieces in use."""
        self.all_available_actions = {}
        for k in range(1, self.num_pieces + 1):
            self.current_piece_coords = self.all_piece_coords.select_piece(k)
            self.current_piece_id = k
            self.all_available_actions[k] = self.compute_available_actions()

    def compute_available_actions(self) -> Dict:
        """
        Computes the actions that are available with the current piece.

        Author:
        > Andrean Lay
        Source:
        > https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/
        engine.py

        :return: the available actions.
        """

        available_actions = {}
        count = 0

        for rotation, piece in enumerate(self.current_piece_coords):
            self.piece = piece.copy()

            max_x = int(max([coord[0] for coord in self.piece]))
            min_x = int(min([coord[0] for coord in self.piece]))

            for translation in range(abs(min_x), self.width - max_x):

                # This ensures that no more than self.num_actions are returned.
                if count == self.num_actions:
                    return available_actions

                self.anchor = [translation, 0]

                self.hard_drop()
                self.update_grid(True)

                # Update available_actions with translation/rotation tuple.
                available_actions[count] = (translation, rotation)

                self.update_grid(False)

                count += 1

        return available_actions
