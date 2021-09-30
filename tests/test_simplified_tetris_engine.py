import unittest

import numpy as np

from gym_simplifiedtetris.envs import SimplifiedTetrisEngine as Engine


class SimplifiedTetrisEngineTest(unittest.TestCase):

    def setUp(self) -> None:
        self.grid_height = 20
        self.grid_width = 10
        self.piece_size = 4

        self.num_actions, self.num_pieces = {
            1: (self.grid_width, 1),
            2: (2 * self.grid_width - 1, 1),
            3: (4 * self.grid_width - 4, 2),
            4: (4 * self.grid_width - 6, 7)
        }[self.piece_size]
        self.engine = Engine(
            grid_dims=(self.grid_height, self.grid_width),
            piece_size=self.piece_size,
            num_pieces=self.num_pieces,
            num_actions=self.num_actions,
        )
        self.engine.reset()

    def tearDown(self) -> None:
        del self.engine

    def test_get_bgr_code(self):
        bgr_code_orange = self.engine.get_bgr_code('orange')
        self.assertEqual(bgr_code_orange, (0.0, 165.0, 255.0))
        bgr_code_coral = self.engine.get_bgr_code('coral')
        self.assertEqual(bgr_code_coral, (80.0, 127.0, 255.0))
        bgr_code_orangered = self.engine.get_bgr_code('orangered')
        self.assertEqual(bgr_code_orangered, (0.0, 69.0, 255.0))

    def test_is_illegal(self):
        # Piece off the top.
        # 'I' piece vertical.
        self.engine.piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self.engine.anchor = [0, 0]  # Top left.
        self.engine.grid = np.zeros(
            (self.grid_width, self.grid_height), dtype=int)
        """print(self.engine.grid.T)
        self.engine.hard_drop()
        print(self.engine.anchor)
        self.engine.update_grid(True)
        print(self.engine.grid.T)
        self.engine.update_grid(False)"""
        self.assertEqual(self.engine.is_illegal(), False)

        # Piece off the bottom.
        # 'L' piece rotated 90 CW.
        self.engine.piece = [(0, 0), (0, 1), (1, 0), (2, 0)]
        # Bottom right.
        self.engine.anchor = [self.engine.width - 1, self.engine.height - 1]
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.assertEqual(self.engine.is_illegal(), True)

        # Piece off the right.
        # 'I' piece horizontal.
        self.engine.piece = [(0, 0), (1, 0), (2, 0), (3, 0)]
        self.engine.anchor = [self.engine.width - 1, 0]  # Top right.
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.assertEqual(self.engine.is_illegal(), True)

        # Piece off the left.
        self.engine.piece = [(0, 0), (-1, 0), (0, 1),
                             (0, 2)]  # 'L' piece rotated 180
        self.engine.anchor = [0, self.engine.height - 1]  # Bottom left.
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.assertEqual(self.engine.is_illegal(), True)

        # Non-empty grid and overlapping.
        # 'I' piece vertical.
        self.engine.piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self.engine.anchor = [0, self.grid_height - 1]  # Bottom left.
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.engine.grid[0, self.grid_height - 1] = 1  # Bottom left
        self.assertEqual(self.engine.is_illegal(), True)

        # Non-empty grid and not overlapping.
        # 'I' piece vertical.
        self.engine.piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self.engine.anchor = [0, self.grid_height - 1]  # Bottom left.
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.engine.grid[1, :self.grid_height - 1] = 1  # Second col from left
        self.assertEqual(self.engine.is_illegal(), False)

    def test_hard_drop(self):
        # Empty grid.
        # 'I' piece vertical.
        self.engine.piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self.engine.anchor = [0, 0]  # Top left.
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.engine.hard_drop()
        self.assertEqual(self.engine.anchor, [0, self.engine.height - 1])

        # Non-empty grid.
        # 'I' piece vertical.
        self.engine.piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self.engine.anchor = [0, 0]  # Top left.
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.engine.grid[0, self.engine.height - 1] = 1  # Bottom left.
        self.engine.hard_drop()
        self.assertEqual(self.engine.anchor, [0, self.engine.height - 2])

    def test_clear_rows(self):
        # Full grid.
        self.engine.grid = np.ones(
            (self.engine.width, self.engine.height), dtype=int)
        self.engine.grid[:, :self.engine.piece_size - 1] = 0
        self.assertEqual(self.engine.clear_rows(),
                         self.engine.height - self.engine.piece_size)
        grid_after = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        np.testing.assert_array_equal(self.engine.grid, grid_after)

        # Empty grid.
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.assertEqual(self.engine.clear_rows(), 0)
        grid_after = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        np.testing.assert_array_equal(self.engine.grid, grid_after)

        # One row full with no full cells above.
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.engine.grid[:, self.engine.height - 1:] = 1
        self.assertEqual(self.engine.clear_rows(), 1)
        grid_after = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        np.testing.assert_array_equal(self.engine.grid, grid_after)

        # Two rows full with no full cells above.
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.engine.grid[:, self.engine.height - 2:] = 1
        self.assertEqual(self.engine.clear_rows(), 2)
        grid_after = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        np.testing.assert_array_equal(self.engine.grid, grid_after)

        # Two rows full with a full cell above.
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.engine.grid[:, self.engine.height - 2:] = 1
        self.engine.grid[3, self.engine.height - 3] = 1
        self.assertEqual(self.engine.clear_rows(), 2)
        grid_after = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        grid_after[3, self.engine.height - 1] = 1
        np.testing.assert_array_equal(self.engine.grid, grid_after)

        # Two rows full with two full cells above.
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.engine.grid[:, self.engine.height - 2:] = 1
        self.engine.grid[3, self.engine.height - 3] = 1
        self.engine.grid[4, self.engine.height - 4] = 1
        self.assertEqual(self.engine.clear_rows(), 2)
        grid_after = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        grid_after[3, self.engine.height - 1] = 1
        grid_after[4, self.engine.height - 2] = 1
        np.testing.assert_array_equal(self.engine.grid, grid_after)

    def test_update_grid(self):
        # Set piece
        # 'I' piece vertical.
        self.engine.piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.engine.anchor = [0, self.engine.height - 1]  # Bottom left.
        self.engine.current_piece_id = 1
        grid_to_compare = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        grid_to_compare[0, self.engine.height - 4:] = 1
        self.engine.update_grid(True)
        np.testing.assert_array_equal(self.engine.grid, grid_to_compare)

        # Undo when grid is empty.
        # 'I' piece vertical.
        self.engine.piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.engine.anchor = [0, self.engine.height - 1]  # Bottom left.
        self.engine.current_piece_id = 1
        grid_to_compare = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.engine.update_grid(False)
        np.testing.assert_array_equal(self.engine.grid, grid_to_compare)

        # Undo when grid is not empty.
        # 'I' piece vertical.
        self.engine.piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self.engine.grid = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.engine.grid[0, self.engine.height - 4:] = 1
        self.engine.anchor = [0, self.engine.height - 1]  # Bottom left.
        self.engine.current_piece_id = 1
        grid_to_compare = np.zeros(
            (self.engine.width, self.engine.height), dtype=int)
        self.engine.update_grid(False)
        np.testing.assert_array_equal(self.engine.grid, grid_to_compare)

    def test_compute_available_actions(self):
        self.engine.current_piece_coords = [
            [(0, 0), (0, -1), (-1, 0), (-1, -1)],
            [(0, 0), (0, -1), (-1, 0), (-1, -1)],
            [(0, 0), (0, -1), (-1, 0), (-1, -1)],
            [(0, 0), (0, -1), (-1, 0), (-1, -1)],
        ]  # 'O'.
        available_actions = self.engine.compute_available_actions()
        values = [(j, i) for i in range(4) for j in range(1, self.engine.width)]
        dict_to_compare = {i: values[i] for i in range(self.num_actions)}
        print(values, dict_to_compare)
        print(available_actions)
        self.assertDictEqual(available_actions, dict_to_compare)

    def test_get_all_available_actions(self):
        self.engine.get_all_available_actions()
        for _, v in self.engine.all_available_actions.items():
            self.assertEqual(self.num_actions, len(v))


if __name__ == '__main__':
    unittest.main()
