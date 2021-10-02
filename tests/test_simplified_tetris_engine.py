import unittest

import numpy as np

from gym_simplifiedtetris.envs import SimplifiedTetrisEngine as Engine


class SimplifiedTetrisEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._height_ = 20
        self._width_ = 10
        self._piece_size_ = 4
        self._num_actions_, self._num_pieces_ = (4 * self._width_ - 6, 7)

        self._engine = Engine(
            grid_dims=(self._height_, self._width_),
            piece_size=self._piece_size_,
            num_pieces=self._num_pieces_,
            num_actions=self._num_actions_,
        )
        self._engine._reset()

    def tearDown(self) -> None:
        del self._engine

    def test__get_bgr_code(self) -> None:
        bgr_code_orange = self._engine._get_bgr_code("orange")
        self.assertEqual(bgr_code_orange, (0.0, 165.0, 255.0))
        bgr_code_coral = self._engine._get_bgr_code("coral")
        self.assertEqual(bgr_code_coral, (80.0, 127.0, 255.0))
        bgr_code_orangered = self._engine._get_bgr_code("orangered")
        self.assertEqual(bgr_code_orangered, (0.0, 69.0, 255.0))

    def test__is_illegal(self) -> None:
        # Piece off the top.
        # 'I' piece vertical.
        self._engine._piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self._engine._anchor = [0, 0]  # Top left.
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self.assertEqual(self._engine._is_illegal(), False)

        # Piece off the bottom.
        # 'L' piece rotated 90 CW.
        self._engine._piece = [(0, 0), (0, 1), (1, 0), (2, 0)]
        # Bottom right.
        self._engine._anchor = [
            self._engine._width - 1, self._engine._height - 1]
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self.assertEqual(self._engine._is_illegal(), True)

        # Piece off the right.
        # 'I' piece horizontal.
        self._engine._piece = [(0, 0), (1, 0), (2, 0), (3, 0)]
        self._engine._anchor = [self._engine._width - 1, 0]  # Top right.
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self.assertEqual(self._engine._is_illegal(), True)

        # Piece off the left.
        self._engine._piece = [(0, 0), (-1, 0), (0, 1),
                               (0, 2)]  # 'L' piece rotated 180
        self._engine._anchor = [0, self._engine._height - 1]  # Bottom left.
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self.assertEqual(self._engine._is_illegal(), True)

        # Non-empty grid and overlapping.
        # 'I' piece vertical.
        self._engine._piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self._engine._anchor = [0, self._engine._height - 1]  # Bottom left.
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self._engine._grid[0, self._engine._height - 1] = 1  # Bottom left
        self.assertEqual(self._engine._is_illegal(), True)

        # Non-empty grid and not overlapping.
        # 'I' piece vertical.
        self._engine._piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self._engine._anchor = [0, self._engine._height - 1]  # Bottom left.
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        # Second col from left
        self._engine._grid[1, : self._engine._height - 1] = 1
        self.assertEqual(self._engine._is_illegal(), False)

    def test__hard_drop(self) -> None:
        # Empty grid.
        # 'I' piece vertical.
        self._engine._piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self._engine._anchor = [0, 0]  # Top left.
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self._engine._hard_drop()
        self.assertEqual(self._engine._anchor, [0, self._engine._height - 1])

        # Non-empty grid.
        # 'I' piece vertical.
        self._engine._piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self._engine._anchor = [0, 0]  # Top left.
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self._engine._grid[0, self._engine._height - 1] = 1  # Bottom left.
        self._engine._hard_drop()
        self.assertEqual(self._engine._anchor, [0, self._engine._height - 2])

    def test__clear_rows(self) -> None:
        # Full grid.
        self._engine._grid = np.ones(
            (self._engine._width, self._engine._height), dtype=int
        )
        self._engine._grid[:, : self._engine._piece_size - 1] = 0
        self.assertEqual(
            self._engine._clear_rows(), self._engine._height - self._engine._piece_size
        )
        grid_after = np.zeros(
            (self._engine._width, self._engine._height), dtype=int)
        np.testing.assert_array_equal(self._engine._grid, grid_after)

        # Empty grid.
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self.assertEqual(self._engine._clear_rows(), 0)
        grid_after = np.zeros(
            (self._engine._width, self._engine._height), dtype=int)
        np.testing.assert_array_equal(self._engine._grid, grid_after)

        # One row full with no full cells above.
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self._engine._grid[:, self._engine._height - 1:] = 1
        self.assertEqual(self._engine._clear_rows(), 1)
        grid_after = np.zeros(
            (self._engine._width, self._engine._height), dtype=int)
        np.testing.assert_array_equal(self._engine._grid, grid_after)

        # Two rows full with no full cells above.
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self._engine._grid[:, self._engine._height - 2:] = 1
        self.assertEqual(self._engine._clear_rows(), 2)
        grid_after = np.zeros(
            (self._engine._width, self._engine._height), dtype=int)
        np.testing.assert_array_equal(self._engine._grid, grid_after)

        # Two rows full with a full cell above.
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self._engine._grid[:, self._engine._height - 2:] = 1
        self._engine._grid[3, self._engine._height - 3] = 1
        self.assertEqual(self._engine._clear_rows(), 2)
        grid_after = np.zeros(
            (self._engine._width, self._engine._height), dtype=int)
        grid_after[3, self._engine._height - 1] = 1
        np.testing.assert_array_equal(self._engine._grid, grid_after)

        # Two rows full with two full cells above.
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self._engine._grid[:, self._engine._height - 2:] = 1
        self._engine._grid[3, self._engine._height - 3] = 1
        self._engine._grid[4, self._engine._height - 4] = 1
        self.assertEqual(self._engine._clear_rows(), 2)
        grid_after = np.zeros(
            (self._engine._width, self._engine._height), dtype=int)
        grid_after[3, self._engine._height - 1] = 1
        grid_after[4, self._engine._height - 2] = 1
        np.testing.assert_array_equal(self._engine._grid, grid_after)

    def test__update_grid(self) -> None:
        # Set piece
        # 'I' piece vertical.
        self._engine._piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self._engine._anchor = [0, self._engine._height - 1]  # Bottom left.
        self._engine._current_piece_id = 1
        grid_to_compare = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        grid_to_compare[0, self._engine._height - 4:] = 1
        self._engine._update_grid(True)
        np.testing.assert_array_equal(self._engine._grid, grid_to_compare)

        # Undo when grid is empty.
        # 'I' piece vertical.
        self._engine._piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self._engine._anchor = [0, self._engine._height - 1]  # Bottom left.
        self._engine._current_piece_id = 1
        grid_to_compare = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self._engine._update_grid(False)
        np.testing.assert_array_equal(self._engine._grid, grid_to_compare)

        # Undo when grid is not empty.
        # 'I' piece vertical.
        self._engine._piece = [(0, 0), (0, -1), (0, -2), (0, -3)]
        self._engine._grid = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self._engine._grid[0, self._engine._height - 4:] = 1
        self._engine._anchor = [0, self._engine._height - 1]  # Bottom left.
        self._engine._current_piece_id = 1
        grid_to_compare = np.zeros(
            (self._engine._width, self._engine._height), dtype=int
        )
        self._engine._update_grid(False)
        np.testing.assert_array_equal(self._engine._grid, grid_to_compare)

    def test__compute_available_actions(self) -> None:
        self._engine._current_piece_coords = [
            [(0, 0), (0, -1), (-1, 0), (-1, -1)],
            [(0, 0), (0, -1), (-1, 0), (-1, -1)],
            [(0, 0), (0, -1), (-1, 0), (-1, -1)],
            [(0, 0), (0, -1), (-1, 0), (-1, -1)],
        ]  # 'O'.
        available_actions = self._engine._compute_available_actions()
        values = [(j, i) for i in range(4)
                  for j in range(1, self._engine._width)]
        dict_to_compare = {i: values[i]
                           for i in range(self._engine._num_actions)}
        self.assertDictEqual(available_actions, dict_to_compare)

    def test__get_all_available_actions(self) -> None:
        self._engine._get_all_available_actions()
        for _, value in self._engine._all_available_actions.items():
            self.assertEqual(self._engine._num_actions, len(value))


if __name__ == "__main__":
    unittest.main()
