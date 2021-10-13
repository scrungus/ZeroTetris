import unittest

import numpy as np

from gym_simplifiedtetris.envs import SimplifiedTetrisEngine as Engine


class SimplifiedTetrisEngineMonominoesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._height_ = 20
        self._width_ = 10
        self._piece_size_ = 1
        self._num_actions_, self._num_pieces_ = (self._width_ - 1, 1)

        self._engine = Engine(
            grid_dims=(self._height_, self._width_),
            piece_size=self._piece_size_,
            num_pieces=self._num_pieces_,
            num_actions=self._num_actions_,
        )
        self._engine._reset()

    def tearDown(self) -> None:
        del self._engine

    def test__get_bgr_code_orange(self) -> None:
        bgr_code_orange = self._engine._get_bgr_code("orange")

        self.assertEqual(bgr_code_orange, (0.0, 165.0, 255.0))

    def test__get_bgr_code_coral(self) -> None:
        bgr_code_coral = self._engine._get_bgr_code("coral")

        self.assertEqual(bgr_code_coral, (80.0, 127.0, 255.0))

    def test__is_illegal_piece_off_top(self) -> None:
        self._engine._piece = [(0, 0)]
        # Top left off the top.
        self._engine._anchor = [0, -1]

        self.assertEqual(self._engine._is_illegal(), False)

    def test__is_illegal_piece_off_bottom(self) -> None:
        self._engine._piece = [(0, 0)]
        # Bottom right off the bottom
        self._engine._anchor = [self._engine._width - 1, self._engine._height]

        self.assertEqual(self._engine._is_illegal(), True)

    def test__is_illegal_piece_off_right(self) -> None:
        self._engine._piece = [(0, 0)]
        self._engine._anchor = [self._engine._width, 0]  # Top right off the right.

        self.assertEqual(self._engine._is_illegal(), True)

    def test__is_illegal_piece_off_left(self) -> None:
        self._engine._piece = [(0, 0)]
        self._engine._anchor = [
            -1,
            self._engine._height - 1,
        ]  # Bottom left off the left

        self.assertEqual(self._engine._is_illegal(), True)

    def test__is_illegal_non_empty_grid_overlapping(self) -> None:
        self._engine._piece = [(0, 0)]
        self._engine._anchor = [0, self._engine._height - 1]  # Bottom left.

        self._engine._grid[0, self._engine._height - 1] = 1  # Bottom left

        self.assertEqual(self._engine._is_illegal(), True)

    def test__is_illegal_piece_not_empty_grid(self) -> None:
        self._engine._piece = [(0, 0)]
        self._engine._anchor = [0, self._engine._height - 1]  # Bottom left.

        # Second col from left
        self._engine._grid[1, : self._engine._height - 1] = 1

        self.assertEqual(self._engine._is_illegal(), False)

    def test__hard_drop_empty_grid(self) -> None:
        self._engine._piece = [(0, 0)]
        self._engine._anchor = [0, 0]  # Top left.

        self._engine._hard_drop()

        self.assertEqual(self._engine._anchor, [0, self._engine._height - 1])

    def test__hard_drop_non_empty_grid(self) -> None:
        self._engine._piece = [(0, 0)]
        self._engine._anchor = [0, 0]  # Top left.

        self._engine._grid[0, self._engine._height - 1] = 1  # Bottom left.

        self._engine._hard_drop()

        self.assertEqual(self._engine._anchor, [0, self._engine._height - 2])

    def test__clear_rows_output_with_empty_grid(self) -> None:
        self.assertEqual(self._engine._clear_rows(), 0)

    def test__clear_rows_empty_grid_after(self) -> None:
        self._engine._clear_rows()
        grid_after = np.zeros((self._engine._width, self._engine._height), dtype="bool")

        np.testing.assert_array_equal(self._engine._grid, grid_after)

    def test__clear_rows_output_one_full_row(self) -> None:
        self._engine._grid[:, self._engine._height - 1 :] = 1

        self.assertEqual(self._engine._clear_rows(), 1)

    def test__clear_rows_one_full_row_grid_after(self) -> None:
        self._engine._grid[:, self._engine._height - 1 :] = 1
        self._engine._clear_rows()
        grid_after = np.zeros((self._engine._width, self._engine._height), dtype="bool")

        np.testing.assert_array_equal(self._engine._grid, grid_after)

    def test__clear_rows_output_two_full_rows(self) -> None:
        self._engine._grid[:, self._engine._height - 2 :] = 1

        self.assertEqual(self._engine._clear_rows(), 2)

    def test__clear_rows_two_full_rows_grid_after(self) -> None:
        self._engine._grid[:, self._engine._height - 2 :] = 1
        self._engine._clear_rows()
        grid_after = np.zeros((self._engine._width, self._engine._height), dtype="bool")

        np.testing.assert_array_equal(self._engine._grid, grid_after)

    def test__clear_rows_output_two_full_rows_full_cell_above(self) -> None:
        self._engine._grid[:, self._engine._height - 2 :] = 1
        self._engine._grid[3, self._engine._height - 3] = 1

        self.assertEqual(self._engine._clear_rows(), 2)

    def test__clear_rows_two_full_rows_full_cell_above_grid_after(self) -> None:
        self._engine._grid[:, self._engine._height - 2 :] = 1
        self._engine._grid[3, self._engine._height - 3] = 1

        self._engine._clear_rows()

        grid_after = np.zeros((self._engine._width, self._engine._height), dtype="bool")
        grid_after[3, self._engine._height - 1] = 1

        np.testing.assert_array_equal(self._engine._grid, grid_after)

    def test__update_grid_true(self) -> None:
        self._engine._piece = [(0, 0)]
        self._engine._anchor = [0, self._engine._height - 1]  # Bottom left.
        self._engine._current_piece_id = 0

        grid_to_compare = np.zeros(
            (self._engine._width, self._engine._height), dtype="bool"
        )
        grid_to_compare[0, self._engine._height - 1 :] = 1

        self._engine._update_grid(True)

        np.testing.assert_array_equal(self._engine._grid, grid_to_compare)

    def test__update_grid_false(self) -> None:
        self._engine._piece = [(0, 0)]
        self._engine._anchor = [0, self._engine._height - 1]  # Bottom left.
        self._engine._current_piece_id = 0

        grid_to_compare = np.zeros(
            (self._engine._width, self._engine._height), dtype="bool"
        )

        self._engine._update_grid(False)
        np.testing.assert_array_equal(self._engine._grid, grid_to_compare)

    def test__update_grid_populated(self) -> None:
        self._engine._piece = [(0, 0)]

        # Add a full row on the bottom row.
        self._engine._grid[0, self._engine._height - 1 :] = 1
        self._engine._anchor = [0, self._engine._height - 1]  # Bottom left.
        self._engine._current_piece_id = 0

        grid_to_compare = np.zeros(
            (self._engine._width, self._engine._height), dtype="bool"
        )
        self._engine._update_grid(False)

        np.testing.assert_array_equal(self._engine._grid, grid_to_compare)

    def test__compute_available_actions(self) -> None:
        self._engine._current_piece_info = {
                "coords": {0: [(0, 0)]},
                "max_y_coord": {0: 0},
                "min_y_coord": {0: 0},
                "max_x_coord": {0: 0},
                "min_x_coord": {0: 0},
                "name": "O",
            }
        available_actions = self._engine._compute_available_actions()
        values = [(j, 0) for j in range(self._engine._width - 1)]
        dict_to_compare = {i: values[i] for i in range(self._engine._num_actions)}

        self.assertDictEqual(available_actions, dict_to_compare)

    def test__get_all_available_actions(self) -> None:
        self._engine._get_all_available_actions()

        for _, value in self._engine._all_available_actions.items():
            self.assertEqual(self._engine._num_actions, len(value))

    def test__get_dellacherie_funcs(self) -> None:
        """
        0000000000
        0000000000
        0000000000
        0000000000
        0000000000
        0000000000
        0000000000
        0000000000
        0000000000
        0000000000
        0000000000
        0000000000
        0000000000
        0000000000
        1000000001
        1011111111
        1011111111
        1011111011
        1011111110
        1111111101
        """
        self._engine._grid[:, -5:] = True
        self._engine._grid[
            1:2, self._engine._height - 5 : self._engine._height - 1
        ] = False
        self._engine._grid[self._engine._width - 1, self._engine._height - 2] = False
        self._engine._grid[self._engine._width - 2, self._engine._height - 1] = False
        self._engine._grid[self._engine._width - 3, self._engine._height - 3] = False
        self._engine._grid[self._engine._width - 1, self._engine._height - 6] = True

        self._engine._piece = [(0, 0)]
        self._engine._current_piece_id = 0
        self._engine._anchor = [0, 0]

        self._engine._hard_drop()
        self._engine._update_grid(True)
        self._engine._clear_rows()

        array_to_compare = np.array(
            [fn() for fn in self._engine._get_dellacherie_funcs()]
        )
        np.testing.assert_array_equal(
            array_to_compare, np.array([6, 0, 44, 16, 3, 10], dtype="double")
        )

    def test__get_landing_height_I_piece(self) -> None:
        self._engine._piece = [(0, 0)]
        self._engine._anchor = [0, self._engine._height - 1]  # Bottom left.
        self._engine._current_piece_id = 0

        self._engine._update_grid(True)

        self.assertEqual(self._engine._get_landing_height(), 1)

    def test__get_landing_height_populated_grid(self) -> None:
        self._engine._piece = [(0, 0)]

        # Add a full row on the bottom row.
        self._engine._grid[:, -1:] = 1
        self._engine._current_piece_id = 0

        # Set the piece above the full row.
        self._engine._anchor = [0, self._engine._height - 2]
        self._engine._update_grid(True)

        self.assertEqual(self._engine._get_landing_height(), 2)

    def test__get_eroded_cells_empty(self) -> None:
        self.assertEqual(self._engine._get_eroded_cells(), 0)

    def test__get_eroded_cells_single(self) -> None:
        # Add a full row on the bottom row.
        self._engine._grid[:, self._engine._height - 1 :] = True
        # Remove the block on the left.
        self._engine._grid[0, self._engine._height - 1] = False

        self._engine._piece = [(0, 0)]
        self._engine._current_piece_id = 0
        self._engine._anchor = [0, 0]

        self._engine._hard_drop()
        self._engine._update_grid(True)
        self._engine._clear_rows()

        self.assertEqual(self._engine._get_eroded_cells(), 1)

    def test__get_eroded_cells_double(self) -> None:
        # Add two full rows on the bottom.
        self._engine._grid[:, self._engine._height - 2 :] = True
        # Remove the two blocks on the left.
        self._engine._grid[0, self._engine._height - 2 :] = False

        self._engine._piece = [(0, 0)]
        self._engine._current_piece_id = 0
        self._engine._anchor = [0, 0]

        self._engine._hard_drop()
        self._engine._update_grid(True)
        self._engine._clear_rows()

        self.assertEqual(self._engine._get_eroded_cells(), 1)

    def test__get_row_transitions_empty(self) -> None:
        self.assertEqual(self._engine._get_row_transitions(), 40)

    def test__get_row_transitions_populated(self) -> None:
        self._engine._grid[:, -2:] = True
        self._engine._grid[0, self._engine._height - 1] = False
        self._engine._grid[2, self._engine._height - 1] = False
        self._engine._grid[1, self._engine._height - 2] = False

        self.assertEqual(self._engine._get_row_transitions(), 42)

    def test__get_row_transitions_populated_more_row_transitions(self) -> None:
        self._engine._grid[:, -2:] = True
        self._engine._grid[0, self._engine._height - 2 :] = False
        self._engine._grid[2, self._engine._height - 2 :] = False
        self._engine._grid[4, self._engine._height - 1] = False

        np.testing.assert_array_equal(self._engine._get_row_transitions(), 46)

    def test__get_column_transitions_empty(self) -> None:
        self.assertEqual(self._engine._get_column_transitions(), 10)

    def test__get_column_transitions_populated(self) -> None:
        self._engine._grid[:, -2:] = True
        self._engine._grid[0, self._engine._height - 1] = False
        self._engine._grid[2, self._engine._height - 1] = False
        self._engine._grid[1, self._engine._height - 2] = False

        self.assertEqual(self._engine._get_column_transitions(), 14)

    def test__get_column_transitions_populated_less_column_transitions(self) -> None:
        self._engine._grid[:, -2:] = True
        self._engine._grid[0, self._engine._height - 2 :] = False
        self._engine._grid[2, self._engine._height - 2 :] = False
        self._engine._grid[4, self._engine._height - 1] = False

        np.testing.assert_array_equal(self._engine._get_column_transitions(), 12)

    def test__get_holes_empty(self) -> None:
        self.assertEqual(self._engine._get_holes(), 0)

    def test__get_holes_populated_two_holes(self) -> None:
        self._engine._grid[:, -2:] = True
        self._engine._grid[0, self._engine._height - 1] = False
        self._engine._grid[2, self._engine._height - 1] = False

        self.assertEqual(self._engine._get_holes(), 2)

    def test__get_holes_populated_no_holes(self) -> None:
        self._engine._grid[:, -2:] = True
        self._engine._grid[0, self._engine._height - 2 :] = False
        self.assertEqual(self._engine._get_holes(), 0)

    def test__get_holes_populated_one_hole(self) -> None:
        self._engine._grid[:, -2:] = True
        self._engine._grid[0, self._engine._height - 2 :] = False
        self._engine._grid[2, self._engine._height - 2 :] = False
        self._engine._grid[4, self._engine._height - 1] = False

        np.testing.assert_array_equal(self._engine._get_holes(), 1)

    def test__get_cumulative_wells_empty(self) -> None:
        np.testing.assert_array_equal(self._engine._get_cumulative_wells(), 0)

    def test__get_cumulative_wells_populated(self) -> None:
        self._engine._grid[:, -2:] = True
        self._engine._grid[0, self._engine._height - 2 :] = False

        np.testing.assert_array_equal(self._engine._get_cumulative_wells(), 3)

    def test__get_cumulative_wells_populated_deeper_well(self) -> None:
        self._engine._grid[:, -2:] = True
        self._engine._grid[0, self._engine._height - 2 :] = False
        self._engine._grid[2, self._engine._height - 2 :] = False
        self._engine._grid[4, self._engine._height - 1] = False

        np.testing.assert_array_equal(self._engine._get_cumulative_wells(), 6)


if __name__ == "__main__":
    unittest.main()
