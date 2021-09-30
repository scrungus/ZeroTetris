import unittest

import numpy as np

from gym_simplifiedtetris.envs import SimplifiedTetrisEngine as Engine


class SimplifiedTetrisEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid_height = 20
        self.grid_width = 10
        self.engine = Engine(
            grid_dims=(self.grid_height, self.grid_width),
            piece_size=4,
            num_pieces=7,
            num_actions=4*self.grid_width-6,
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
        self.engine.piece = [(0, 0), (0, -1), (0, -2), (0, -3)]  # 'I' piece vertical.
        self.engine.anchor = [0, 0]  # Top left.
        self.engine.grid = np.zeros((self.grid_width, self.grid_height), dtype=int)
        """print(self.engine.grid.T)
        self.engine.hard_drop()
        print(self.engine.anchor)
        self.engine.update_grid(True)
        print(self.engine.grid.T)
        self.engine.update_grid(False)"""
        self.assertEqual(self.engine.is_illegal(), False)

        # Piece off the bottom.
        self.engine.piece = [(0, 0), (0, 1), (1, 0), (2, 0)]  # 'L' piece rotated 90 CW.
        self.engine.anchor = [self.engine.width - 1, self.engine.height - 1]  # Bottom right.
        self.engine.grid = np.zeros((self.engine.width, self.engine.height), dtype=int)
        self.assertEqual(self.engine.is_illegal(), True)

        # Piece off the right.
        self.engine.piece = [(0, 0), (1, 0), (2, 0), (3, 0)]  # 'I' piece horizontal.
        self.engine.anchor = [self.engine.width - 1, 0]  # Top right.
        self.engine.grid = np.zeros((self.engine.width, self.engine.height), dtype=int)
        self.assertEqual(self.engine.is_illegal(), True)

        # Piece off the left.
        self.engine.piece = [(0, 0), (-1, 0), (0, 1), (0, 2)]  # 'L' piece rotated 180
        self.engine.anchor = [0, self.engine.height - 1]  # Bottom left.
        self.engine.grid = np.zeros((self.engine.width, self.engine.height), dtype=int)
        self.assertEqual(self.engine.is_illegal(), True)

        # Non-empty grid and overlapping.
        self.engine.piece = [(0, 0), (0, -1), (0, -2), (0, -3)]  # 'I' piece vertical.
        self.engine.anchor = [0, self.grid_height - 1]  # Bottom left.
        self.engine.grid = np.zeros((self.engine.width, self.engine.height), dtype=int)
        self.engine.grid[0, self.grid_height - 1] = 1  # Bottom left
        self.assertEqual(self.engine.is_illegal(), True)

        # Non-empty grid and not overlapping.
        self.engine.piece = [(0, 0), (0, -1), (0, -2), (0, -3)]  # 'I' piece vertical.
        self.engine.anchor = [0, self.grid_height - 1]  # Bottom left.
        self.engine.grid = np.zeros((self.engine.width, self.engine.height), dtype=int)
        self.engine.grid[1, :self.grid_height - 1] = 1  # Second col from left
        self.assertEqual(self.engine.is_illegal(), False)

    def test_hard_drop(self):
        pass

    def test_clear_rows(self):
        pass

    def test_update_grid(self):
        pass

    def test_compute_available_actions(self):
        pass


if __name__ == '__main__':
    unittest.main()
