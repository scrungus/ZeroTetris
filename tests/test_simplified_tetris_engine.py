import unittest   # The test framework

from gym_simplifiedtetris.envs import SimplifiedTetrisEngine as Engine

class SimplifiedTetrisEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = Engine(
            grid_dims=(20, 10),
            piece_size=4,
            num_pieces=7,
            num_actions=34,
        )

    def tearDown(self) -> None:
        del self.engine
    
    def test_get_bgr_code(self):
        bgr_code_orange = self.engine.get_bgr_code('red')
        self.assertEqual(bgr_code_orange, (0, 165, 255))
        bgr_code_coral = self.engine.get_bgr_code('coral')
        self.assertEqual(bgr_code_coral, (80, 127, 255))
        bgr_code_orangered = self.engine.get_bgr_code('red')
        self.assertEqual(bgr_code_orangered, (0, 69, 255))

    def test_decrement(self):
        self.assertEqual(3, 4)

if __name__ == '__main__':
    unittest.main()
