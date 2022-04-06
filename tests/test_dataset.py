from users360 import *
import unittest
from numpy import ndarray


class DatasetTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset.singleton()

    def test_entropy(self):
        self.dataset.users_entropy(VPEXTRACT_RECT_6_4_CENTER)
        self.assertIsNotNone(self.dataset.users_low)
        self.assertIsNotNone(self.dataset.users_medium)
        self.assertIsNotNone(self.dataset.users_hight)

    def test_poles(self):
        poles = self.dataset.traces_video_poles()
        equator = self.dataset.traces_video_equator()
        self.assertIsNotNone(poles)
        self.assertIsNotNone(equator)


if __name__ == '__main__':
    unittest.main()
