from users360 import *
import unittest
from numpy import ndarray


class TestDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset.singleton()

    def test_entropy(self):
        self.dataset.users_entropy(TILES_4_6_CENTER)
        self.assertIsNotNone(self.dataset.users_low)
        self.assertIsNotNone(self.dataset.users_medium)
        self.assertIsNotNone(self.dataset.users_hight)

    def test_poles(self):
        poles = self.dataset.traces_video_poles()
        equator = self.dataset.traces_video_equator()
        self.assertIsNotNone(poles)
        self.assertIsNotNone(equator)

    def test_traces(self):
        one_user = PlotTraces(self.dataset.traces_video_user(), title_sufix="one_video_one_user")
        self.assertIsNotNone(one_user)
        self.assertIsInstance(one_user.traces, ndarray)
        one_user.show(to_html=True)
        all_users = PlotTraces(self.dataset.traces_video(), title_sufix="one_video_all_users")
        self.assertIsNotNone(all_users)
        self.assertIsInstance(all_users.traces, ndarray)
        all_users.show(to_html=True)
        some_traces = PlotTraces(self.dataset.traces_video_user(0.1), title_sufix="some_traces")
        self.assertIsNotNone(some_traces)
        self.assertIsInstance(some_traces.traces, ndarray)
        some_traces.show(to_html=True)