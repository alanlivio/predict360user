from users360 import *
import unittest
from numpy import ndarray


class TestTrajectories(unittest.TestCase):

    # def test_metrics(self):
    #     Trajectories.singleton().calc_metrics_tiles(
    #         [*TilesVoro.variations(), *Tiles.variations()], users=['0', '1'], perc=0.05)

    def test_one(self):
        show_trajects(Trajectories.singleton().get_one_traject(), to_html=True)

    def test_one(self):
        show_trajects(Trajectories.singleton().get_one_video_trajects(), to_html=True)
