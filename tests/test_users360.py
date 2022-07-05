from users360 import *
import unittest
from numpy import ndarray


class TestUsers360(unittest.TestCase):

    # def test_metrics(self):
    #     Users360.singleton().calc_metrics_tiles(
    #         [*TilesVoro.variations(), *Tiles.variations()], users=['0', '1'], perc=0.05)

    def test_traces(self):
        one_user = ProjectTrajectories(Users360.singleton().example_one_trajectory())
        one_user.show(to_html=True)
        one_video = ProjectTrajectories(Users360.singleton().example_one_video_trajectories())
        one_video.show(to_html=True)
