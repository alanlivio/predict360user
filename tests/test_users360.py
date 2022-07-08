from users360 import *
import unittest
from numpy import ndarray


class TestTrajectories(unittest.TestCase):

    # def test_metrics(self):
    #     Trajectories.singleton().calc_metrics_tiles(
    #         [*TilesVoro.variations(), *Tiles.variations()], users=['0', '1'], perc=0.05)

    def test_one(self):
        one_user = ProjectTrajectories(Trajectories.singleton().example_one_trajectory())
        one_user.show(to_html=True)
    
    def test_one(self):
        one_video = ProjectTrajectories(Trajectories.singleton().example_one_video_trajectories())
        one_video.show(to_html=True)
