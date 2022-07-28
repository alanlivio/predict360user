from users360 import *
import unittest


class TestData(unittest.TestCase):

    def test_trajects(self):
        show_trajects(Data.singleton().get_one_traject(), to_html=True)

    def test_entropy(self):
        calc_entropy()
        calc_entropy_users()

    def test_poles(self):
        calc_poles_prc()

    # def test_tileset_metrics(self):
    #     Data.singleton().calc_metrics_tiles(
    #         [*TilesVoro.variations(), *Tiles.variations()], users=['0', '1'], perc=0.05)
