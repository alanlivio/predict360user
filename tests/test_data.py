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

    def test_tileset_metrics(self):
        ts_l = [TileSet.default()]
        calc_tileset_metrics(ts_l, 2)
