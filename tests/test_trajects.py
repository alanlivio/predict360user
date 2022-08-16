from users360 import *
import unittest


class Test(unittest.TestCase):

    def test_entropy(self):
        calc_trajects_entropy()
        calc_trajects_entropy_users()

    def test_poles(self):
        calc_trajects_poles_prc()

    def test_tileset_metrics(self):
        ts_l = [TILESET_DEFAULT]
        calc_trajects_tileset_metrics(ts_l, 2)
