from users360 import *
import unittest

class TestProjection(unittest.TestCase):

    def test_vp(self):
        traces = [ [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1] ]
        for trace in traces:
            onevp = PlotVP(trace)
            self.assertIsNotNone(onevp)
            onevp.show(to_html=True)
            onevp = PlotVP(trace,TILES_VORO_14_CENTER)
            self.assertIsNotNone(onevp)
            onevp.show(to_html=True)
