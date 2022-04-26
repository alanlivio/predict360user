from users360 import *
import unittest

class DatasetTest(unittest.TestCase):

    def test_vp(self):
        traces = [ [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1] ]
        for trace in traces:
            onevp = PlotVP(trace)
            self.assertIsNotNone(onevp)
            onevp.show(VPEXTRACT_RECT_4_6_CENTER, to_html=True)
            onevp.show(VPEXTRACT_VORO_14_CENTER, to_html=True)
            onevp.show(VPEXTRACT_RECT_4_6_CENTER, to_html=True)
            onevp.show(VPEXTRACT_VORO_14_CENTER, to_html=True)
