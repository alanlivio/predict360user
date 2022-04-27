from users360 import *
import unittest

class TestProjection(unittest.TestCase):


    def test_vp_at_axis(self):
        traces = [ [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1] ]
        for trace in traces:
            onevp = PlotVP(trace)
            self.assertIsNotNone(onevp)
            onevp.show(to_html=True)
            onevp = PlotVP(trace, TilesVoro.default())
            self.assertIsNotNone(onevp)
            onevp.show(to_html=True)
            
    def test_polygon(self):
        plot = PlotPolygon()
        plot.add_polygon_from_trace([1, 0, 0])
        plot.add_polygon_as_row_col_tile(4,6,0,0)

        plot = PlotPolygon()
        plot.add_polygon_from_trace([1, 0, 0])
        plot.add_polygon_as_points(tile_points(4,6,0,0))