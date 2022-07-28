from users360 import *
import unittest


class TestProjection(unittest.TestCase):
    def test_projection(self):
        ploj = Projection()
        ploj.add_polygon_from_trace([1, 0, 0])
        ploj.add_polygon_as_row_col_tile(4, 6, 0, 0)

        ploj = Projection()
        ploj.add_polygon_from_trace([1, 0, 0])
        ploj.add_polygon_as_points(TileSet.tile_points(4, 6, 0, 0))
    
    def test_fov_at_axis(self):
        traces = [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]
        for trace in traces:
            show_fov(trace, to_html=True)
    
    def test_trajects(self):
        show_trajects(Data.singleton().get_one_trace(), to_html=True)