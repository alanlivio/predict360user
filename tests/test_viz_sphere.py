from users360 import *
import unittest


class Test(unittest.TestCase):
    
    def test_polygons(self):
        sphere = VizSphere()
        sphere.add_polygon_from_trace([1, 0, 0])
        sphere.add_polygon_as_row_col_tile(4, 6, 0, 0)

        sphere = VizSphere()
        sphere.add_polygon_from_trace([1, 0, 0])
        sphere.add_polygon_as_points(tile_points(4, 6, 0, 0))

    def test_fov_at_axis(self):
        traces = [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]
        for trace in traces:
            show_sphere_fov(trace, to_html=True)

    def test_trajects(self):
        show_sphere_fov(get_one_trace(), to_html=True)
        show_sphere_trajects(get_one_traject(), to_html=True)
