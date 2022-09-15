from users360 import *
import unittest


class Test(unittest.TestCase):

    def test_polygons(self):
        sphere = VizSphere()
        sphere.add_trace_and_fov([1, 0, 0])
        sphere.add_polygon_from_tile_row_col(4, 6, 0, 0)

        sphere = VizSphere()
        sphere.add_trace_and_fov([1, 0, 0])
        sphere.add_polygon_from_points(tile_points(4, 6, 0, 0))

    def test_fov_at_axis(self):
        traces = [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]
        for trace in traces:
            show_sphere_fov(trace, to_html=True)

    def test_trajects(self):
        show_sphere_fov(get_one_trace(), to_html=True)
        show_sphere_trajects(get_one_traject(), to_html=True)
