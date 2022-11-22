import unittest

import plotly.io as pio

from users360 import *

pio.renderers.default = None

class Test(unittest.TestCase):

    def test_visualize_polygons(self) -> None:
        sphere = VizSphere()
        sphere.add_trace_and_fov([1, 0, 0])
        sphere.add_polygon_from_tile_row_col(4, 6, 0, 0)

        sphere = VizSphere()
        sphere.add_trace_and_fov([1, 0, 0])
        sphere.add_polygon_from_points(tile_points(4, 6, 0, 0))

    def test_visualize_fov_at_axis(self) -> None:
        traces = [[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]]
        for trace in traces:
            show_fov(trace)

    def test_visualize_trajects(self) -> None:
        show_trajects(get_df_trajects().head(1))
