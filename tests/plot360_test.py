import unittest

import plotly.io as pio

from users360.dataset import Dataset
from users360.plot360 import Plot360

pio.renderers.default = None

class Test(unittest.TestCase):

  def test_polygons(self) -> None:
    plot = Plot360()
    plot.add_trace_and_fov([1, 0, 0])
    plot.add_polygon_from_tile_row_col(4, 6, 0, 0)
    plot = Plot360()
    plot.add_trace_and_fov([1, 0, 0])
    plot.add_polygon_from_points(tile_points(4, 6, 0, 0))

  def test_show_fov_at_axis(self) -> None:
    plot = Plot360()
    traces = [[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]]
    for trace in traces:
      plot.show_fov(trace)

  def test_show_traject(self) -> None:
    one_row = Dataset().random_traject()
    self.assertFalse(one_row.empty)
    plot = Plot360()
    plot.show_traject(one_row)