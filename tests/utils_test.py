import unittest

import numpy as np
import plotly.io as pio

from predict360user.utils.math360 import calc_actual_entropy_from_ids
from predict360user.utils.plot360 import Plot360
from predict360user.utils.tileset360 import fov_poly, tile_points, tile_poly

pio.renderers.default = None


class TileSet360TestCase(unittest.TestCase):
    def test_tileset_polys(self) -> None:
        tile_area_6x4_equator = 0.927295
        tile_area_6x4_tropics = 0.500154
        tile_area_6x4_poles = 0.143348
        # tile area at poles
        for row in [0, 5]:
            for col in range(4):
                self.assertAlmostEqual(
                    tile_poly(6, 4, row, col).area(), tile_area_6x4_poles, places=4
                )
        # tile area at equator
        for row in [1, 4]:
            for col in range(4):
                self.assertAlmostEqual(
                    tile_poly(6, 4, row, col).area(), tile_area_6x4_tropics, places=4
                )
        # tile area tropics
        for row in [2, 3]:
            for col in range(4):
                self.assertAlmostEqual(
                    tile_poly(6, 4, row, col).area(), tile_area_6x4_equator, places=4
                )

        # trace area
        area_fov = 3.161202
        traces = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        for trace in traces:
            self.assertAlmostEqual(fov_poly(*trace).area(), area_fov, places=4)
        # trace at poles overlap tiles at poles
        for col in range(4):
            self.assertAlmostEqual(
                tile_poly(6, 4, 0, col).overlap(fov_poly(0, 0, 1)), 1.0, places=4
            )
            self.assertAlmostEqual(
                tile_poly(6, 4, 0, col).overlap(fov_poly(0, 0, -1)), 0.0, places=4
            )
        for col in range(4):
            self.assertAlmostEqual(
                tile_poly(6, 4, 5, col).overlap(fov_poly(0, 0, 1)), 0.0, places=4
            )
            self.assertAlmostEqual(
                tile_poly(6, 4, 5, col).overlap(fov_poly(0, 0, -1)), 1.0, places=4
            )


class Plot360TestCase(unittest.TestCase):
    def test_polygons(self) -> None:
        plot = Plot360()
        plot.add_trace_and_fov([1, 0, 0])
        plot.add_polygon_from_tile_row_col(4, 6, 0, 0)
        plot = Plot360()
        plot.add_trace_and_fov([1, 0, 0])
        plot.add_polygon_from_points(tile_points(4, 6, 0, 0))

    def test_show_fov_at_axis(self) -> None:
        plot = Plot360()
        traces = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
        for trace in traces:
            plot.show_fov(trace)


class Math360TestCase(unittest.TestCase):
    def test_actual_entropy(self) -> None:
        ids = np.array([1, 2, 3, 4, 5, 6, 7])
        ent, sub = calc_actual_entropy_from_ids(ids, return_sub_len_t=True)  # type: ignore
        self.assertEqual(ent, 2.807)
        self.assertTrue(np.array_equal(sub, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        ids = np.array([1, 2, 3, 1, 2, 3, 4])
        ent, sub = calc_actual_entropy_from_ids(ids, return_sub_len_t=True)  # type: ignore
        self.assertEqual(ent, 1.512)
        self.assertTrue(np.array_equal(sub, [1.0, 1.0, 1.0, 4.0, 3.0, 2.0, 1.0]))

        ids = np.array([7, 7, 7, 7, 7, 7, 7])
        ent, sub = calc_actual_entropy_from_ids(ids, return_sub_len_t=True)  # type: ignore
        self.assertEqual(ent, 1.228)
        self.assertTrue(np.array_equal(sub, [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]))

        ids = np.array([7, 1, 2, 3, 1, 2, 3, 4])
        ent, sub = calc_actual_entropy_from_ids(ids, return_sub_len_t=True)  # type: ignore
        self.assertEqual(ent, 1.714)
        self.assertTrue(np.array_equal(sub, [1.0, 1.0, 1.0, 1.0, 4.0, 3.0, 2.0, 1.0]))
