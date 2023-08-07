import unittest

from predict360user.dataset import Dataset
from predict360user.tileset import *


class TileSetTestCase(unittest.TestCase):
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

    def test_tileset_metrics_reqs(self) -> None:
        # limit testing df to 2
        self.ds = Dataset()
        self.assertFalse(self.ds.df.empty)
        self.ds.df = self.ds.df[:2]
        # force hmp recalc
        self.ds.df.drop(["traces_hmp"], axis=1, errors="ignore")
        tileset_variations = [
            TileSet(3, 3, TileCover.ANY),
            TileSetVoro(14, TileCover.ANY),
        ]
        self.ds.calc_tileset_reqs_metrics(tileset_variations)
