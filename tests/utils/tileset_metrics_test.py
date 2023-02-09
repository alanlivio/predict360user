import unittest

from users360.dataset import Dataset
from users360.utils.tileset import TileCover, TileSet
from users360.utils.tileset_metrics import calc_tileset_reqs_metrics
from users360.utils.tileset_voro import TileSetVoro


class Test(unittest.TestCase):

  def setUp(self) -> None:
    # limit testing df to 2
    self.ds = Dataset()
    self.assertFalse(self.ds.df.empty)
    # force hmp recalc
    self.ds.df.drop(['traject_hmp'], axis=1, errors='ignore')

  def test_tileset_metrics_reqs(self) -> None:
    tileset_variations = [TileSet(3, 3, TileCover.ANY), TileSetVoro(14, TileCover.ANY)]
    calc_tileset_reqs_metrics(self.ds.df[:2], tileset_variations)
