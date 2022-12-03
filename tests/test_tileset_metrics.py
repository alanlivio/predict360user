import unittest

from users360.tileset_metrics import calc_tileset_reqs_metrics
from users360.trajects import get_df_trajects
from users360.utils.tileset import TileCover, TileSet
from users360.utils.tileset_voro import TileSetVoro


class Test(unittest.TestCase):

  def setUp(self):
    # limit testing df to 2
    self.df_trajects = get_df_trajects()[:2]
    assert not self.df_trajects.empty
    # force hmp recalc
    self.df_trajects.drop(['traject_hmps'], axis=1, errors='ignore')

  def test_tileset_metrics_reqs(self) -> None:
    tileset_variations = [
        TileSet(3, 3, TileCover.ANY),
        TileSetVoro(14, TileCover.ANY)
    ]
    calc_tileset_reqs_metrics(self.df_trajects, tileset_variations)
