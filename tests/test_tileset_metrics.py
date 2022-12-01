import unittest

from users360.tileset_metrics import calc_tileset_reqs_metrics
from users360.trajects import get_df_trajects
from users360.utils.tileset import TileCover, TileSet


class Test(unittest.TestCase):

  def test_tileset_metrics_reqs(self) -> None:
    # limit testing df to 1
    self.df_trajects = get_df_trajects()[:1]
    assert not self.df_trajects.empty
    # use a simple tileset
    self.tileset_variations = [TileSet(3, 3, TileCover.CENTER), TileSet(3, 3, TileCover.ANY)]
    assert not calc_tileset_reqs_metrics(self.df_trajects, self.tileset_variations).empty
