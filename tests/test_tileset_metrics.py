import unittest

from users360.tileset_metrics import (calc_tileset_reqs_metrics,
                                      calc_trajects_poles_prc)
from users360.utils.tileset import TILESET_VARIATIONS_FOR_TEST


class Test(unittest.TestCase):

  def test_tilset_metrics_poles(self) -> None:
    calc_trajects_poles_prc(testing=True)

  def test_tileset_metrics_reqs(self) -> None:
    calc_tileset_reqs_metrics(TILESET_VARIATIONS_FOR_TEST, testing=True)
