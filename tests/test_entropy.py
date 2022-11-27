import unittest

from users360 import *


class Test(unittest.TestCase):

  def test_calc_entropy(self) -> None:
    calc_trajects_entropy(testing=True)
  
  def test_calc_entropy_users(self) -> None:
    calc_trajects_entropy_users(testing=True)

  def test_calc_poles(self) -> None:
    calc_trajects_poles_prc(testing=True)

  def test_calc_tileset_reqs_metrics(self) -> None:
    calc_tileset_reqs_metrics([TILESET_DEFAULT], testing=True)
