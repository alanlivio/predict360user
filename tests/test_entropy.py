import unittest

from users360.entropy import calc_trajects_entropy, calc_trajects_entropy_users


class Test(unittest.TestCase):

  def test_entropy(self) -> None:
    calc_trajects_entropy(testing=True)

  def test_entropy_users(self) -> None:
    calc_trajects_entropy_users(testing=True)
