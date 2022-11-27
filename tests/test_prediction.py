import unittest

from users360.prediction import get_train_test_split


class Test(unittest.TestCase):

  def test_prediction_train_test_split(self) -> None:
    x_train, x_test = [], []
    x_train, x_test = get_train_test_split("all", "all", 0.2)
    assert (not x_test.empty and not x_train.empty)
    x_train, x_test = get_train_test_split("all", "hight", 0.2)
    assert (not x_test.empty and not x_train.empty)
    x_train, x_test = get_train_test_split("all", "medium", 0.2)
    assert (not x_test.empty and not x_train.empty)
    x_train, x_test = get_train_test_split("all", "low", 0.2)
    assert (not x_test.empty and not x_train.empty)
