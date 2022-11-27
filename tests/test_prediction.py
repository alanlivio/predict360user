import unittest

from users360 import *


class Test(unittest.TestCase):

  def test_get_train_test_split(self) -> None:
    X_train, X_test = [], []
    X_train, X_test = get_train_test_split("all", "all", 0.2)
    assert (not X_test.empty and not X_train.empty)
    X_train, X_test = get_train_test_split("all", "hight", 0.2)
    assert (not X_test.empty and not X_train.empty)
    X_train, X_test = get_train_test_split("all", "medium", 0.2)
    assert (not X_test.empty and not X_train.empty)
    X_train, X_test = get_train_test_split("all", "low", 0.2)
    assert (not X_test.empty and not X_train.empty)
