import unittest

import plotly.io as pio

from users360.entropy import calc_trajects_entropy
from users360.trainer import get_train_test_split
from users360.trajects import get_df_trajects

pio.renderers.default = None


class Test(unittest.TestCase):

  # trajects

  def setUp(self) -> None:
    self.df = get_df_trajects()
    assert not self.df.empty

  def test_prediction_train_test_split(self) -> None:
    # x_train all
    x_train, x_test = get_train_test_split(self.df, 'all', 0.2)
    unique_s = set(x_train['traject_entropy_class'].unique())
    assert unique_s == set(['low', 'medium', 'hight'])
    unique_s = set(x_test['traject_entropy_class'].unique())
    assert unique_s == set(['low', 'medium', 'hight'])

    # x_train low
    x_train, x_test = get_train_test_split(self.df, 'low', 0.2)
    unique_s = set(x_train['traject_entropy_class'].unique())
    assert unique_s == set(['low'])
    unique_s = set(x_test['traject_entropy_class'].unique())
    assert unique_s == set(['low'])

    # x_train medium
    x_train, x_test = get_train_test_split(self.df, 'medium', 0.2)
    unique_s = set(x_train['traject_entropy_class'].unique())
    assert unique_s == set(['medium'])
    unique_s = set(x_test['traject_entropy_class'].unique())
    assert unique_s == set(['medium'])

    # x_train hight
    x_train, x_test = get_train_test_split(self.df, 'hight', 0.2)
    unique_s = set(x_train['traject_entropy_class'].unique())
    assert unique_s == set(['hight'])
    unique_s = set(x_test['traject_entropy_class'].unique())
    assert unique_s == set(['hight'])


  def test_prediction_train_test_split_user_entropy(self) -> None:
    x_train, x_test = get_train_test_split(self.df, 'low_users', 0.2)
    unique_s = set(x_train['user_entropy_class'].unique())
    assert unique_s == set(['low'])
    unique_s = set(x_test['user_entropy_class'].unique())
    assert unique_s == set(['low'])

    x_train, x_test = get_train_test_split(self.df, 'medium_users', 0.2)
    unique_s = set(x_train['user_entropy_class'].unique())
    assert unique_s == set(['medium'])
    unique_s = set(x_test['user_entropy_class'].unique())
    assert unique_s == set(['medium'])

    x_train, x_test = get_train_test_split(self.df, 'hight_users', 0.2)
    unique_s = set(x_train['user_entropy_class'].unique())
    assert unique_s == set(['hight'])
    unique_s = set(x_test['user_entropy_class'].unique())
    assert unique_s == set(['hight'])
