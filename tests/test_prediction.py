import unittest

import plotly.io as pio

from users360.prediction import get_train_test_split
from users360.trajects import get_df_trajects

pio.renderers.default = None


class Test(unittest.TestCase):

  # trajects

  def setUp(self) -> None:
    self.df_trajects = get_df_trajects()
    assert not self.df_trajects.empty

  def test_prediction_train_test_split(self) -> None:
    # x_train all
    x_train, x_test = get_train_test_split(self.df_trajects, 'all', 'all', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']

    x_train, x_test = get_train_test_split(self.df_trajects, 'all', 'hight', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == 'hight'

    x_train, x_test = get_train_test_split(self.df_trajects, 'all', 'medium', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['medium']

    x_train, x_test = get_train_test_split(self.df_trajects, 'all', 'low', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low']

    # x_train low
    x_train, x_test = get_train_test_split(self.df_trajects, 'low', 'all', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']

    x_train, x_test = get_train_test_split(self.df_trajects, 'low', 'low', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low']

    x_train, x_test = get_train_test_split(self.df_trajects, 'low', 'medium', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['medium']

    x_train, x_test = get_train_test_split(self.df_trajects, 'low', 'hight', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == 'hight'

    # x_train medium
    x_train, x_test = get_train_test_split(self.df_trajects, 'medium', 'all', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['medium']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']

    x_train, x_test = get_train_test_split(self.df_trajects, 'medium', 'low', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['medium']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low']

    x_train, x_test = get_train_test_split(self.df_trajects, 'medium', 'medium', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['medium']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['medium']

    x_train, x_test = get_train_test_split(self.df_trajects, 'medium', 'hight', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['medium']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == 'hight'

    # x_train hight
    x_train, x_test = get_train_test_split(self.df_trajects, 'hight', 'all', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == 'hight'
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']

    x_train, x_test = get_train_test_split(self.df_trajects, 'hight', 'low', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == 'hight'
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low']

    x_train, x_test = get_train_test_split(self.df_trajects, 'hight', 'medium', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == 'hight'
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['medium']

    x_train, x_test = get_train_test_split(self.df_trajects, 'hight', 'hight', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == 'hight'
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == 'hight'

  def test_prediction_train_test_split_user_entropy(self) -> None:
    x_train, x_test = get_train_test_split(self.df_trajects, 'low_users', 'low_users', 0.2)
    unique_l = x_train['user_entropy_class'].unique()
    assert unique_l == 'low'
    unique_l = x_test['user_entropy_class'].unique()
    assert unique_l == 'low'

    x_train, x_test = get_train_test_split(self.df_trajects, 'medium_users', 'medium_users', 0.2)
    unique_l = x_train['user_entropy_class'].unique()
    assert unique_l == 'medium'
    unique_l = x_test['user_entropy_class'].unique()
    assert unique_l == 'medium'

    x_train, x_test = get_train_test_split(self.df_trajects, 'hight', 'hight', 0.2)
    unique_l = x_train['user_entropy_class'].unique()
    assert unique_l == 'hight'
    unique_l = x_test['user_entropy_class'].unique()
    assert unique_l == 'hight'
