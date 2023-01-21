import unittest

import plotly.io as pio

from users360.entropy import *
from users360.trajects import *

pio.renderers.default = None


class Test(unittest.TestCase):

  def setUp(self) -> None:
    # limit testing df to 2
    self.df = get_df_trajects()[:2]
    assert not self.df.empty
    # force hmp recalc
    self.df.drop(['traject_hmps'], axis=1, errors='ignore')

  def test_trajects_entropy(self) -> None:
    self.df.drop(['traject_entropy', 'traject_entropy_class'],
                          axis=1,
                          errors='ignore')
    calc_trajects_entropy(self.df)

  def test_poles_prc(self) -> None:
    self.df.drop(['poles_prc', 'poles_prc_class'],
                          axis=1,
                          errors='ignore')
    calc_trajects_poles_prc(self.df)

  def test_actual_entropy(self) -> None:
    ids = np.array([1, 2, 3, 4, 5, 6, 7])
    ret = calc_actual_entropy_from_ids(ids, return_sub_len_t=True)
    assert ret[0] == 2.807
    assert np.array_equal(ret[1], [1., 1., 1., 1., 1., 1., 1.])

    ids = np.array([1, 2, 3, 1, 2, 3, 4])
    ret = calc_actual_entropy_from_ids(ids, return_sub_len_t=True)
    assert ret[0] == 1.512
    assert np.array_equal(ret[1], [1., 1., 1., 4., 3., 2., 1.])

    ids = np.array([7, 7, 7, 7, 7, 7, 7])
    ret = calc_actual_entropy_from_ids(ids, return_sub_len_t=True)
    assert ret[0] == 1.228
    assert np.array_equal(ret[1], [1., 2., 3., 4., 3., 2., 1.])

    ids = np.array([7, 1, 2, 3, 1, 2, 3, 4])
    ret = calc_actual_entropy_from_ids(ids, return_sub_len_t=True)
    assert ret[0] == 1.714
    assert np.array_equal(ret[1], [1., 1., 1., 1., 4., 3., 2., 1.])
