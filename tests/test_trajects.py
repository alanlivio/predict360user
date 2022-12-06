import unittest

import plotly.io as pio

from users360.entropy import (calc_trajects_entropy, calc_trajects_poles_prc,
                              calc_users_entropy)
from users360.trajects import (get_df_trajects, get_ds_ids, get_traces,
                               get_user_ids, get_video_ids,
                               sample_one_trace_from_traject_row,
                               sample_traject_row, show_trajects)

pio.renderers.default = None


class Test(unittest.TestCase):

  def setUp(self) -> None:
    # limit testing df to 2
    self.df_trajects = get_df_trajects()[:2]
    assert not self.df_trajects.empty
    # force hmp recalc
    self.df_trajects.drop(['traject_hmps'], axis=1, errors='ignore')

  def test_sample(self) -> None:
    # query traject row
    one_row = self.df_trajects.query("ds=='david' and ds_user=='david_0' and ds_video=='david_10_Cows'")
    assert not one_row.empty
    # sample traject row
    one_row = sample_traject_row(self.df_trajects)
    assert not one_row.empty
    trace = sample_one_trace_from_traject_row(one_row)
    assert trace.shape == (3,)


  def test_trajects_get(self) -> None:
    videos_l = get_video_ids(self.df_trajects)
    assert videos_l.size
    users_l = get_user_ids(self.df_trajects)
    assert users_l.size
    ds_l = get_ds_ids(self.df_trajects)
    assert ds_l.size
    assert get_traces(self.df_trajects, videos_l[0], users_l[0], ds_l[0]).size

  def test_trajects_show(self) -> None:
    show_trajects(self.df_trajects)

  def test_trajects_entropy(self) -> None:
    self.df_trajects.drop(['traject_entropy', 'traject_entropy_class'], axis=1, errors='ignore')
    calc_trajects_entropy(self.df_trajects)

  def test_entropy_users(self) -> None:
    self.df_trajects.drop(['user_entropy', 'user_entropy_class'], axis=1, errors='ignore')
    assert not calc_users_entropy(self.df_trajects)

  def test_poles_prc(self) -> None:
    self.df_trajects.drop(['poles_prc', 'poles_prc_class'], axis=1, errors='ignore')
    calc_trajects_poles_prc(self.df_trajects)