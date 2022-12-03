import unittest

import plotly.io as pio

from users360.entropy import (calc_trajects_entropy, calc_trajects_poles_prc,
                              calc_users_entropy)
from users360.trajects import (get_df_trajects, get_ds_ids, get_one_trace,
                               get_traces, get_user_ids, get_video_ids,
                               show_trajects)

pio.renderers.default = None


class Test(unittest.TestCase):

  # trajects

  def setUp(self):
    # limit testing df to 2
    self.df_trajects = get_df_trajects()[:2]
    assert not self.df_trajects.empty
    # force hmp recalc
    self.df_trajects.drop(['traject_hmps'], axis=1)

  def test_trajects_get(self) -> None:
    assert get_one_trace(self.df_trajects).size
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
    self.df_trajects.drop(['traject_entropy', 'traject_entropy_class'], axis=1)
    calc_trajects_entropy(self.df_trajects)

  def test_entropy_users(self) -> None:
    self.df_trajects.drop(['user_entropy', 'user_entropy_class'], axis=1)
    assert not calc_users_entropy(self.df_trajects)

  def test_tilset_metrics_poles(self) -> None:
    self.df_trajects.drop(['poles_prc', 'poles_prc_class'], axis=1)
    calc_trajects_poles_prc(self.df_trajects)