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
    self.df.drop(['traject_hmp'], axis=1, errors='ignore')

  def test_sample(self) -> None:
    # query traject row
    one_row = self.df.query(
        "ds=='david' and user=='david_0' and video=='david_10_Cows'")
    assert not one_row.empty
    # sample traject row
    one_row = sample_traject_row(self.df)
    assert not one_row.empty
    trace = sample_one_trace_from_traject_row(one_row)
    assert trace.shape == (3, )

  def test_trajects_get(self) -> None:
    videos_l = get_video_ids(self.df)
    assert videos_l.size
    users_l = get_user_ids(self.df)
    assert users_l.size
    ds_l = get_ds_ids(self.df)
    assert ds_l.size
    assert get_traces(self.df, videos_l[0], users_l[0], ds_l[0]).size

  def test_show_one_traject(self) -> None:
    one_row = sample_traject_row(self.df)
    assert not one_row.empty
    show_one_traject(one_row)