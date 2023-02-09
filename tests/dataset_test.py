import unittest

import plotly.io as pio

from users360.dataset import Dataset
from users360.utils.viz_sphere import *

pio.renderers.default = None

class Test(unittest.TestCase):

  def setUp(self) -> None:
    self.ds = Dataset()
    assert not self.ds.df.empty

  def test_random(self) -> None:
    one_row = self.ds.random_traject()
    assert not one_row.empty
    one_row = self.ds.df.query("ds=='david' and user=='david_0' and video=='david_10_Cows'")
    assert not one_row.empty
    trace = self.ds.random_trace()
    assert trace.shape == (3, )

  def test_trajects_get(self) -> None:
    videos_l = self.ds.get_video_ids()
    assert videos_l.size
    users_l = self.ds.get_user_ids()
    assert users_l.size
    ds_l = self.ds.get_ds_ids()
    assert ds_l.size
    assert self.ds.get_traces(videos_l[0], users_l[0], ds_l[0]).size

  def test_show_one_traject(self) -> None:
    one_row = self.ds.random_traject()
    assert not one_row.empty
    show_one_traject(one_row)