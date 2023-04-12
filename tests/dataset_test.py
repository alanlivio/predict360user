import unittest

from predict360user.dataset import Dataset


class Test(unittest.TestCase):

  def setUp(self) -> None:
    self.ds = Dataset()
    self.assertFalse(self.ds.df.empty)

  def test_random(self) -> None:
    one_row = self.ds.get_traject_random()
    self.assertFalse(one_row.empty)
    one_row = self.ds.df.query("ds=='david' and user=='david_0' and video=='david_10_Cows'")
    self.assertFalse(one_row.empty)
    trace = self.ds.get_trace_random()
    self.assertEqual(trace.shape, (3, ))

  def test_trajects_get(self) -> None:
    videos_l = self.ds.get_video_ids()
    self.assertTrue(videos_l.size)
    users_l = self.ds.get_user_ids()
    self.assertTrue(users_l.size)
    ds_l = self.ds.get_ds_ids()
    self.assertTrue(ds_l.size)
    self.assertTrue(self.ds.get_traces(videos_l[0], users_l[0]).size)

  def test_trajects_entropy(self) -> None:
    self.ds.df = self.ds.df.sample(n=8) # limitig given time
    self.ds.calc_traces_entropy()
    self.ds.calc_traces_entropy_hmp()
    self.ds.calc_traces_poles_prc()