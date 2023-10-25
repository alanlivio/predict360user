import unittest

from predict360user.ingest import Dataset
from predict360user.utils.utils import *


class DatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = Dataset()
        self.assertFalse(self.ds.df.empty)

    def test_random(self) -> None:
        one_row = self.ds.df.loc[('david','0','10_Cows')]
        self.assertFalse(one_row.empty)
        traject_ar = self.ds.df.sample(1).iloc[0]["traces"]
        trace = traject_ar[np.random.randint(len(traject_ar - 1))]
        self.assertEqual(trace.shape, (3,))

    def test_trajects_entropy(self) -> None:
        self.ds.df = self.ds.df.sample(n=8)  # limitig given time
        self.ds.calc_traces_entropy()


    def test_partition(self) -> None:
        self.ds.partition("all")
        self.assertGreater(len(self.ds.train), len(self.ds.val))
        classes = set(self.ds.train["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        classes = set(self.ds.train["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        classes = set(self.ds.test["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # low
        self.ds.partition("low")
        self.assertGreater(len(self.ds.train), len(self.ds.val))
        classes = set(self.ds.train["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low"]))
        classes = set(self.ds.val["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low"]))
        classes = set(self.ds.test["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # medium
        self.ds.partition("medium")
        self.assertGreater(len(self.ds.train), len(self.ds.val))
        classes = set(self.ds.train["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["medium"]))
        classes = set(self.ds.val["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["medium"]))
        classes = set(self.ds.test["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # nolow
        self.ds.partition("nolow")
        self.assertGreater(len(self.ds.train), len(self.ds.val))
        classes = set(self.ds.train["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["medium", "high"]))
        classes = set(self.ds.val["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["medium", "high"]))
        classes = set(self.ds.test["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # nohigh
        self.ds.partition("nohigh")
        self.assertGreater(len(self.ds.train), len(self.ds.val))
        classes = set(self.ds.train["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium"]))
        classes = set(self.ds.val["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium"]))
        classes = set(self.ds.test["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # high
        self.ds.partition("high")
        self.assertGreater(len(self.ds.train), len(self.ds.val))
        classes = set(self.ds.train["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["high"]))
        classes = set(self.ds.val["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["high"]))
        classes = set(self.ds.test["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
