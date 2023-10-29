import unittest

from predict360user.ingest import *
from predict360user.utils.utils import *


class DatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.df = load_df_trajecs()
        self.assertFalse(self.df.empty)

    def test_random(self) -> None:
        one_row = self.df.sample(1)
        self.assertFalse(one_row.empty)
        traject_ar = one_row.iloc[0]["traces"]
        trace = traject_ar[np.random.randint(len(traject_ar - 1))]
        self.assertEqual(trace.shape, (3,))

    def test_trajects_entropy(self) -> None:
        df = self.df.sample(n=8)  # limitig given time
        df.calc_traces_entropy()

    def test_split(self) -> None:
        df = split(self.df, train_entropy="all")
        self.assertGreater(len(df[df["partition"] == "train"]), len(df[df["partition"] == "val"]))
        classes = set(df[df["partition"] == "train"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        classes = set(df[df["partition"] == "train"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        classes = set(df[df["partition"] == "test"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # low
        df = split(self.df, train_entropy="low")
        self.assertGreater(len(df[df["partition"] == "train"]), len(df[df["partition"] == "val"]))
        classes = set(df[df["partition"] == "train"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low"]))
        classes = set(df[df["partition"] == "val"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low"]))
        classes = set(df[df["partition"] == "test"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # medium
        df = split(self.df, train_entropy="medium")
        self.assertGreater(len(df[df["partition"] == "train"]), len(df[df["partition"] == "val"]))
        classes = set(df[df["partition"] == "train"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["medium"]))
        classes = set(df[df["partition"] == "val"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["medium"]))
        classes = set(df[df["partition"] == "test"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # nolow
        df = split(self.df, train_entropy="nolow")
        self.assertGreater(len(df[df["partition"] == "train"]), len(df[df["partition"] == "val"]))
        classes = set(df[df["partition"] == "train"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["medium", "high"]))
        classes = set(df[df["partition"] == "val"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["medium", "high"]))
        classes = set(df[df["partition"] == "test"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # nohigh
        df = split(self.df, train_entropy="nohigh")
        self.assertGreater(len(df[df["partition"] == "train"]), len(df[df["partition"] == "val"]))
        classes = set(df[df["partition"] == "train"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium"]))
        classes = set(df[df["partition"] == "val"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium"]))
        classes = set(df[df["partition"] == "test"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # high
        df = split(self.df, train_entropy="high")
        self.assertGreater(len(df[df["partition"] == "train"]), len(df[df["partition"] == "val"]))
        classes = set(df[df["partition"] == "train"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["high"]))
        classes = set(df[df["partition"] == "val"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["high"]))
        classes = set(df[df["partition"] == "test"]["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
