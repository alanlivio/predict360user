import unittest

from predict360user.dataset import Dataset, filter_df_by_entropy
from predict360user.trainer import ARGS_ENTROPY_NAMES


class DatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = Dataset()
        self.assertFalse(self.ds.df.empty)

    def test_random(self) -> None:
        one_row = self.ds.get_random_traject()
        self.assertFalse(one_row.empty)
        one_row = self.ds.df.loc[('david','0','10_Cows')]
        self.assertFalse(one_row.empty)
        trace = self.ds.get_random_trace()
        self.assertEqual(trace.shape, (3,))

    def test_trajects_get_ids(self) -> None:
        videos_l = self.ds.get_video_ids()
        self.assertTrue(videos_l.size)
        users_l = self.ds.get_user_ids()
        self.assertTrue(users_l.size)
        ds_l = self.ds.get_ds_ids()
        self.assertTrue(ds_l.size)

    def test_trajects_entropy(self) -> None:
        self.ds._df = self.ds._df.sample(n=8)  # limitig given time
        self.ds.calc_traces_entropy()

    def test_filter_df_by_entropy(self) -> None:
        min_size = self.ds.df["actS_c"].value_counts().min()
        for train_entropy in ARGS_ENTROPY_NAMES[1:]:
            fdf = filter_df_by_entropy(df=self.ds.df, entropy_filter=train_entropy)
            self.assertAlmostEqual(min_size, len(fdf), delta=2)

    def test_partition(self) -> None:
        self.ds.partition("all")
        self.assertGreater(len(self.ds.x_train), len(self.ds.x_val))
        classes = set(self.ds.x_train["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        classes = set(self.ds.x_train["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        classes = set(self.ds.x_test["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # low
        self.ds.partition("low")
        self.assertGreater(len(self.ds.x_train), len(self.ds.x_val))
        classes = set(self.ds.x_train["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low"]))
        classes = set(self.ds.x_val["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low"]))
        classes = set(self.ds.x_test["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # medium
        self.ds.partition("medium")
        self.assertGreater(len(self.ds.x_train), len(self.ds.x_val))
        classes = set(self.ds.x_train["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["medium"]))
        classes = set(self.ds.x_val["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["medium"]))
        classes = set(self.ds.x_test["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # nolow
        self.ds.partition("nolow")
        self.assertGreater(len(self.ds.x_train), len(self.ds.x_val))
        classes = set(self.ds.x_train["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["medium", "high"]))
        classes = set(self.ds.x_val["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["medium", "high"]))
        classes = set(self.ds.x_test["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # nohigh
        self.ds.partition("nohigh")
        self.assertGreater(len(self.ds.x_train), len(self.ds.x_val))
        classes = set(self.ds.x_train["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium"]))
        classes = set(self.ds.x_val["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium"]))
        classes = set(self.ds.x_test["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
        # high
        self.ds.partition("high")
        self.assertGreater(len(self.ds.x_train), len(self.ds.x_val))
        classes = set(self.ds.x_train["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["high"]))
        classes = set(self.ds.x_val["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["high"]))
        classes = set(self.ds.x_test["actS_c"].unique())
        self.assertSequenceEqual(classes, set(["low", "medium", "high"]))
