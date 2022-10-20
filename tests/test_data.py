import unittest

import pandas as pd

from users360 import *


class Test(unittest.TestCase):

    def test_df_trajects(self) -> None:
        df: pd.DataFrame = get_df_trajects()
        assert (not df.empty)
