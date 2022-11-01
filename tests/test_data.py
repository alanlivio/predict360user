import unittest

import pandas as pd

from users360 import *


class Test(unittest.TestCase):

    def test_trajects_get(self) -> None:
        df = get_df_trajects()
        assert (not df.empty)

    def test_trajects_tileset_metrics(self) -> None:
        calc_trajects_tileset_metrics([TILESET_DEFAULT], 1)