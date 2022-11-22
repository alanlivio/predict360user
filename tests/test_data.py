import unittest

from users360 import *


class Test(unittest.TestCase):

    def test_trajects_get(self) -> None:
        df = get_df_trajects()
        assert (not df.empty)
