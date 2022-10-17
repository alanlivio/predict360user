import unittest

from numpy import array as array

from users360 import *


class Test(unittest.TestCase):

    def test_df_trajects(self):
        df = get_df_trajects()
        assert (not df.empty)