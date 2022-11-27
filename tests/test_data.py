import unittest

import plotly.io as pio

from users360.data import get_df_trajects, show_trajects

pio.renderers.default = None


class Test(unittest.TestCase):

  def test_trajects_get(self) -> None:
    df = get_df_trajects()
    assert not df.empty

  def test_visualize_trajects(self) -> None:
    show_trajects(get_df_trajects().head(1))