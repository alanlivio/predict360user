import datetime
import logging
import pathlib
import sys
from os.path import join

import IPython
import plotly.graph_objs as go

DATADIR = f"{pathlib.Path(__file__).parent.parent.parent / 'data/'}"
logging.basicConfig(level=logging.INFO, format='-- predict360user: %(message)s')
info = logging.getLogger(__name__).info
error = logging.getLogger(__name__).error

def show_or_save(output, title = '') -> None:
  if 'ipykernel' in sys.modules:
    IPython.display.display(output)
  else:
    if not title:
      if isinstance(output, go.Figure):
        title = output.layout.title.text
      else:
        title = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    html_file = join(DATADIR, title +'.html')
    if isinstance(output, go.Figure):
      output.write_html(html_file)
    else:
      output.to_html(html_file)
    print(pathlib.Path(html_file).as_uri())