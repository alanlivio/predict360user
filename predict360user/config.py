import datetime
import logging
import pathlib
import sys
from os.path import join

import IPython
import plotly.graph_objs as go

# global constants
RAWDIR = f"{pathlib.Path(__file__).parent.parent / 'rawdata/'}"
DEFAULT_SAVEDIR = f"{pathlib.Path().resolve() / 'saved'}"
HMDDIR = f"{pathlib.Path(__file__).parent / 'head_motion_prediction/'}"
ARGS_MODEL_NAMES = ['pos_only', 'pos_only_3d', 'no_motion', 'interpolation', 'TRACK', 'CVPR18', 'MM18', 'most_salient_point']
MODELS_NAMES_NO_TRAIN = ['no_motion', 'interpolation']
ARGS_DS_NAMES = ['all', 'david', 'fan', 'nguyen', 'xucvpr', 'xupami']
ARGS_ENTROPY_NAMES = [ 'all', 'low', 'medium', 'high', 'nohigh', 'nolow', 'allminsize', 'low_hmp', 'medium_hmp', 'high_hmp', 'nohigh_hmp', 'nolow_hmp' ]
ARGS_ENTROPY_AUTO_NAMES = ['auto', 'auto_m_window', 'auto_since_start']
BATCH_SIZE = 128
DEFAULT_EPOCHS = 30
LEARNING_RATE = 0.0005
DS_NAMES = ['david', 'fan', 'nguyen', 'xucvpr', 'xupami']
DS_SIZES = [1083, 300, 432, 6654, 4408]
# DS_SIZES = [1083, 300, 432, 7106, 4543] # TODO: check sample_dataset folders
ENTROPY_CLASS_COLORS = {'low': 'blue', 'medium': 'green', 'high': 'red'}

# global funcs
logging.basicConfig(level=logging.INFO, format='-- predict360user: %(message)s')
info = logging.getLogger(__name__).info
error = logging.getLogger(__name__).error

def show_or_save(output, savedir, title = '') -> None:
  if 'ipykernel' in sys.modules:
    IPython.display.display(output)
  else:
    if not title:
      if isinstance(output, go.Figure):
        title = output.layout.title.text
      else:
        title = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    html_file = join(savedir, title +'.html')
    if isinstance(output, go.Figure):
      output.write_html(html_file)
    else:
      output.to_html(html_file)
    print(pathlib.Path(html_file).as_uri())