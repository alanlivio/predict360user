"""
Provides shared data
"""
import logging
import os
import pathlib

import pandas as pd

DATADIR = f"{pathlib.Path(__file__).parent.parent / 'data/'}"
HMDDIR = f"{pathlib.Path(__file__).parent / 'head_motion_prediction/'}"
DS_NAMES = ['david', 'fan', 'nguyen', 'xucvpr', 'xupami']
DS_SIZES = [1083, 300, 432, 6654, 4408]

logging.basicConfig(level=logging.INFO, format='-- %(filename)s: %(message)s')

# https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
df_trajects: pd.DataFrame = None # type: ignore
df_trajects_f = os.path.join(DATADIR, 'df_trajects.pickle')
df_tileset_metrics: pd.DataFrame = None # type: ignore
df_tileset_metrics_f = os.path.join(DATADIR, 'df_tileset_metrics.pickle')
