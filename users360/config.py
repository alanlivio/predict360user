"""
Provides shared data
"""
import logging
import pathlib

DATADIR = f"{pathlib.Path(__file__).parent.parent / 'data/'}"
HMDDIR = f"{pathlib.Path(__file__).parent / 'head_motion_prediction/'}"
DS_NAMES = ['david', 'fan', 'nguyen', 'xucvpr', 'xupami']
DS_SIZES = [1083, 300, 432, 6654, 4408]
MODEL_NAMES = ['pos_only', 'TRACK', 'CVPR18', 'MM18', 'most_salient_point']
ARGS_DS_NAMES = ['all', 'david', 'fan', 'nguyen', 'xucvpr', 'xupami']
ARGS_ENTROPY_NAMES = ['all','low','medium','hight']
ARGS_ENTROPY_AUTO_NAMES = ['auto', 'auto_m_window', 'auto_since_start']

logging.basicConfig(level=logging.INFO, format='-- users360: %(message)s')
info = logging.getLogger(__name__).info
error = logging.getLogger(__name__).error