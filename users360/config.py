"""
Provides shared data
"""
import logging
import pathlib

DATADIR = f"{pathlib.Path(__file__).parent.parent / 'data/'}"
HMDDIR = f"{pathlib.Path(__file__).parent / 'head_motion_prediction/'}"
DS_NAMES = ['david', 'fan', 'nguyen', 'xucvpr', 'xupami']
DS_SIZES = [1083, 300, 432, 6654, 4408]

logging.basicConfig(level=logging.INFO, format='-- %(filename)s: %(message)s')
info = logging.getLogger(__name__).info
error = logging.getLogger(__name__).error