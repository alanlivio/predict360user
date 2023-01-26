import logging
import pathlib

DATADIR = f"{pathlib.Path(__file__).parent.parent / 'data/'}"
HMDDIR = f"{pathlib.Path(__file__).parent / 'head_motion_prediction/'}"
DS_NAMES = ['david', 'fan', 'nguyen', 'xucvpr', 'xupami']
DS_SIZES = [1083, 300, 432, 6654, 4408]
ARGS_MODEL_NAMES = ['pos_only', 'TRACK', 'CVPR18', 'MM18', 'most_salient_point']
ARGS_DS_NAMES = ['all', 'david', 'fan', 'nguyen', 'xucvpr', 'xupami']
ARGS_ENTROPY_NAMES = ['all','low','medium','hight', 'nohight', 'low_hmp','medium_hmp','hight_hmp', 'nohight_hmp']
ARGS_ENTROPY_AUTO_NAMES = ['auto', 'auto_m_window', 'auto_since_start']
DEFAULT_EPOCHS = 50

logging.basicConfig(level=logging.INFO, format='-- users360: %(message)s')
info = logging.getLogger(__name__).info
error = logging.getLogger(__name__).error