import pathlib
import logging
DATADIR = f"{pathlib.Path(__file__).parent.parent / 'data/'}"
HMDDIR = f"{pathlib.Path(__file__).parent / 'head_motion_prediction/'}"
logging.basicConfig(level=logging.INFO,format = '%(levelname)s:%(message)s')
