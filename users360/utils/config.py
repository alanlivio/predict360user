import logging
import pathlib

DATADIR = f"{pathlib.Path(__file__).parent.parent.parent / 'data/'}"
logging.basicConfig(level=logging.INFO, format='-- users360: %(message)s')
info = logging.getLogger(__name__).info
error = logging.getLogger(__name__).error