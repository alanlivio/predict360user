import logging
import pathlib

DATADIR = f"{pathlib.Path(__file__).parent.parent.parent / 'data/'}"
logging.basicConfig(level=logging.INFO, format='-- predict360user: %(message)s')
info = logging.getLogger(__name__).info
error = logging.getLogger(__name__).error