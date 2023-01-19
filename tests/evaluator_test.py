import unittest
from os.path import join

from users360 import config
from users360.evaluator import Evaluator
from users360.trainer import count_traject_entropy_classes


class Test(unittest.TestCase):

  def test_init(self) -> None:
    etr = Evaluator(dry_run=True)
    assert etr.model_dir == join(config.DATADIR, 'pos_only')
    assert etr.model_weights == join(config.DATADIR, 'pos_only', 'weights.hdf5')


  def test_partition(self) -> None:
    etr = Evaluator(dry_run=True)
    etr._partition()
    assert (count_traject_entropy_classes(etr.x_train) == (10301, 6125, 3133, 1043))
    assert (count_traject_entropy_classes(etr.x_test) == (2576, 1510, 814, 252))