import unittest
from os.path import join

from users360 import config
from users360.trainer import Trainer, count_traject_entropy_classes


class Test(unittest.TestCase):

  def test_partition_train(self) -> None:
    trn = Trainer(dry_run=True)
    trn.partition_train()
    assert trn.model_dir == join(config.DATADIR, 'pos_only')
    assert trn.model_weights == join(config.DATADIR, 'pos_only', 'weights.hdf5')
    assert (count_traject_entropy_classes(trn.x_train) == (10301, 6125, 3133, 1043))
    assert (count_traject_entropy_classes(trn.x_test) == (2576, 1510, 814, 252))

  def test_partition_evaluate(self) -> None:
    trn = Trainer(dry_run=True)
    assert trn.model_dir == join(config.DATADIR, 'pos_only')
    assert trn.model_weights == join(config.DATADIR, 'pos_only', 'weights.hdf5')
    trn.partition_evaluate()
    assert (count_traject_entropy_classes(trn.x_test) == (2576, 1510, 814, 252))
