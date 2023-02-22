import unittest

import numpy as np

from predict360user.utils.fov import calc_actual_entropy_from_ids


class Test(unittest.TestCase):

  def test_actual_entropy(self) -> None:
    ids = np.array([1, 2, 3, 4, 5, 6, 7])
    ret = calc_actual_entropy_from_ids(ids, return_sub_len_t=True)
    self.assertEqual(ret[0], 2.807)
    self.assertTrue(np.array_equal(ret[1], [1., 1., 1., 1., 1., 1., 1.]))

    ids = np.array([1, 2, 3, 1, 2, 3, 4])
    ret = calc_actual_entropy_from_ids(ids, return_sub_len_t=True)
    self.assertEqual(ret[0], 1.512)
    self.assertTrue(np.array_equal(ret[1], [1., 1., 1., 4., 3., 2., 1.]))

    ids = np.array([7, 7, 7, 7, 7, 7, 7])
    ret = calc_actual_entropy_from_ids(ids, return_sub_len_t=True)
    self.assertEqual(ret[0], 1.228)
    self.assertTrue(np.array_equal(ret[1], [1., 2., 3., 4., 3., 2., 1.]))

    ids = np.array([7, 1, 2, 3, 1, 2, 3, 4])
    ret = calc_actual_entropy_from_ids(ids, return_sub_len_t=True)
    self.assertEqual(ret[0], 1.714)
    self.assertTrue(np.array_equal(ret[1], [1., 1., 1., 1., 4., 3., 2., 1.]))
