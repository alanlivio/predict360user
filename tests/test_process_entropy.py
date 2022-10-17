import unittest

from users360 import *


class Test(unittest.TestCase):

    def test_entropy(self):
        calc_trajects_entropy()
        calc_trajects_entropy_users()

    def test_poles(self):
        calc_trajects_poles_prc()
