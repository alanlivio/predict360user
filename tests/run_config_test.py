import unittest

import plotly.io as pio
from omegaconf import OmegaConf as oc

import predict360user as p3u

pio.renderers.default = None


class RunConfigTestCase(unittest.TestCase):

    def test_run_config(self) -> None:
        cfg = p3u.RunConfig()
        self.assertEqual(cfg.name, cfg.model)

    def test_run_config_from_cli(self) -> None:
        cfg = p3u.RunConfig(**oc.from_cli(["name=test"])) # type: ignore
        self.assertEqual(cfg.name, "test")
