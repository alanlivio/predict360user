import unittest

from omegaconf import OmegaConf as oc

import predict360user as p3u


class RunConfigTestCase(unittest.TestCase):

    def test_run_config(self) -> None:
        cfg = p3u.RunConfig(model="test")
        self.assertEqual(cfg.model, "test")
        self.assertEqual(cfg.name, cfg.model)

    def test_run_config_from_cli(self) -> None:
        cfg = p3u.RunConfig(**oc.from_cli(["model=test"]))  # type: ignore
        self.assertEqual(cfg.model, "test")
        self.assertEqual(cfg.model, "test")
