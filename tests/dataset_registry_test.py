import unittest
import pooch

import predict360user as p3u
from predict360user.data import (
    POOCH,
    DATA_POOCH,
    DATASET_URLS,
    Trajectory360Dataset,
    fetch_dataset,
)


class DatasetRegistryTestCase(unittest.TestCase):
    def test_pooch_registry_in_data_ingestion(self) -> None:
        self.assertIsInstance(POOCH, pooch.Pooch)
        self.assertIs(POOCH, DATA_POOCH)
        for fname in (
            "David_MMSys_18.zip",
            "Fan_NOSSDAV_17.zip",
            "Nguyen_MM_18.zip",
            "Xu_CVPR_18.zip",
            "Xu_PAMI_18.zip",
        ):
            self.assertIn(fname, POOCH.registry)
            self.assertIn(fname, DATASET_URLS)

        self.assertTrue(DATASET_URLS["David_MMSys_18.zip"].startswith("doi:10.5281/zenodo.4673531/"))
        self.assertIn("gitlab.com/miguelfromeror/head-motion-prediction", DATASET_URLS["Fan_NOSSDAV_17.zip"])

    def test_resolve_dataset_manifest_for_david(self) -> None:
        ds = Trajectory360Dataset(name="david", split="train", download=False)
        self.assertGreater(len(ds), 0)
        manifest = p3u.resolve_dataset_manifest("david", download=False)
        self.assertEqual(manifest["name"], "david")
        self.assertEqual(manifest["fname"], "David_MMSys_18.zip")
        self.assertTrue(manifest["verified"])
        self.assertGreater(manifest["num_files"], 0)
        self.assertNotEqual(manifest["content_hash"], "")
        self.assertEqual(fetch_dataset("david")["content_hash"], manifest["content_hash"])


if __name__ == "__main__":
    unittest.main()
