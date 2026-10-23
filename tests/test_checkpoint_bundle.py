"""Small, offline tests for exact checkpoint packing and restoration."""

import json
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from collections import OrderedDict

import torch
from safetensors.torch import save_file

RELEASE = Path(__file__).resolve().parents[1] / "release"
sys.path.insert(0, str(RELEASE))
import checkpoint_bundle as bundle
from download_zenodo import fetch_model


class CheckpointBundleTests(unittest.TestCase):
    def make_bundle(self, root):
        shared = {"embedding_model.weight": torch.tensor([0.0, -0.0, float("nan")]),
                  "embedding_model.bias": torch.tensor(3, dtype=torch.int64)}
        override = {"embedding_model.bias": torch.tensor(4, dtype=torch.int64),
                    "fc.weight": torch.tensor([[1.25, -2.5]])}
        save_file(shared, str(root / "base.safetensors"))
        save_file(override, str(root / "seed.safetensors"))
        expected = OrderedDict({**shared, **override})
        model = {"condition": "example", "seed": 1, "backbone": "esm2",
                 "file": "seed.safetensors", "sha256": bundle.sha256_file(root / "seed.safetensors"),
                 "tensor_order": list(expected),
                 "tensor_sha256": {k: bundle.tensor_digest(v) for k, v in expected.items()},
                 "state_dict_metadata": {"": {"version": 1}}}
        manifest = {"schema": bundle.SCHEMA, "backbones": {"esm2": {
            "file": "base.safetensors", "sha256": bundle.sha256_file(root / "base.safetensors")}},
            "models": [model]}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return expected

    def test_roundtrip_preserves_bits_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = self.make_bundle(root)
            actual = bundle.load_bundle_state(root, "example", 1)
            self.assertEqual(list(actual), list(expected))
            for k in expected:
                self.assertEqual(bundle.tensor_digest(actual[k]), bundle.tensor_digest(expected[k]))
            self.assertEqual(actual._metadata, {"": {"version": 1}})

    def test_modified_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_bundle(root)
            with (root / "seed.safetensors").open("ab") as f:
                f.write(b"bad")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                bundle.load_bundle_state(root, "example", 1)

    def test_scalar_and_signed_zero(self):
        self.assertNotEqual(bundle.tensor_digest(torch.tensor(0.0)),
                            bundle.tensor_digest(torch.tensor(-0.0)))
        bundle.tensor_digest(torch.tensor(True))
        bundle.tensor_digest(torch.empty(0))
        bundle.tensor_digest(torch.tensor([1.0], dtype=torch.bfloat16))

    def test_path_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                bundle.local_file(Path(tmp), "../outside")

    def test_selective_download_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_bundle(root)
            manifest = json.loads((root / "manifest.json").read_text())
            loader = b"# test fixture only\n"
            manifest["source_code"] = {"file_sha256": {
                "release/checkpoint_bundle.py": hashlib.sha256(loader).hexdigest()}}
            downloads = {"manifest.json": json.dumps(manifest).encode(),
                         "checkpoint_bundle.py": loader,
                         "base.safetensors": (root / "base.safetensors").read_bytes(),
                         "seed.safetensors": (root / "seed.safetensors").read_bytes()}
            def open_url(url, **kwargs):
                return io.BytesIO(downloads[url.rsplit("/", 1)[-1].split("?")[0]])
            with patch("download_zenodo.urllib.request.urlopen", side_effect=open_url) as opened:
                fetch_model(123, "example", [1], root / "download")
                self.assertEqual(opened.call_count, 4)
                opened.reset_mock()
                fetch_model(123, "example", [1], root / "download")
                self.assertEqual(opened.call_count, 1)
            bundle.load_bundle_state(root / "download", "example", 1)



if __name__ == "__main__":
    unittest.main()
