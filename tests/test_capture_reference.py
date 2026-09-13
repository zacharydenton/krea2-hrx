"""Check reference-loader wiring without downloading models or importing Torch."""
import ast
import re
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class ReferenceLoaderTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "scripts/capture_reference.py"
        tree = ast.parse(source.read_text())
        # Execute the actual loader functions with mocked model libraries. The
        # rest of the capture script requires NumPy and inference dependencies.
        nodes = [node for node in tree.body if
                 isinstance(node, ast.FunctionDef) and node.name in {"build", "model_revision"}
                 or isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id in {"REPO", "REVISION"}
                     for target in node.targets)]
        self.scope = {"re": re}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), self.scope)
        self.scope["diffusers_state"] = lambda weights: {}
        self.scope["cast_transformer"] = lambda transformer, dtype: transformer
        self.pipe = MagicMock()
        self.pipe.to.return_value = self.pipe
        self.loader = MagicMock(return_value=self.pipe)
        self.transformer = MagicMock()
        self.transformer.named_parameters.return_value = []
        self.transformer.load_state_dict.return_value = ([], [])
        self.modules = {
            "torch": types.SimpleNamespace(device=MagicMock()),
            "diffusers": types.SimpleNamespace(
                Krea2Pipeline=types.SimpleNamespace(from_pretrained=self.loader)),
            "diffusers.models.transformers.transformer_krea2": types.SimpleNamespace(
                Krea2Transformer2DModel=MagicMock(return_value=self.transformer)),
            "safetensors.torch": types.SimpleNamespace(load_file=MagicMock(return_value={})),
        }

    def load(self, checkpoint):
        args = types.SimpleNamespace(repo=self.scope["REPO"], revision=None,
                                     checkpoint=checkpoint, device="cpu")
        with patch.dict("sys.modules", self.modules):
            return self.scope["build"](args, "bfloat16")

    def test_default_reference_uses_the_pinned_official_repository(self):
        pipe, provenance = self.load(None)
        self.loader.assert_called_once_with(
            self.scope["REPO"], revision=self.scope["REVISION"], dtype="bfloat16")
        self.assertIs(pipe, self.pipe)
        self.assertEqual(provenance["revision"], self.scope["REVISION"])

    def test_local_transformer_keeps_other_components_in_the_pinned_hf_repository(self):
        _, provenance = self.load("/anywhere/model.safetensors")
        self.loader.assert_called_once_with(
            self.scope["REPO"], revision=self.scope["REVISION"],
            dtype="bfloat16", transformer=self.transformer)
        self.assertEqual(provenance["checkpoint"], "/anywhere/model.safetensors")
        self.assertEqual(provenance["repo"], self.scope["REPO"])
        self.assertEqual(provenance["revision"], self.scope["REVISION"])
        self.pipe.vae.enable_tiling.assert_called_once()


if __name__ == "__main__":
    unittest.main()
