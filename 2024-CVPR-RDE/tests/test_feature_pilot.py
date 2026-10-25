"""Data-free checks for the optional TSE feature enhancements."""

import ast
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError:
    torch = None


def make_args(mode):
    return SimpleNamespace(
        loss_names="TAL", pretrain_choice="fake", img_size=(384, 128),
        stride_size=16, temperature=0.02, select_ratio=0.3,
        tse_enhancement=mode, adapter_bottleneck=16, adapter_scale=0.1,
        tau=0.015, margin=0.1)


class OptionTests(unittest.TestCase):
    def test_three_modes_and_defaults(self):
        from utils.options import get_args

        with patch.object(sys, "argv", ["train.py"]):
            args = get_args()
        self.assertEqual((args.tse_enhancement, args.adapter_bottleneck,
                          args.adapter_scale), ("none", 256, 0.1))
        self.assertEqual(args.init_checkpoint, "")
        for mode in ("none", "adapter", "delta_gate"):
            with patch.object(sys, "argv", ["train.py", "--tse_enhancement", mode,
                                            "--adapter_bottleneck", "32",
                                            "--adapter_scale", "0.2"]):
                args = get_args()
            self.assertEqual((args.tse_enhancement, args.adapter_bottleneck,
                              args.adapter_scale), (mode, 32, 0.2))

    def test_weight_only_initialization_precedes_optimizer(self):
        tree = ast.parse((PROJECT_ROOT / "train.py").read_text(encoding="utf-8"))
        calls = [(node.func.id if isinstance(node.func, ast.Name) else node.func.attr,
                  node.lineno) for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Name, ast.Attribute))]
        model_line = min(line for name, line in calls if name == "build_model")
        load_line = min(line for name, line in calls if name == "load")
        optimizer_line = min(line for name, line in calls if name == "build_optimizer")
        self.assertLess(model_line, load_line)
        self.assertLess(load_line, optimizer_line)

    def test_enhancer_optimizer_rule_is_present(self):
        tree = ast.parse((PROJECT_ROOT / "solver" / "build.py").read_text(encoding="utf-8"))
        enhancer_rules = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                          and isinstance(node.test, ast.Compare)
                          and isinstance(node.test.left, ast.Constant)
                          and node.test.left.value == "_tse_enhancer."]
        self.assertEqual(len(enhancer_rules), 1)
        self.assertTrue(any(isinstance(node, ast.Assign) and
                            any(isinstance(target, ast.Name) and target.id == "lr"
                                for target in node.targets) and
                            isinstance(node.value, ast.Constant) and node.value.value == 0.001
                            for node in enhancer_rules[0].body))


if torch is not None:
    class FakeClip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.width = torch.nn.Parameter(torch.linspace(0.5, 1.5, 512))

        def encode_image(self, image):
            tokens = image.unsqueeze(1).expand(-1, 4, -1) * self.width
            attention = torch.zeros(image.shape[0], 4, 4)
            return tokens, attention

        def encode_text(self, text):
            positions = torch.arange(text.shape[1]).view(1, -1, 1)
            tokens = (text.float().unsqueeze(-1) + positions).expand(-1, -1, 512)
            attention = torch.zeros(text.shape[0], text.shape[1], text.shape[1])
            return tokens * self.width, attention

        def forward(self, image, text):
            image_tokens, image_attention = self.encode_image(image)
            text_tokens, text_attention = self.encode_text(text)
            return image_tokens, image_attention, text_tokens, text_attention

    class FakeVisual(torch.nn.Module):
        embed_dim = 1024

        def __init__(self, ratio):
            super().__init__()

        def forward(self, features, attention):
            pooled = features.mean(dim=1)
            return torch.cat((pooled, pooled), dim=-1)

    class FakeText(FakeVisual):
        def forward(self, features, text, attention):
            pooled = features.mean(dim=1)
            return torch.cat((pooled, pooled), dim=-1)


@unittest.skipUnless(torch is not None, "PyTorch is required for tensor tests")
class TensorTests(unittest.TestCase):
    def setUp(self):
        import model.build as current

        self.current = current
        self.patches = [
            patch.object(current, "build_CLIP_from_openai_pretrained",
                         side_effect=lambda *args: (FakeClip(), {"embed_dim": 512})),
            patch.object(current, "VisualEmbeddingLayer", FakeVisual),
            patch.object(current, "TexualEmbeddingLayer", FakeText),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        torch.manual_seed(1)
        self.images = torch.randn(3, 512)
        self.text = torch.tensor([[1, 2, 9, 0, 0], [1, 8, 3, 0, 0],
                                  [1, 3, 7, 0, 0]], dtype=torch.long)
        self.batch = {"images": self.images, "caption_ids": self.text,
                      "pids": torch.tensor([0, 1, 2]),
                      "label_hat": torch.ones(3)}

    def test_modules_shape_gradients_dtype_and_parameter_counts(self):
        from model.tse_enhancement import TSEEnhancer

        expected = {"adapter": 527616, "delta_gate": 2108928}
        for mode in ("adapter", "delta_gate"):
            with self.subTest(mode=mode):
                module = TSEEnhancer(mode, 512, 1024, 256, 0.1)
                self.assertEqual(sum(p.numel() for p in module.parameters()), expected[mode])
                tse = torch.randn(3, 1024).half()
                bge = torch.randn(3, 512).half()
                output = module(tse, bge)
                self.assertEqual((output.shape, output.dtype), ((3, 1024), torch.float32))
                self.assertTrue(torch.isfinite(output).all().item())
                self.assertLess((output - tse.float()).abs().max().item(), 0.02)
                output.square().mean().backward()
                for name, parameter in module.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_optimizer_registers_new_parameters_with_tse_settings(self):
        from solver import build_optimizer
        from utils.options import get_args

        with patch.object(sys, "argv", ["train.py"]):
            args = get_args()
        for mode in ("adapter", "delta_gate"):
            with self.subTest(mode=mode):
                model = self.current.build_model(make_args(mode))
                optimizer = build_optimizer(args, model)
                groups = {id(group["params"][0]): group for group in optimizer.param_groups}
                named = dict(model.named_parameters())
                self.assertEqual(set(groups), {id(p) for p in named.values() if p.requires_grad})
                for name, parameter in named.items():
                    if "_tse_enhancer." not in name:
                        continue
                    group = groups[id(parameter)]
                    self.assertEqual(group["lr"], 0.001, name)
                    self.assertEqual(group["weight_decay"],
                                     args.weight_decay_bias if "bias" in name else args.weight_decay,
                                     name)
                    self.assertEqual(parameter.dtype, torch.float32, name)
                before = model.image_tse_enhancer.adapter.up.weight.detach().clone()
                optimizer.zero_grad()
                loss = (model.encode_image_tse(self.images).square().mean() +
                        model.encode_text_tse(self.text).square().mean())
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                optimizer.step()
                self.assertFalse(torch.equal(before,
                                             model.image_tse_enhancer.adapter.up.weight))

    def test_none_matches_baseline_commit_with_identical_weights(self):
        source = subprocess.run(
            ["git", "show", "83a538e:2024-CVPR-RDE/model/build.py"],
            cwd=str(REPO_ROOT), check=True, capture_output=True, text=True).stdout
        baseline = types.ModuleType("model._baseline_build_for_test")
        baseline.__package__ = "model"
        exec(compile(source, "baseline/model/build.py", "exec"), baseline.__dict__)
        legacy_args = make_args("none")
        del legacy_args.tse_enhancement  # Existing Baseline YAML has no new option.
        with patch.object(baseline, "build_CLIP_from_openai_pretrained",
                          side_effect=lambda *args: (FakeClip(), {"embed_dim": 512})), \
             patch.object(baseline, "VisualEmbeddingLayer", FakeVisual), \
             patch.object(baseline, "TexualEmbeddingLayer", FakeText):
            old = baseline.build_model(legacy_args)
        new = self.current.build_model(legacy_args)
        new.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(sum(p.numel() for p in new.parameters()),
                         sum(p.numel() for p in old.parameters()))
        for method, value in (("encode_image", self.images),
                              ("encode_text", self.text),
                              ("encode_image_tse", self.images),
                              ("encode_text_tse", self.text)):
            self.assertTrue(torch.equal(getattr(old, method)(value),
                                        getattr(new, method)(value)), method)

    def test_training_and_inference_use_same_enhancement(self):
        def surrogate_loss(i_bge, t_bge, i_tse, t_tse, pids, **kwargs):
            return ((i_bge - t_bge) ** 2).mean(), ((i_tse - t_tse) ** 2).mean()

        def surrogate_per_loss(image, text, pids, **kwargs):
            return ((image - text) ** 2).mean(dim=-1), (image * text).mean(dim=-1)

        old = self.current.build_model(make_args("none"))
        old_image = old.encode_image(self.images)
        old_text = old.encode_text(self.text)
        for mode, expected_extra in (("adapter", 2 * 35856),
                                     ("delta_gate", 2 * 633888)):
            with self.subTest(mode=mode):
                model = self.current.build_model(make_args(mode))
                model.load_state_dict(old.state_dict(), strict=False)
                self.assertEqual(sum(p.numel() for p in model.parameters()) -
                                 sum(p.numel() for p in old.parameters()), expected_extra)
                self.assertTrue(torch.equal(model.encode_image(self.images), old_image))
                self.assertTrue(torch.equal(model.encode_text(self.text), old_text))
                self.assertEqual(model.encode_image_tse(self.images).shape, (3, 1024))
                self.assertEqual(model.encode_text_tse(self.text).shape, (3, 1024))
                self.assertTrue(all(p.dtype == torch.float32
                                    for p in model.image_tse_enhancer.parameters()))
                with patch.object(self.current.objectives, "compute_per_loss",
                                  side_effect=surrogate_per_loss):
                    old_bge_loss, old_tse_loss, _, _ = old.compute_per_loss(self.batch)
                    bge_loss, tse_loss, _, _ = model.compute_per_loss(self.batch)
                self.assertTrue(torch.equal(bge_loss, old_bge_loss))
                self.assertFalse(torch.equal(tse_loss, old_tse_loss))
                with patch.object(self.current.objectives, "compute_rbs",
                                  side_effect=surrogate_loss):
                    total = sum(model(self.batch).values())
                total.backward()
                for name, parameter in model.named_parameters():
                    if "_tse_enhancer." in name:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

    def test_old_checkpoint_load_and_new_checkpoint_round_trip(self):
        from utils.checkpoint import Checkpointer

        old = self.current.build_model(make_args("none"))
        with tempfile.TemporaryDirectory() as scratch:
            old_file = Path(scratch) / "old.pth"
            torch.save({"model": old.state_dict()}, str(old_file))
            for mode in ("adapter", "delta_gate"):
                with self.subTest(mode=mode):
                    model = self.current.build_model(make_args(mode))
                    fresh = {key: value.clone() for key, value in model.state_dict().items()
                             if "_tse_enhancer." in key}
                    Checkpointer(model).load(f=str(old_file))
                    for key, value in old.state_dict().items():
                        self.assertTrue(torch.equal(model.state_dict()[key], value), key)
                    for key, value in fresh.items():
                        self.assertTrue(torch.equal(model.state_dict()[key], value), key)
                    with torch.no_grad():
                        next(model.image_tse_enhancer.parameters()).add_(0.1)
                    name = "new_" + mode
                    Checkpointer(model, save_dir=scratch, save_to_disk=True).save(name)
                    restored = self.current.build_model(make_args(mode))
                    Checkpointer(restored).load(f=str(Path(scratch) / (name + ".pth")))
                    for key, value in model.state_dict().items():
                        self.assertTrue(torch.equal(restored.state_dict()[key], value), key)


if __name__ == "__main__":
    unittest.main()
