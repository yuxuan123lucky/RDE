"""Mechanism A checks: no data/weights required; tensor tests require PyTorch."""

import ast
import importlib.util
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import torch
except ImportError:
    torch = None


class OptionTests(unittest.TestCase):
    def test_default_is_original_rde_and_a_options_are_explicit(self):
        from utils.options import get_args

        with patch.object(sys, "argv", ["train.py"]):
            args = get_args()
        self.assertEqual(args.evidence_mode, "none")
        self.assertFalse(args.skip_final_test)
        self.assertEqual((args.evidence_candidates, args.evidence_weight,
                          args.evidence_temperature, args.evidence_margin), (4, 0.1, 0.07, 0.1))
        with patch.object(sys, "argv", ["train.py", "--evidence_mode", "a_only",
                                       "--noisy_rate", "0.0", "--evidence_candidates", "2"]):
            args = get_args()
        self.assertEqual((args.evidence_mode, args.evidence_candidates, args.noisy_rate), ("a_only", 2, 0.0))
        with patch.object(sys, "argv", ["train.py", "--evidence_mode", "off", "--skip_final_test"]):
            args = get_args()
        self.assertEqual(args.evidence_mode, "off")
        self.assertTrue(args.skip_final_test)

    def test_output_reservation_precedes_log_and_config_writes(self):
        tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
        reservations, writes = [], []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "makedirs":
                if any(kw.arg == "exist_ok" and isinstance(kw.value, ast.Constant)
                       and kw.value.value is False for kw in node.keywords):
                    reservations.append(node.lineno)
            if isinstance(node.func, ast.Name) and node.func.id in ("setup_logger", "save_train_configs"):
                writes.append(node.lineno)
        self.assertEqual(len(reservations), 1)
        self.assertLess(reservations[0], min(writes))

    def test_original_encoders_ccd_and_model_conversion_are_unchanged(self):
        for relative, names in (
                ("model/build.py", {"encode_image", "encode_text", "encode_image_tse",
                                    "encode_text_tse", "compute_per_loss", "build_model"}),
                ("processor/processor.py", {"get_loss", "split_prob"})):
            old = subprocess.run(["git", "show", "83a538e:2024-CVPR-RDE/" + relative],
                                 cwd=str(ROOT.parent), check=True, capture_output=True, text=True).stdout
            def functions(source):
                return {node.name: ast.dump(node, include_attributes=False)
                        for node in ast.walk(ast.parse(source))
                        if isinstance(node, ast.FunctionDef) and node.name in names}
            before = functions(old)
            self.assertEqual(set(before), names)
            self.assertEqual(before, functions((ROOT / relative).read_text(encoding="utf-8")), relative)

    def test_final_test_opt_out_preserves_original_test_block(self):
        original = subprocess.run(["git", "show", "83a538e:2024-CVPR-RDE/train.py"],
                                  cwd=str(ROOT.parent), check=True, capture_output=True, text=True).stdout
        def tail(source):
            main = next(node for node in ast.parse(source).body if isinstance(node, ast.If))
            train_index = next(i for i, node in enumerate(main.body)
                               if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                               and isinstance(node.value.func, ast.Name) and node.value.func.id == "do_train")
            return main.body[train_index + 1:]
        before = tail(original)
        after = tail((ROOT / "train.py").read_text(encoding="utf-8"))
        self.assertEqual(len(after), 1)
        guard = after[0]
        self.assertIsInstance(guard, ast.If)
        self.assertIsInstance(guard.test, ast.UnaryOp)
        self.assertIsInstance(guard.test.op, ast.Not)
        self.assertEqual(guard.test.operand.attr, "skip_final_test")
        self.assertEqual([ast.dump(node, include_attributes=False) for node in before],
                         [ast.dump(node, include_attributes=False) for node in guard.body])


@unittest.skipUnless(torch is not None, "PyTorch unavailable: mechanism A tensor/gradient checks NOT executed")
class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("standalone_local_evidence", ROOT / "model" / "local_evidence.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        torch.manual_seed(7)
        self.patches = torch.randn(4, 5, 8, requires_grad=True)
        self.text = torch.randn(4, 8, requires_grad=True)
        self.images = torch.randn(4, 8, requires_grad=True)
        self.pids = torch.tensor([0, 0, 1, 2])

    def call(self, **kwargs):
        return self.module.local_evidence_loss(self.patches, self.text, self.images,
                                               self.pids, **kwargs)

    def test_repeated_pids_all_positives_and_only_different_pid_rivals(self):
        result = self.call(candidates=3)
        self.assertTrue(result["positive_mask"][0, 1].item())
        self.assertTrue(result["positive_mask"][1, 0].item())
        for q in range(4):
            for k, index in enumerate(result["negative_indices"][q].tolist()):
                negative_columns = (result["candidate_mask"][q, k] &
                                    ~result["positive_mask"][q]).nonzero().flatten().tolist()
                if index < 0:
                    self.assertEqual(negative_columns, [])
                    self.assertFalse(result["candidate_mask"][q, k].any().item())
                    continue
                self.assertEqual(negative_columns, [index])
                self.assertNotEqual(q, index)
                self.assertNotEqual(int(self.pids[q]), int(self.pids[index]))
                positives = result["positive_mask"][q]
                self.assertTrue(result["candidate_mask"][q, k, positives].all().item())
        self.assertTrue(torch.isfinite(result["loss"]).item())
        result["loss"].backward()
        for value in (self.patches, self.text):
            self.assertTrue(torch.isfinite(value.grad).all().item())
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_hardness_selection_excludes_even_high_scoring_same_pid(self):
        images = torch.tensor([[1., 0.], [1., 0.], [0.8, 0.6], [-1., 0.]])
        text = torch.tensor([[1., 0.], [0.9, 0.1], [0., 1.], [-1., 0.]])
        result = self.module.local_evidence_loss(torch.randn(4, 3, 2), text, images, self.pids, candidates=1)
        self.assertEqual(int(result["negative_indices"][0, 0]), 2)

    def test_no_negatives_singleton_and_ccd_mask_return_differentiable_zero(self):
        for size in (1, 4):
            patches = self.patches[:size].detach().clone().requires_grad_()
            text = self.text[:size].detach().clone().requires_grad_()
            result = self.module.local_evidence_loss(patches, text, self.images[:size], torch.zeros(size, dtype=torch.long))
            self.assertEqual(result["loss"].item(), 0)
            self.assertFalse(result["valid_query_mask"].any().item())
            result["loss"].backward()
            self.assertEqual(patches.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(text.grad).all().item())
        masked = self.call(query_mask=torch.zeros(4))
        self.assertEqual(masked["loss"].item(), 0)
        self.assertFalse(masked["valid_relation_mask"].any().item())
        # A full reduction of finite elements can overflow; the zero-loss
        # anchor must remain finite without performing that full reduction.
        huge = torch.full((2, 3, 4), 1e38, requires_grad=True)
        result = self.module.local_evidence_loss(huge, torch.ones(2, 4), torch.zeros(2, 4),
                                               torch.zeros(2, dtype=torch.long))
        self.assertEqual(result["loss"].item(), 0)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(huge.grad).all().item())

    def test_no_valid_positive_or_negative_evidence_returns_zero(self):
        for zero_rows in ((0, 1, 2, 3), (0, 1), (2, 3)):
            patches = self.patches.detach().clone()
            patches[list(zero_rows)] = 0
            patches.requires_grad_()
            result = self.module.local_evidence_loss(patches, self.text, self.images, self.pids,
                                                   query_mask=torch.tensor([1, 0, 0, 0]))
            self.assertEqual(result["loss"].item(), 0)
            self.assertFalse(result["valid_relation_mask"].any().item())
            result["loss"].backward()
            self.assertTrue(torch.isfinite(patches.grad).all().item())

    def test_valid_query_mean_and_single_weight_application(self):
        patches = torch.tensor([[[1., 0.], [0., 1.]],
                                [[1., 0.], [0., 1.]],
                                [[-1., 0.], [0., 1.]]])
        text = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        images = text.clone()
        pids = torch.tensor([0, 0, 1])
        def run(mask, weight):
            return self.module.local_evidence_loss(patches, text, images, pids,
                                                   query_mask=torch.tensor(mask),
                                                   candidates=1, weight=weight)
        one = run([1, 0, 0], 0.1)
        two = run([1, 1, 0], 0.1)
        doubled_weight = run([1, 1, 0], 0.2)
        self.assertEqual(int(one["valid_query_count"]), 1)
        self.assertEqual(int(two["valid_query_count"]), 2)
        self.assertGreater(one["loss"].item(), 0)
        self.assertTrue(torch.allclose(one["loss"], two["loss"]))
        self.assertTrue(torch.allclose(doubled_weight["loss"], 2 * two["loss"]))
        self.assertTrue(torch.allclose(two["loss"], 0.1 * two["unweighted_loss"]))

    def test_identical_text_differences_do_not_fall_back_to_plain_alignment(self):
        result = self.module.local_evidence_loss(self.patches, self.text[:1].expand(4, -1),
                                               self.images, self.pids, return_attention=True)
        self.assertEqual(result["loss"].item(), 0)
        self.assertEqual(len(result["attention_weights"]), 0)
        near = torch.zeros(2, 8)
        near[:, 0] = 1
        near[1, 1] = 1e-5
        result = self.module.local_evidence_loss(self.patches[:2], near, self.images[:2], torch.arange(2))
        self.assertEqual(result["loss"].item(), 0)
        self.assertFalse(result["valid_relation_mask"].any().item())

    def test_rival_changes_attention_with_fixed_query_and_image(self):
        patches = torch.tensor([[[1., 0.], [0., 1.]]] * 3)
        image = torch.tensor([[1., 0.], [1., 0.], [-1., 0.]])
        text = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        first = self.module.local_evidence_loss(patches, text, image, torch.arange(3),
                                                candidates=1, return_attention=True)
        changed = text.clone()
        changed[1] = torch.tensor([0., -1.])
        second = self.module.local_evidence_loss(patches, changed, image, torch.arange(3),
                                                 candidates=1, return_attention=True)
        target = torch.tensor([0, 0, 0])
        a = first["attention_weights"][(first["pair_indices"] == target).all(-1)]
        b = second["attention_weights"][(second["pair_indices"] == target).all(-1)]
        self.assertEqual(a.shape, (1, 2))
        self.assertFalse(torch.allclose(a, b))
        self.assertEqual(int(first["negative_indices"][0, 0]), int(second["negative_indices"][0, 0]))

    def test_fp16_inputs_and_nonfinite_guards(self):
        result = self.module.local_evidence_loss(self.patches.half(), self.text.half(),
                                               self.images.half(), self.pids)
        self.assertEqual(result["loss"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(result["loss"]).item())
        bad = self.patches.detach().clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.module.local_evidence_loss(bad, self.text, self.images, self.pids)
        for kwargs in ({"weight": float("nan")}, {"temperature": 0}, {"candidates": 0}, {"margin": -1}):
            with self.assertRaises(ValueError):
                self.call(**kwargs)

    def test_attention_softmax_stays_finite_at_minimum_temperature(self):
        result = self.call(temperature=1e-4, return_attention=True)
        attention = result["attention_weights"]
        self.assertGreater(len(attention), 0)
        self.assertTrue(torch.isfinite(attention).all().item())
        self.assertTrue(torch.allclose(attention.sum(-1), torch.ones(len(attention)), atol=1e-6))
        self.assertTrue(torch.isfinite(result["loss"]).item())
        result["loss"].backward()
        self.assertTrue(torch.isfinite(self.patches.grad).all().item())
        self.assertTrue(torch.isfinite(self.text.grad).all().item())

    def test_loss_matches_explicit_all_positive_ranking_formula(self):
        result = self.call(candidates=2, weight=0.2, temperature=0.1, margin=0.15)
        query_terms = []
        for q in range(4):
            terms = []
            for k in range(2):
                if not result["valid_relation_mask"][q, k]:
                    continue
                n = int(result["negative_indices"][q, k])
                positives = result["positive_mask"][q].nonzero().flatten()
                scores = result["local_scores"][q, k]
                terms.append((0.1 * torch.nn.functional.softplus((0.15 + scores[n] - scores[positives]) / 0.1)).mean())
            if terms:
                query_terms.append(torch.stack(terms).mean())
        expected = torch.stack(query_terms).mean()
        self.assertTrue(torch.allclose(result["loss"], 0.2 * expected))

    def test_a_alone_updates_original_visual_and_text_encoder(self):
        from model.clip_model import CLIP

        clip = CLIP(embed_dim=8, image_resolution=(8, 8), vision_layers=1,
                    vision_width=64, vision_patch_size=4, stride_size=4,
                    context_length=5, vocab_size=32, transformer_width=64,
                    transformer_heads=1, transformer_layers=1)
        images = torch.randn(4, 3, 8, 8)
        captions = torch.tensor([[1, 2, 31, 0, 0], [1, 3, 31, 0, 0],
                                 [1, 4, 31, 0, 0], [1, 5, 31, 0, 0]])
        post_norm = []
        handle = clip.visual.ln_post.register_forward_hook(lambda module, args, output: post_norm.append(output))
        visual, _, text, _ = clip(images, captions)
        handle.remove()
        self.assertEqual(post_norm[0].shape[-1], 64)
        self.assertEqual(visual.shape[-1], 8)
        self.assertTrue(torch.allclose(visual, post_norm[0] @ clip.visual.proj))
        self.assertTrue(visual[:, 1:].requires_grad)
        visual.retain_grad()
        global_text = text[torch.arange(4), captions.argmax(-1)]
        before_bge = visual[:, 0].detach().clone()
        result = self.module.local_evidence_loss(visual[:, 1:], global_text, visual[:, 0], self.pids)
        optimizer = torch.optim.SGD(clip.parameters(), lr=0.01)
        optimizer.zero_grad()
        result["loss"].backward()  # No RDE/TAL loss participates in this check.
        self.assertGreater(visual.grad[:, 1:].abs().sum().item(), 0)
        self.assertEqual(visual.grad[:, 0].abs().sum().item(), 0)
        for parameter in (clip.visual.conv1.weight, clip.visual.proj,
                          clip.token_embedding.weight, clip.text_projection):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        optimizer.step()
        with torch.no_grad():
            after_bge, _ = clip.encode_image(images)
        self.assertFalse(torch.equal(before_bge, after_bge[:, 0]))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),
                         "CUDA unavailable: original fp16 encoder-gradient check NOT executed")
    def test_original_fp16_cuda_encoder_gets_finite_a_gradients(self):
        from model.clip_model import CLIP, convert_weights

        clip = CLIP(8, (8, 8), 1, 64, 4, 4, 5, 32, 64, 1, 1)
        convert_weights(clip)
        clip = clip.cuda()
        images = torch.randn(4, 3, 8, 8, device="cuda")
        captions = torch.tensor([[1, 2, 31, 0, 0], [1, 3, 31, 0, 0],
                                 [1, 4, 31, 0, 0], [1, 5, 31, 0, 0]], device="cuda")
        visual, _, text, _ = clip(images, captions)
        global_text = text[torch.arange(4, device="cuda"), captions.argmax(-1)]
        result = self.module.local_evidence_loss(visual[:, 1:], global_text,
                                               visual[:, 0], self.pids.cuda())
        result["loss"].backward()
        for parameter in (clip.visual.conv1.weight, clip.visual.proj,
                          clip.token_embedding.weight, clip.text_projection):
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


@unittest.skipUnless(torch is not None, "PyTorch unavailable: RDE integration/parity checks NOT executed")
class RDEIntegrationTests(unittest.TestCase):
    def test_baseline_parity_training_only_path_and_checkpoint_schema(self):
        import model.build as current
        from model.clip_model import CLIP

        def make_clip(*unused):
            return CLIP(512, (8, 8), 1, 64, 4, 4, 5, 32, 64, 1, 1), {"embed_dim": 512}

        class Pool(torch.nn.Module):
            def __init__(self, ratio):
                super().__init__()

            def forward(self, tokens, *unused):
                pooled = tokens.mean(1)
                return torch.cat((pooled, pooled), -1)

        args = SimpleNamespace(loss_names="TAL", pretrain_choice="unused", img_size=(8, 8),
                               stride_size=4, temperature=0.02, select_ratio=0.3,
                               tau=0.015, margin=0.1, noisy_rate=0.0)
        source = subprocess.run(["git", "show", "83a538e:2024-CVPR-RDE/model/build.py"],
                                cwd=str(ROOT.parent), capture_output=True, text=True, check=True).stdout
        baseline = types.ModuleType("model._baseline_for_evidence_test")
        baseline.__package__ = "model"
        exec(compile(source, "baseline/model/build.py", "exec"), baseline.__dict__)
        def surrogate(i, t, vi, vt, *unused, **kwargs):
            return (i - t).square().mean(), (vi - vt).square().mean()
        batch = {"images": torch.randn(4, 3, 8, 8),
                 "caption_ids": torch.tensor([[1, 2, 31, 0, 0], [1, 3, 31, 0, 0],
                                               [1, 4, 31, 0, 0], [1, 5, 31, 0, 0]]),
                 "pids": torch.tensor([0, 0, 1, 2]), "label_hat": torch.ones(4)}
        patches = []
        for module in (baseline, current):
            patches.extend([patch.object(module, "build_CLIP_from_openai_pretrained", side_effect=make_clip),
                            patch.object(module, "VisualEmbeddingLayer", Pool),
                            patch.object(module, "TexualEmbeddingLayer", Pool),
                            patch.object(module, "convert_weights", side_effect=lambda model: None)])
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        old, off = baseline.build_model(args), current.build_model(args)
        off.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(set(old.state_dict()), set(off.state_dict()))
        with patch.object(current, "local_evidence_loss", side_effect=AssertionError("A called while disabled")), \
             patch.object(current.objectives, "compute_rbs", side_effect=surrogate):
            old_ret, off_ret = old(batch), off(batch)
            for key in old_ret:
                self.assertTrue(torch.equal(old_ret[key], off_ret[key]), key)
            for method, value in (("encode_image", batch["images"]), ("encode_text", batch["caption_ids"]),
                                  ("encode_image_tse", batch["images"]), ("encode_text_tse", batch["caption_ids"])):
                self.assertTrue(torch.equal(getattr(old, method)(value), getattr(off, method)(value)), method)
            for mode in ("none", "off"):
                alias = current.build_model(SimpleNamespace(**vars(args), evidence_mode=mode))
                alias.load_state_dict(old.state_dict(), strict=True)
                self.assertEqual(alias.evidence_mode, "none")
                alias_ret = alias(batch)
                for key in old_ret:
                    self.assertTrue(torch.equal(old_ret[key], alias_ret[key]), (mode, key))
        on_args = SimpleNamespace(**vars(args), evidence_mode="a_only")
        on = current.build_model(on_args)
        on.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(set(on.state_dict()), set(old.state_dict()))
        self.assertEqual(sum(p.numel() for p in on.parameters()), sum(p.numel() for p in old.parameters()))
        def zero_rde_losses(i, t, vi, vt, *unused, **kwargs):
            return i.sum() * 0, vi.sum() * 0
        with patch.object(current.objectives, "compute_rbs", side_effect=zero_rde_losses), \
             patch.object(on.base_model, "forward", wraps=on.base_model.forward) as encoded:
            on.train()
            result = on(batch, return_evidence=True)
            self.assertEqual(encoded.call_count, 1)
            total = sum(v for key, v in result.items() if "loss" in key)
            self.assertTrue(torch.equal(total, result["evidence_a_loss"]))
            self.assertTrue(torch.allclose(total, 0.1 * result["evidence_relations"]["unweighted_loss"]))
            total.backward()
            self.assertGreater(on.base_model.visual.conv1.weight.grad.abs().sum().item(), 0)
        on.eval()
        with patch.object(current, "local_evidence_loss", side_effect=AssertionError("A called during inference")):
            for method, value in (("encode_image", batch["images"]), ("encode_text", batch["caption_ids"]),
                                  ("encode_image_tse", batch["images"]), ("encode_text_tse", batch["caption_ids"])):
                with torch.no_grad():
                    self.assertTrue(torch.equal(getattr(old.eval(), method)(value), getattr(on, method)(value)))
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / "weights.pth"
            torch.save({"model": on.state_dict()}, str(filename))
            restored = current.build_model(on_args)
            restored.load_state_dict(torch.load(str(filename), map_location="cpu")["model"], strict=True)
            for key, value in on.state_dict().items():
                self.assertTrue(torch.equal(restored.state_dict()[key], value), key)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),
                         "CUDA unavailable: real TAL/TSE Baseline parity check NOT executed")
    def test_off_matches_original_tal_tse_and_ccd_values_on_cuda(self):
        import model.build as current
        from model.clip_model import CLIP

        source = subprocess.run(["git", "show", "83a538e:2024-CVPR-RDE/model/build.py"],
                                cwd=str(ROOT.parent), capture_output=True, text=True, check=True).stdout
        baseline = types.ModuleType("model._baseline_cuda_evidence_test")
        baseline.__package__ = "model"
        exec(compile(source, "baseline/model/build.py", "exec"), baseline.__dict__)
        def make_clip(*unused):
            return CLIP(512, (8, 8), 1, 64, 4, 4, 77, 32, 64, 1, 1), {"embed_dim": 512}
        args = SimpleNamespace(loss_names="TAL", pretrain_choice="unused", img_size=(8, 8),
                               stride_size=4, temperature=0.02, select_ratio=0.3,
                               tau=0.015, margin=0.1, noisy_rate=0.0)
        with patch.object(baseline, "build_CLIP_from_openai_pretrained", side_effect=make_clip), \
             patch.object(current, "build_CLIP_from_openai_pretrained", side_effect=make_clip):
            old = baseline.build_model(args).cuda()
            off = current.build_model(SimpleNamespace(**vars(args), evidence_mode="off")).cuda()
        off.load_state_dict(old.state_dict(), strict=True)
        captions = torch.zeros(4, 77, dtype=torch.long, device="cuda")
        captions[:, 0], captions[:, 1], captions[:, 2] = 1, torch.arange(2, 6, device="cuda"), 31
        batch = {"images": torch.randn(4, 3, 8, 8, device="cuda"), "caption_ids": captions,
                 "pids": torch.tensor([0, 0, 1, 2], device="cuda"), "label_hat": torch.ones(4, device="cuda")}
        with patch.object(torch.backends.cudnn, "deterministic", True), \
             patch.object(torch.backends.cudnn, "benchmark", False), \
             patch.object(current, "local_evidence_loss", side_effect=AssertionError("off mode invoked A")), \
             torch.no_grad():
            old_ret, off_ret = old.train()(batch), off.train()(batch)
            for key in old_ret:
                self.assertTrue(torch.equal(old_ret[key], off_ret[key]), key)
            old.eval()
            off.eval()
            for left, right in zip(old.compute_per_loss(batch), off.compute_per_loss(batch)):
                self.assertTrue(torch.equal(left, right))
            for method, value in (("encode_image", batch["images"]), ("encode_text", captions),
                                  ("encode_image_tse", batch["images"]), ("encode_text_tse", captions)):
                self.assertTrue(torch.equal(getattr(old, method)(value), getattr(off, method)(value)), method)


if __name__ == "__main__":
    unittest.main()
