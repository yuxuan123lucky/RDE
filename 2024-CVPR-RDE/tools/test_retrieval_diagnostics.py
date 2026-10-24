"""Small, data-free checks for the retrieval diagnostic tool."""

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_diagnostics import (build_query_metadata, classify_query, parse_args,
                                   select_split, set_seed, summarize_outcomes)


class SeedTests(unittest.TestCase):
    def test_seed_initialization_matches_train_order(self):
        calls = []
        cudnn = SimpleNamespace(deterministic=False, benchmark=False)
        fake_torch = SimpleNamespace(
            manual_seed=lambda value: calls.append(("torch", value)),
            cuda=SimpleNamespace(
                manual_seed=lambda value: calls.append(("cuda", value)),
                manual_seed_all=lambda value: calls.append(("cuda_all", value))),
            backends=SimpleNamespace(cudnn=cudnn))
        fake_numpy = SimpleNamespace(random=SimpleNamespace(
            seed=lambda value: calls.append(("numpy", value))))
        with patch.dict(sys.modules, {"torch": fake_torch, "numpy": fake_numpy}):
            with patch("random.seed", side_effect=lambda value: calls.append(("random", value))):
                set_seed(7)
        self.assertEqual(calls, [(name, 7) for name in
                                 ("torch", "cuda", "cuda_all", "numpy", "random")])
        self.assertTrue(cudnn.deterministic)
        self.assertTrue(cudnn.benchmark)

    def test_cli_default_and_override_seed(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            root = base / "datasets"
            run = base / "checkpoint_run"
            root.mkdir()
            run.mkdir()
            checkpoint = run / "best.pth"
            checkpoint.touch()
            (run / "configs.yaml").touch()
            arguments = ["--dataset_name", "RSTPReid", "--root_dir", str(root),
                         "--checkpoint", str(checkpoint), "--output_dir", str(base / "diagnostics")]
            self.assertEqual(parse_args(arguments).seed, 1)
            self.assertEqual(parse_args(arguments + ["--seed", "7"]).seed, 7)

    def test_seed_call_precedes_model_and_feature_extraction(self):
        source = Path(__file__).with_name("retrieval_diagnostics.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        run = next(node for node in module.body
                   if isinstance(node, ast.FunctionDef) and node.name == "run")
        calls = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr: node.lineno
                 for node in ast.walk(run) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertLess(calls["set_seed"], calls["build_dataloader"])
        self.assertLess(calls["set_seed"], calls["build_model"])
        self.assertLess(calls["set_seed"], calls["_compute_embedding"])


class MetadataTests(unittest.TestCase):
    def test_val_selection_never_falls_back_to_test(self):
        val = {"image_pids": [1], "caption_pids": [1]}
        test = {"image_pids": [2], "caption_pids": [2]}
        dataset = SimpleNamespace(val_annos=[{"id": 1}], val=val,
                                  test_annos=[{"id": 2}], test=test)
        self.assertEqual(select_split(dataset, "val"), (dataset.val_annos, val))
        self.assertEqual(select_split(dataset, "test"), (dataset.test_annos, test))
        dataset.val_annos = []
        with self.assertRaisesRegex(ValueError, "val split has no"):
            select_split(dataset, "val")

    def test_multiple_captions_and_same_identity_images(self):
        annos = [
            {"id": 7, "file_path": "a.jpg", "captions": ["red coat", "black shoes"]},
            {"id": 7, "file_path": "b.jpg", "captions": ["blue bag"]},
        ]
        images = SimpleNamespace(image_pids=[7, 7], img_paths=["root/a.jpg", "root/b.jpg"])
        texts = SimpleNamespace(caption_pids=[7, 7, 7],
                                captions=["red coat", "black shoes", "blue bag"])
        rows = build_query_metadata(annos, images, texts, "root", "file_path")
        self.assertEqual([row["source_image_index"] for row in rows], [0, 0, 1])
        self.assertEqual([row["pid"] for row in rows], [7, 7, 7])

    def test_annotation_order_mismatch_fails(self):
        annos = [{"id": 1, "img_path": "a.jpg", "captions": ["first"]}]
        images = SimpleNamespace(image_pids=[1], img_paths=["root/a.jpg"])
        texts = SimpleNamespace(caption_pids=[1], captions=["different"])
        with self.assertRaisesRegex(ValueError, "caption order"):
            build_query_metadata(annos, images, texts, "root", "img_path")

    def test_fusion_outcome_categories(self):
        self.assertEqual(classify_query(False, False, True), "fusion_corrects_both")
        self.assertEqual(classify_query(True, False, False), "fusion_introduces_error")
        self.assertEqual(classify_query(False, True, True), "fusion_preserves_single_success")
        self.assertEqual(classify_query(False, False, False), "all_wrong")

    def test_all_top1_outcomes_and_fusion_losses(self):
        from itertools import product

        branch_outcomes, complementarity = summarize_outcomes(product((False, True), repeat=3))
        self.assertEqual(branch_outcomes, {
            "both_correct": 2, "only_bge_correct": 2,
            "only_tse_correct": 2, "both_wrong": 2})
        self.assertEqual(complementarity["bge_correct_fusion_wrong"], 2)
        self.assertEqual(complementarity["tse_correct_fusion_wrong"], 2)
        self.assertEqual(complementarity["fusion_corrects_both"], 1)
        self.assertEqual(complementarity["fusion_corrects_at_least_one"], 3)
        self.assertEqual(complementarity["fusion_introduces_error"], 3)


class MetricParityTests(unittest.TestCase):
    def test_chunked_metrics_match_original_rank_with_multiple_positives(self):
        try:
            import torch
            import torch.nn.functional as F
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from utils.metrics import rank
            from retrieval_diagnostics import compute_diagnostics
        except ImportError as exc:
            self.skipTest(f"server-side dependencies unavailable locally: {exc}")

        gids = torch.tensor([1, 2, 1, 3, 2, 1, 3, 2, 3, 1])
        qids = torch.tensor([1, 2, 3])
        q_bge = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        q_tse = torch.tensor([[0.7, 0.3], [0.3, 0.7], [-0.8, 0.2]])
        g_bge = torch.tensor([[1., 0.], [0., 1.], [0.8, 0.2], [-1., 0.],
                              [0.2, 0.8], [0.9, 0.1], [-0.8, 0.2],
                              [0.1, 0.9], [-0.9, 0.1], [0.7, 0.3]])
        g_tse = torch.flip(g_bge, dims=[1])
        queries = [{"query_index": i, "pid": int(pid), "text": f"query {i}",
                    "source_image_index": i, "source_image_path": f"image_{i}.jpg"}
                   for i, pid in enumerate(qids)]
        paths = [f"image_{i}.jpg" for i in range(len(gids))]
        summary, rows, comparisons = compute_diagnostics(q_bge, g_bge, q_tse, g_tse,
                                                           qids, gids, paths, queries, 10, 1)
        sb = F.normalize(q_bge, dim=1) @ F.normalize(g_bge, dim=1).t()
        st = F.normalize(q_tse, dim=1) @ F.normalize(g_tse, dim=1).t()
        for name, scores in (("BGE", sb), ("TSE", st), ("BGE+TSE", (sb + st) / 2)):
            cmc, ap, inp, _ = rank(scores, qids, gids, max_rank=10, get_mAP=True)
            expected = (cmc[0], cmc[4], cmc[9], ap, inp)
            for key, value in zip(("R1", "R5", "R10", "mAP", "mINP"), expected):
                self.assertAlmostEqual(summary["metrics_percent"][name][key],
                                       float(value), places=4)
        for row in rows:
            self.assertTrue(all(item["is_positive"] == (item["pid"] == row["pid"])
                                for item in json.loads(row["fused_topk"])))
        self.assertEqual(sum(summary["branch_outcomes"].values()), len(qids))
        self.assertEqual(summary["complementarity"]["bge_correct_fusion_wrong"],
                         sum(row["bge_correct"] and not row["fused_correct"]
                             for row in comparisons))
        self.assertEqual(summary["complementarity"]["tse_correct_fusion_wrong"],
                         sum(row["tse_correct"] and not row["fused_correct"]
                             for row in comparisons))


if __name__ == "__main__":
    unittest.main()
