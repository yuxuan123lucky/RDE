"""Data-free tests; tensor/metric parity tests require the existing RDE environment."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import analyze_branch_complementarity as analysis

sys.path.insert(0, str(analysis.PROJECT_ROOT))
try:
    import torch
except ImportError:
    torch = None


def row(pid, bge, tse, fused, b_index=0, t_index=1):
    return {"pid": pid, "bge_top1_pid": bge, "tse_top1_pid": tse,
            "fused_top1_pid": fused, "bge_top1_index": b_index, "tse_top1_index": t_index}


class ProtocolTests(unittest.TestCase):
    def test_cli_defaults_to_val_and_reference_settings(self):
        args = analysis.parse_args(["--dataset_name", "CUHK-PEDES", "--root_dir", "data",
                                    "--checkpoint", "best.pth"])
        self.assertEqual((args.split, args.seed, args.score_chunk_size), ("val", 1, 0))
        self.assertIsNone(args.batch_size)

    def test_weight_grid_is_val_only(self):
        for name in ("CUHK-PEDES", "RSTPReid"):
            self.assertEqual(analysis.analysis_weights(name, "val"), (0.25, 0.5, 0.75))
        for name in analysis.DATASETS:
            self.assertEqual(analysis.analysis_weights(name, "test"), (0.5,))
        with self.assertRaisesRegex(ValueError, "no independent Val"):
            analysis.analysis_weights("ICFG-PEDES", "val")

    def test_real_val_without_test_fallback(self):
        val = {"image_pids": [2], "img_paths": ["val.jpg"],
               "caption_pids": [2, 2], "captions": ["a", "b"]}
        dataset = SimpleNamespace(val=val, val_annos=[{"id": 2, "split": "val"}],
                                  train_annos=[{"id": 1}], test_annos=[{"id": 3}])
        self.assertIs(analysis.select_split(dataset, "val"), val)
        dataset.val_annos[0]["split"] = "unknown"
        with self.assertRaisesRegex(ValueError, "split labels"):
            analysis.select_split(dataset, "val")
        dataset.val_annos = []
        with self.assertRaisesRegex(ValueError, "no fallback"):
            analysis.select_split(dataset, "val")

    def test_val_identity_overlap_is_rejected(self):
        data = {"image_pids": [1], "img_paths": ["val.jpg"],
                "caption_pids": [1], "captions": ["a"]}
        dataset = SimpleNamespace(val=data, val_annos=[{"id": 1, "split": "val"}],
                                  train_annos=[{"id": 1}], test_annos=[])
        with self.assertRaisesRegex(ValueError, "overlap train"):
            analysis.select_split(dataset, "val")

    def test_complementarity_counts_and_multiple_positive_identity(self):
        result = analysis.summarize_predictions([
            row(1, 1, 1, 1),  # Different images, same correct identity.
            row(2, 2, 3, 3), row(3, 4, 3, 3), row(4, 5, 6, 4)])
        expected = {"bge_correct": 2, "tse_correct": 2, "fused_correct": 3,
                    "both_correct": 1, "only_bge_correct": 1, "only_tse_correct": 1,
                    "both_wrong": 1, "top1_pid_disagreement": 3,
                    "top1_image_disagreement": 4, "fusion_loses_any_correct": 1,
                    "fusion_corrects_both_wrong": 1}
        for key, value in expected.items():
            self.assertEqual(result[key]["count"], value, key)
            self.assertEqual(result[key]["fraction"], value / 4, key)
        self.assertEqual(result["oracle_branch_selector"]["R1_percent"], 75)
        self.assertEqual(result["oracle_branch_selector"]["gain_over_best_single_branch_pp"], 25)

    def test_branch_selector_is_not_a_fusion_upper_bound(self):
        result = analysis.summarize_predictions([row(1, 2, 3, 1)])
        self.assertEqual(result["oracle_branch_selector"]["R1_percent"], 0)
        self.assertEqual(result["fused_correct"]["percent"], 100)
        self.assertTrue(result["oracle_branch_selector"]["analysis_only"])

    def test_checkpoint_prefix_and_architecture_checks(self):
        shape = lambda *dims: SimpleNamespace(shape=dims)
        expected = {"base.weight": shape(2, 2), "image_tse_enhancer.adapter.up.weight": shape(2, 1)}
        prefixed = {"module." + k: v for k, v in expected.items()}
        self.assertEqual(set(analysis.check_checkpoint_state(expected, prefixed)), set(expected))
        for saved in ({"base.weight": shape(2, 2)},
                      {**expected, "extra.weight": shape(1)},
                      {**expected, "base.weight": shape(3, 2)}):
            with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                analysis.check_checkpoint_state(expected, saved)

    def test_output_is_new_and_confined_to_dataset_outputs(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "data").mkdir()
            (root / "training").mkdir()
            checkpoint = root / "training" / "best.pth"
            checkpoint.write_bytes(b"test fixture")
            config = root / "training" / "configs.yaml"
            config.write_text("test fixture", encoding="utf-8")
            def args(output=None):
                return SimpleNamespace(root_dir=root / "data", checkpoint=checkpoint,
                                       config_file=config, dataset_name="CUHK-PEDES",
                                       split="val", output_dir=output)
            with patch.object(analysis, "REPO_ROOT", root):
                candidate = args()
                analysis.prepare_paths(candidate)
                self.assertIn(root / "outputs" / "CUHK-PEDES", candidate.output_dir.parents)
                self.assertFalse(candidate.output_dir.exists())  # Validation itself writes nothing.
                candidate.output_dir.mkdir(parents=True)
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    analysis.prepare_paths(candidate)
                with self.assertRaisesRegex(ValueError, "NEW subdirectory"):
                    analysis.prepare_paths(args(root / "outside"))
                with self.assertRaisesRegex(ValueError, "NEW subdirectory"):
                    analysis.prepare_paths(args(root / "outputs" / "RSTPReid" / "wrong_dataset"))
                run = root / "outputs" / "CUHK-PEDES" / "training_run"
                run.mkdir()
                (run / "best.pth").write_bytes(b"another test fixture")
                protected = args(run / "analysis")
                protected.checkpoint = run / "best.pth"
                with self.assertRaisesRegex(ValueError, "checkpoint directory"):
                    analysis.prepare_paths(protected)
            self.assertEqual(checkpoint.read_bytes(), b"test fixture")


@unittest.skipUnless(torch is not None, "PyTorch unavailable; tensor/metric tests NOT executed")
class MetricParityTests(unittest.TestCase):
    def test_all_weights_match_original_rank_and_query_pids(self):
        import torch.nn.functional as F
        from utils.metrics import rank

        torch.manual_seed(1)
        qb, gb, qt, gt = (torch.randn(4, 7), torch.randn(12, 7),
                          torch.randn(4, 9), torch.randn(12, 9))
        qids = torch.tensor([1, 2, 3, 4])
        gids = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        data = {"image_pids": gids.tolist(), "caption_pids": qids.tolist(),
                "img_paths": [str(i) + ".jpg" for i in range(12)],
                "captions": ["query " + str(i) for i in range(4)]}
        weights = (0.25, 0.5, 0.75)
        metrics, _, rows = analysis.analyze_features(qb, gb, qt, gt, qids, gids, data, weights)
        b = F.normalize(qb, dim=1) @ F.normalize(gb, dim=1).t()
        t = F.normalize(qt, dim=1) @ F.normalize(gt, dim=1).t()
        scores = {"BGE": b, "TSE": t, "fusion_0.25": 0.25 * b + 0.75 * t,
                  "fusion_0.50": (b + t) / 2, "fusion_0.75": 0.75 * b + 0.25 * t}
        for name, values in scores.items():
            cmc, ap, inp, indices = rank(values, qids, gids)
            expected = (cmc[0], cmc[4], cmc[9], ap, inp)
            for key, value in zip(("R1", "R5", "R10", "mAP", "mINP"), expected):
                self.assertEqual(metrics[name][key], float(value), (name, key))
            if name == "fusion_0.50":
                for i, result in enumerate(rows):
                    self.assertEqual(result["fused_top1_index"], int(indices[i, 0]))
                    self.assertEqual(result["fused_top1_correct"], int(gids[indices[i, 0]]) == int(qids[i]))
        chunked, _, _ = analysis.analyze_features(qb, gb, qt, gt, qids, gids, data, weights, 3)
        for name in metrics:
            for key in metrics[name]:
                self.assertAlmostEqual(metrics[name][key], chunked[name][key], delta=2e-5)


if __name__ == "__main__":
    unittest.main()
