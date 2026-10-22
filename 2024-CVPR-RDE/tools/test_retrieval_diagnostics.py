"""Small, data-free checks for the retrieval diagnostic tool."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_diagnostics import build_query_metadata, classify_query


class MetadataTests(unittest.TestCase):
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
        summary, rows, _ = compute_diagnostics(q_bge, g_bge, q_tse, g_tse,
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


if __name__ == "__main__":
    unittest.main()
