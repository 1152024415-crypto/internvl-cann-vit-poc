import tempfile
import unittest
from pathlib import Path

import numpy as np


class OnnxPairEvalTests(unittest.TestCase):
    def test_load_labels_csv_accepts_header_and_relative_paths(self):
        from internvl_vit_poc.onnx_pair_eval import load_labels_csv

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            path.write_text("image,label\nclass_a/a.jpg,3\nb.jpg,7\n", encoding="utf-8")

            labels = load_labels_csv(path)

            self.assertEqual(labels["class_a/a.jpg"], 3)
            self.assertEqual(labels["b.jpg"], 7)

    def test_top1_prediction_uses_flattened_logits(self):
        from internvl_vit_poc.onnx_pair_eval import top1_prediction

        logits = np.array([[0.1, 2.0, -4.0]], dtype=np.float32)

        self.assertEqual(top1_prediction(logits), 1)

    def test_summarize_pair_records_reports_accuracy_and_similarity(self):
        from internvl_vit_poc.onnx_pair_eval import summarize_pair_records

        records = [
            {
                "label": 1,
                "fp32_pred": 1,
                "int8_pred": 1,
                "same_pred": True,
                "compare": {"cosine": 0.99, "mean_abs_diff": 0.1, "max_abs_diff": 0.3},
            },
            {
                "label": 2,
                "fp32_pred": 2,
                "int8_pred": 3,
                "same_pred": False,
                "compare": {"cosine": 0.95, "mean_abs_diff": 0.2, "max_abs_diff": 0.8},
            },
        ]

        summary = summarize_pair_records(records)

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["label_count"], 2)
        self.assertAlmostEqual(summary["fp32_accuracy"], 1.0)
        self.assertAlmostEqual(summary["int8_accuracy"], 0.5)
        self.assertAlmostEqual(summary["same_pred_rate"], 0.5)
        self.assertAlmostEqual(summary["cosine_min"], 0.95)
        self.assertAlmostEqual(summary["cosine_mean"], 0.97)
        self.assertAlmostEqual(summary["mean_abs_diff_mean"], 0.15)
        self.assertAlmostEqual(summary["max_abs_diff_max"], 0.8)

    def test_summarize_pair_records_keeps_prediction_rate_empty_for_embeddings(self):
        from internvl_vit_poc.onnx_pair_eval import summarize_pair_records

        records = [
            {
                "compare": {"cosine": 0.99, "mean_abs_diff": 0.1, "max_abs_diff": 0.3},
            }
        ]

        summary = summarize_pair_records(records)

        self.assertIsNone(summary["fp32_accuracy"])
        self.assertIsNone(summary["int8_accuracy"])
        self.assertIsNone(summary["same_pred_rate"])


if __name__ == "__main__":
    unittest.main()
