import unittest


class OnnxVerifyTests(unittest.TestCase):
    def test_compare_tensors_reports_error_metrics(self):
        import numpy as np

        from internvl_vit_poc.onnx_verify import compare_tensors

        actual = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        expected = np.array([[1.0, 1.5, 2.0]], dtype=np.float32)

        metrics = compare_tensors(actual, expected)

        self.assertEqual(metrics["shape"], [1, 3])
        self.assertEqual(metrics["reference_shape"], [1, 3])
        self.assertAlmostEqual(metrics["max_abs_diff"], 1.0)
        self.assertAlmostEqual(metrics["mean_abs_diff"], 0.5)
        self.assertGreater(metrics["cosine"], 0.99)

    def test_summarize_tensor_records_shape_and_stats(self):
        import numpy as np

        from internvl_vit_poc.onnx_verify import summarize_tensor

        tensor = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        summary = summarize_tensor(tensor)

        self.assertEqual(summary["shape"], [1, 3])
        self.assertEqual(summary["dtype"], "float32")
        self.assertAlmostEqual(summary["mean"], 2.0)
        self.assertAlmostEqual(summary["std"], float(np.std(tensor)))


if __name__ == "__main__":
    unittest.main()
