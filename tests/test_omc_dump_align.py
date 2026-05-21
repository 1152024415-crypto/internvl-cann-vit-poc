import tempfile
import unittest
from pathlib import Path

import numpy as np


class OmcDumpAlignTests(unittest.TestCase):
    def test_inventory_dump_files_records_sizes_and_candidate_elements(self):
        from internvl_vit_poc.omc_dump_align import inventory_dump_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            output = nested / "output_0"
            np.array([1.0, 2.0, 3.0, 4.0], dtype="<f4").tofile(output)
            (root / "log.txt").write_text("dump log\n", encoding="utf-8")

            result = inventory_dump_files(root)

            self.assertEqual(result["count"], 2)
            records = {record["relative_path"]: record for record in result["records"]}
            self.assertEqual(records["nested/output_0"]["relative_path"], "nested/output_0")
            self.assertEqual(records["nested/output_0"]["byte_count"], 16)
            self.assertEqual(records["nested/output_0"]["fp32_element_count"], 4)
            self.assertEqual(records["nested/output_0"]["fp16_element_count"], 8)

    def test_inventory_dump_files_rejects_nonexistent_root(self):
        from internvl_vit_poc.omc_dump_align import inventory_dump_files

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"

            with self.assertRaisesRegex(FileNotFoundError, "OMC dump directory does not exist"):
                inventory_dump_files(missing)

    def test_inventory_dump_files_rejects_regular_file_root(self):
        from internvl_vit_poc.omc_dump_align import inventory_dump_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            regular_file = root / "dump.bin"
            regular_file.write_bytes(b"not a directory")

            with self.assertRaisesRegex(NotADirectoryError, "OMC dump path is not a directory"):
                inventory_dump_files(regular_file)

    def test_compare_indexed_outputs_maps_output_files_by_onnx_order(self):
        from internvl_vit_poc.onnx_intermediate_debug import save_outputs_npz
        from internvl_vit_poc.omc_dump_align import compare_indexed_outputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx_npz = root / "onnx_outputs.npz"
            save_outputs_npz(
                onnx_npz,
                {
                    "visual_features": np.array([1.0, 2.0], dtype=np.float32),
                    "encoder.layers.0.output": np.array([3.0, 4.0], dtype=np.float32),
                },
            )
            om_dir = root / "om_outputs"
            om_dir.mkdir()
            np.array([1.0, 2.0], dtype="<f4").tofile(om_dir / "output_0")
            np.array([4.0, 3.0], dtype="<f4").tofile(om_dir / "output_1")

            result = compare_indexed_outputs(
                onnx_npz=onnx_npz,
                om_output_dir=om_dir,
                cosine_threshold=0.9999,
            )

            self.assertEqual(result["summary"]["count"], 2)
            self.assertEqual(result["summary"]["ok_count"], 2)
            self.assertEqual(result["records"][0]["onnx_name"], "visual_features")
            self.assertEqual(result["records"][0]["om_output"], str(om_dir / "output_0"))
            self.assertLess(result["records"][1]["compare"]["cosine"], 0.99)
            self.assertEqual(result["first_bad"]["index"], 1)

    def test_compare_indexed_outputs_reports_missing_output_file(self):
        from internvl_vit_poc.onnx_intermediate_debug import save_outputs_npz
        from internvl_vit_poc.omc_dump_align import compare_indexed_outputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx_npz = root / "onnx_outputs.npz"
            save_outputs_npz(onnx_npz, {"visual_features": np.array([1.0, 2.0], dtype=np.float32)})
            om_dir = root / "om_outputs"
            om_dir.mkdir()

            result = compare_indexed_outputs(onnx_npz=onnx_npz, om_output_dir=om_dir)

            self.assertEqual(result["summary"]["error_count"], 1)
            self.assertIn("OMC output not found", result["records"][0]["error"])


if __name__ == "__main__":
    unittest.main()
