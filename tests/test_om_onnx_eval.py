import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class OmOnnxEvalTests(unittest.TestCase):
    def test_case_id_is_stable_and_filename_safe(self):
        from internvl_vit_poc.om_onnx_eval import build_case_id

        self.assertEqual(build_case_id(Path("class a/0001 cat.jpg")), "class_a_0001_cat")
        self.assertEqual(build_case_id(Path("000000018463.jpg")), "000000018463")

    def test_compare_manifest_outputs_reports_missing_and_metrics(self):
        from internvl_vit_poc.om_onnx_eval import compare_manifest_outputs, write_fp32_bin

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx_dir = root / "onnx_outputs"
            om_dir = root / "om_outputs"
            onnx_dir.mkdir()
            om_dir.mkdir()

            write_fp32_bin(onnx_dir / "ok_visual_tokens_int8_onnx_fp32.bin", np.array([1.0, 2.0], dtype=np.float32))
            write_fp32_bin(om_dir / "ok_visual_tokens_om_fp32.bin", np.array([1.0, 2.1], dtype=np.float32))

            manifest = {
                "cases": [
                    {
                        "case_id": "ok",
                        "image": "ok.jpg",
                        "onnx_output_bin": str(onnx_dir / "ok_visual_tokens_int8_onnx_fp32.bin"),
                        "output_shape": [2],
                    },
                    {
                        "case_id": "missing",
                        "image": "missing.jpg",
                        "onnx_output_bin": str(onnx_dir / "missing_visual_tokens_int8_onnx_fp32.bin"),
                        "output_shape": [2],
                    },
                ]
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = compare_manifest_outputs(
                manifest_path=manifest_path,
                om_output_dir=om_dir,
                om_output_suffix="_visual_tokens_om_fp32.bin",
            )

            self.assertEqual(result["summary"]["count"], 2)
            self.assertEqual(result["summary"]["valid_count"], 1)
            self.assertEqual(result["summary"]["missing_count"], 1)
            self.assertGreater(result["summary"]["cosine_mean"], 0.99)
            self.assertEqual(result["records"][0]["case_id"], "ok")
            self.assertIn("compare", result["records"][0])

    def test_om_output_map_overrides_default_output_name(self):
        from internvl_vit_poc.om_onnx_eval import compare_manifest_outputs, write_fp32_bin

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fp32_bin(root / "ref.bin", np.array([1.0, 0.0], dtype=np.float32))
            write_fp32_bin(root / "tool_dump_0.bin", np.array([0.9, 0.1], dtype=np.float32))

            manifest = {
                "cases": [
                    {
                        "case_id": "case0",
                        "image": "case0.jpg",
                        "onnx_output_bin": str(root / "ref.bin"),
                        "output_shape": [2],
                    }
                ]
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_map = root / "om_output_map.csv"
            output_map.write_text("case_id,om_output_bin\ncase0,tool_dump_0.bin\n", encoding="utf-8")

            result = compare_manifest_outputs(
                manifest_path=manifest_path,
                om_output_dir=root / "missing_default_dir",
                om_output_map=output_map,
            )

            self.assertEqual(result["summary"]["valid_count"], 1)
            self.assertEqual(result["records"][0]["om_output_bin"], str(root / "tool_dump_0.bin"))

    def test_compare_accepts_manifest_paths_relative_to_current_working_directory(self):
        from internvl_vit_poc.om_onnx_eval import compare_manifest_outputs, write_fp32_bin

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            onnx_dir = work / "int8_onnx_outputs"
            om_dir = work / "om_outputs"
            onnx_dir.mkdir(parents=True)
            om_dir.mkdir(parents=True)

            write_fp32_bin(onnx_dir / "case_visual_tokens_int8_onnx_fp32.bin", np.array([1.0, 2.0], dtype=np.float32))
            write_fp32_bin(om_dir / "case_visual_tokens_om_fp32.bin", np.array([1.0, 2.0], dtype=np.float32))

            manifest = {
                "cases": [
                    {
                        "case_id": "case",
                        "image": "case.jpg",
                        "onnx_output_bin": str((work / "int8_onnx_outputs" / "case_visual_tokens_int8_onnx_fp32.bin").relative_to(root)),
                        "output_shape": [2],
                    }
                ]
            }
            manifest_path = work / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            cwd = Path.cwd()
            try:
                import os

                os.chdir(root)
                result = compare_manifest_outputs(
                    manifest_path=Path("work") / "manifest.json",
                    om_output_dir=Path("work") / "om_outputs",
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(result["summary"]["valid_count"], 1)
            self.assertEqual(result["summary"]["missing_count"], 0)


if __name__ == "__main__":
    unittest.main()
