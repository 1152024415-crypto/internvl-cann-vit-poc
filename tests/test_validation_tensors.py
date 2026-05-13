import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class ValidationTensorTests(unittest.TestCase):
    def test_write_fp32_bin_writes_little_endian_float32(self):
        from internvl_vit_poc.validation_tensors import write_fp32_bin

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tensor.bin"
            tensor = np.array([[1.0, 2.5, -3.0]], dtype=np.float64)

            info = write_fp32_bin(path, tensor)

            expected_bytes = np.array([1.0, 2.5, -3.0], dtype="<f4").tobytes()
            self.assertEqual(path.read_bytes(), expected_bytes)
            self.assertEqual(info["path"], str(path))
            self.assertEqual(info["shape"], [1, 3])
            self.assertEqual(info["dtype"], "float32")
            self.assertEqual(info["byte_count"], 12)
            self.assertEqual(info["sha256"], hashlib.sha256(expected_bytes).hexdigest())

    def test_build_case_metadata_records_files_and_sha256(self):
        from internvl_vit_poc.validation_tensors import build_case_metadata, write_fp32_bin

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_info = write_fp32_bin(root / "dog_pixel_values_fp32.bin", np.zeros((1, 3), dtype=np.float32))
            output_info = write_fp32_bin(root / "dog_visual_tokens_fp32.bin", np.ones((1, 2), dtype=np.float32))

            metadata = build_case_metadata(
                case_name="dog",
                source_image="data/test-images/dog.jpg",
                model_artifact="internvl3_5_vit_projector_fp32_opset18_staticpos.om",
                baseline_source="artifacts/baseline-projector-dog/baseline_output.pt",
                input_info=input_info,
                output_info=output_info,
            )

            self.assertEqual(metadata["case_name"], "dog")
            self.assertEqual(metadata["input"]["shape"], [1, 3])
            self.assertEqual(metadata["expected_output"]["shape"], [1, 2])
            self.assertEqual(metadata["input"]["sha256"], input_info["sha256"])
            self.assertEqual(metadata["expected_output"]["sha256"], output_info["sha256"])

    def test_save_case_metadata_writes_json(self):
        from internvl_vit_poc.validation_tensors import save_case_metadata

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dog.metadata.json"
            metadata = {"case_name": "dog", "input": {"shape": [1, 3]}}

            save_case_metadata(path, metadata)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), metadata)


if __name__ == "__main__":
    unittest.main()
