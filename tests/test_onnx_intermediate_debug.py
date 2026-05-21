import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def make_tiny_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    const = helper.make_tensor("bias", TensorProto.FLOAT, [1, 2], [1.0, -1.0])
    add = helper.make_node("Add", ["x", "bias"], ["add_out"], name="AddBias")
    relu = helper.make_node("Relu", ["add_out"], ["relu_out"], name="ReluActivation")
    identity = helper.make_node("Identity", ["relu_out"], ["y"], name="FinalIdentity")
    graph = helper.make_graph([add, relu, identity], "tiny", [x], [y], [const])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


class OnnxIntermediateDebugTests(unittest.TestCase):
    def test_parse_shape_accepts_comma_separated_dims(self):
        from internvl_vit_poc.onnx_intermediate_debug import parse_shape

        self.assertEqual(parse_shape("1,3,448,448"), [1, 3, 448, 448])

    def test_parse_shape_accepts_x_separated_dims(self):
        from internvl_vit_poc.onnx_intermediate_debug import parse_shape

        self.assertEqual(parse_shape("1x256x1024"), [1, 256, 1024])

    def test_load_input_array_reads_raw_little_endian_float32_bin(self):
        from internvl_vit_poc.onnx_intermediate_debug import load_input_array

        with tempfile.TemporaryDirectory() as tmp:
            input_bin = Path(tmp) / "pixel_values_fp32.bin"
            expected = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype="<f4")
            expected.tofile(input_bin)

            actual = load_input_array(
                image=None,
                input_bin=input_bin,
                input_shape="1,1,2,2",
                input_dtype="float32",
                image_size=448,
            )

            self.assertEqual(actual.dtype, np.float32)
            self.assertEqual(actual.shape, (1, 1, 2, 2))
            np.testing.assert_array_equal(actual, expected.astype(np.float32))

    def test_load_input_array_casts_raw_float16_bin_to_float32(self):
        from internvl_vit_poc.onnx_intermediate_debug import load_input_array

        with tempfile.TemporaryDirectory() as tmp:
            input_bin = Path(tmp) / "pixel_values_fp16.bin"
            expected = np.array([[[[1.5, 2.5], [3.5, 4.5]]]], dtype="<f2")
            expected.tofile(input_bin)

            actual = load_input_array(
                image=None,
                input_bin=input_bin,
                input_shape="1,1,2,2",
                input_dtype="float16",
                image_size=448,
            )

            self.assertEqual(actual.dtype, np.float32)
            self.assertEqual(actual.shape, (1, 1, 2, 2))
            np.testing.assert_array_equal(actual, expected.astype(np.float32))

    def test_load_input_array_rejects_image_and_input_bin_together(self):
        from internvl_vit_poc.onnx_intermediate_debug import load_input_array

        with self.assertRaises(ValueError):
            load_input_array(
                image="image.jpg",
                input_bin="pixel_values_fp32.bin",
                input_shape="1,3,448,448",
                input_dtype="float32",
                image_size=448,
            )

    def test_select_node_outputs_can_filter_by_op_type_and_regex(self):
        from internvl_vit_poc.onnx_intermediate_debug import select_node_outputs

        model = make_tiny_model()

        selected = select_node_outputs(model, op_types={"Relu"}, name_regex="relu")

        self.assertEqual(selected, ["relu_out"])

    def test_augment_graph_outputs_adds_requested_tensor_once(self):
        from internvl_vit_poc.onnx_intermediate_debug import augment_graph_outputs

        model = make_tiny_model()

        augmented = augment_graph_outputs(model, ["add_out", "add_out"])
        output_names = [output.name for output in augmented.graph.output]

        self.assertIn("y", output_names)
        self.assertEqual(output_names.count("add_out"), 1)

    def test_compare_npz_outputs_reports_common_tensor_metrics(self):
        from internvl_vit_poc.onnx_intermediate_debug import compare_npz_outputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left.npz"
            right = root / "right.npz"
            np.savez(left, a=np.array([1.0, 2.0], dtype=np.float32), b=np.array([0.0], dtype=np.float32))
            np.savez(right, a=np.array([1.0, 3.0], dtype=np.float32), c=np.array([1.0], dtype=np.float32))

            result = compare_npz_outputs(left, right)

            self.assertEqual(result["common_count"], 1)
            self.assertEqual(result["left_only"], ["b"])
            self.assertEqual(result["right_only"], ["c"])
            self.assertEqual(result["records"][0]["name"], "a")
            self.assertAlmostEqual(result["records"][0]["metrics"]["max_abs_diff"], 1.0)

    def test_save_outputs_npz_manifest_preserves_output_order(self):
        from internvl_vit_poc.onnx_intermediate_debug import save_outputs_npz

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_npz = root / "outputs.npz"

            manifest = save_outputs_npz(
                output_npz,
                {
                    "visual_features": np.array([1.0], dtype=np.float32),
                    "encoder.layers.0.output": np.array([2.0], dtype=np.float32),
                },
            )

            self.assertEqual(manifest["output_names"], ["visual_features", "encoder.layers.0.output"])


if __name__ == "__main__":
    unittest.main()
