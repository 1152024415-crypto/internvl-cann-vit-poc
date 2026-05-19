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


if __name__ == "__main__":
    unittest.main()
