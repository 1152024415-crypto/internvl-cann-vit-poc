import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


class OnnxStaticSurgeryTests(unittest.TestCase):
    def test_replaces_class_embedding_expand_and_prunes_dead_nodes(self):
        from internvl_vit_poc.onnx_static_surgery import (
            CLASS_EMBEDDING_INITIALIZER,
            EMBEDDING_CONCAT_OUTPUT,
            prune_unused_graph_inputs,
            replace_class_embedding_expand,
        )

        class_embedding = numpy_helper.from_array(
            np.zeros((1, 1, 4), dtype=np.float32),
            name=CLASS_EMBEDDING_INITIALIZER,
        )
        patch_embedding = numpy_helper.from_array(
            np.zeros((1, 2, 4), dtype=np.float32),
            name="patch_embedding",
        )
        target_shape = numpy_helper.from_array(
            np.array([1, 1, 4], dtype=np.int64),
            name="target_shape",
        )

        graph = helper.make_graph(
            [
                helper.make_node(
                    "Expand",
                    [CLASS_EMBEDDING_INITIALIZER, "target_shape"],
                    ["expanded"],
                    name="/vision_model/embeddings/Expand",
                ),
                helper.make_node(
                    "Cast",
                    ["expanded"],
                    ["expanded_fp32"],
                    name="/vision_model/embeddings/Cast",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Concat",
                    ["expanded_fp32", "patch_embedding"],
                    [EMBEDDING_CONCAT_OUTPUT],
                    name="/vision_model/embeddings/Concat_1",
                    axis=1,
                ),
            ],
            "static-surgery-test",
            inputs=[],
            outputs=[
                helper.make_tensor_value_info(
                    EMBEDDING_CONCAT_OUTPUT, TensorProto.FLOAT, [1, 3, 4]
                )
            ],
            initializer=[class_embedding, patch_embedding, target_shape],
        )
        model = helper.make_model(graph)

        self.assertTrue(replace_class_embedding_expand(model))
        removed = prune_unused_graph_inputs(model)

        self.assertEqual(2, removed)
        self.assertEqual(["Concat"], [node.op_type for node in model.graph.node])
        self.assertEqual(CLASS_EMBEDDING_INITIALIZER, model.graph.node[0].input[0])
        self.assertEqual(
            {CLASS_EMBEDDING_INITIALIZER, "patch_embedding"},
            {initializer.name for initializer in model.graph.initializer},
        )


if __name__ == "__main__":
    unittest.main()
