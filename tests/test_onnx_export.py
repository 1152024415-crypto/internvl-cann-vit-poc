import unittest


class FakeVisionOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeVisionModel:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeVisionOutput(self.output)


class FakeEmbeddings:
    image_size = 448
    patch_size = 14

    def __init__(self):
        self.original_calls = []

    def _get_pos_embed(self, pos_embed, height, width):
        self.original_calls.append((pos_embed, height, width))
        return pos_embed + 1


class FakeInternVLModel:
    def __init__(self, vision_output, projector_output):
        self.vision_model = FakeVisionModel(vision_output)
        self.projector_output = projector_output
        self.extract_feature_calls = []

    def extract_feature(self, pixel_values):
        self.extract_feature_calls.append(pixel_values)
        return self.projector_output


class OnnxExportTests(unittest.TestCase):
    def test_vision_wrapper_returns_last_hidden_state(self):
        import torch

        from internvl_vit_poc.onnx_export import VisionLastHiddenStateWrapper

        vision_output = torch.ones(1, 1025, 1024)
        model = FakeInternVLModel(vision_output, torch.zeros(1, 256, 1024))
        wrapper = VisionLastHiddenStateWrapper(model)
        pixel_values = torch.zeros(1, 3, 448, 448)

        output = wrapper(pixel_values)

        self.assertIs(output, vision_output)
        self.assertEqual(
            model.vision_model.calls[0],
            {
                "pixel_values": pixel_values,
                "output_hidden_states": False,
                "return_dict": True,
            },
        )
        self.assertIsInstance(wrapper, torch.nn.Module)

    def test_projector_wrapper_returns_extract_feature(self):
        import torch

        from internvl_vit_poc.onnx_export import ProjectorVisualTokensWrapper

        projector_output = torch.ones(1, 256, 1024)
        model = FakeInternVLModel(torch.zeros(1, 1025, 1024), projector_output)
        wrapper = ProjectorVisualTokensWrapper(model)
        pixel_values = torch.zeros(1, 3, 448, 448)

        output = wrapper(pixel_values)

        self.assertIs(output, projector_output)
        self.assertEqual(model.extract_feature_calls, [pixel_values])

    def test_make_onnx_metadata_records_static_export_contract(self):
        from internvl_vit_poc.onnx_export import make_onnx_metadata

        metadata = make_onnx_metadata(
            model_id="OpenGVLab/InternVL3_5-1B-Instruct",
            revision="abc123",
            image_path="data/sample.jpg",
            output_path="artifacts/onnx/vision.onnx",
            target="vision",
            dtype="fp32",
            device="cpu",
            opset=18,
            input_shape=(1, 3, 448, 448),
            output_shape=(1, 1025, 1024),
            input_name="pixel_values",
            output_name="last_hidden_state",
        )

        self.assertEqual(metadata["target"], "vision")
        self.assertEqual(metadata["opset"], 18)
        self.assertEqual(metadata["input_name"], "pixel_values")
        self.assertEqual(metadata["output_name"], "last_hidden_state")
        self.assertEqual(metadata["input_shape"], [1, 3, 448, 448])
        self.assertEqual(metadata["output_shape"], [1, 1025, 1024])
        self.assertFalse(metadata["dynamic_axes"])
        self.assertFalse(metadata["static_position_embedding"])

    def test_patch_static_position_embedding_bypasses_native_grid_resize(self):
        import torch

        from internvl_vit_poc.onnx_export import patch_static_position_embedding

        class Model:
            pass

        model = Model()
        model.vision_model = Model()
        model.vision_model.embeddings = FakeEmbeddings()
        pos_embed = torch.ones(1, 1024, 8)

        patched = patch_static_position_embedding(model)
        native_output = model.vision_model.embeddings._get_pos_embed(pos_embed, 32, 32)
        resized_output = model.vision_model.embeddings._get_pos_embed(pos_embed, 16, 16)

        self.assertTrue(patched)
        self.assertIs(native_output, pos_embed)
        self.assertTrue(torch.equal(resized_output, pos_embed + 1))
        self.assertEqual(
            model.vision_model.embeddings.original_calls,
            [(pos_embed, 16, 16)],
        )


if __name__ == "__main__":
    unittest.main()
