import unittest


class ModelSplitTests(unittest.TestCase):
    def test_split_state_dict_keeps_only_vision_by_default(self):
        from internvl_vit_poc.model_split import split_state_dict

        state = {
            "vision_model.encoder.layers.0.attn.qkv.weight": "vision-qkv",
            "vision_model.embeddings.position_embedding": "vision-pos",
            "mlp1.0.weight": "projector",
            "language_model.model.embed_tokens.weight": "llm",
        }

        split = split_state_dict(state)

        self.assertEqual(
            split,
            {
                "encoder.layers.0.attn.qkv.weight": "vision-qkv",
                "embeddings.position_embedding": "vision-pos",
            },
        )

    def test_split_state_dict_can_include_projector_with_prefixed_keys(self):
        from internvl_vit_poc.model_split import split_state_dict

        state = {
            "vision_model.encoder.layers.0.mlp.fc1.bias": "vision",
            "mlp1.0.weight": "projector0",
            "mlp1.1.bias": "projector1",
            "language_model.lm_head.weight": "llm",
        }

        split = split_state_dict(state, include_projector=True)

        self.assertEqual(
            split,
            {
                "vision_model.encoder.layers.0.mlp.fc1.bias": "vision",
                "mlp1.0.weight": "projector0",
                "mlp1.1.bias": "projector1",
            },
        )

    def test_static_baseline_metadata_records_shapes_and_kind(self):
        from internvl_vit_poc.metadata import make_baseline_metadata

        metadata = make_baseline_metadata(
            model_id="OpenGVLab/InternVL3_5-1B-Instruct",
            image_path="sample.jpg",
            input_shape=(1, 3, 448, 448),
            output_shape=(1, 1025, 1024),
            output_kind="vision_last_hidden_state",
            dtype="fp32",
            device="cpu",
            include_projector=False,
        )

        self.assertEqual(metadata["input_shape"], [1, 3, 448, 448])
        self.assertEqual(metadata["output_shape"], [1, 1025, 1024])
        self.assertEqual(metadata["output_kind"], "vision_last_hidden_state")
        self.assertFalse(metadata["include_projector"])


if __name__ == "__main__":
    unittest.main()
