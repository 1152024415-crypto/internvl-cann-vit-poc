import unittest
from unittest.mock import patch


class FakeVisionConfig:
    use_flash_attn = True
    use_fa3 = True


class FakeConfig:
    vision_config = FakeVisionConfig()


class FakeVisionModel:
    config = FakeVisionConfig()


class FakeModel:
    config = FakeConfig()
    vision_model = FakeVisionModel()

    def eval(self):
        return self


class BaselineLoadTests(unittest.TestCase):
    def test_load_model_pins_revision_and_disables_flash_attention(self):
        from internvl_vit_poc.baseline import PINNED_REVISION, load_model

        fake_model = FakeModel()

        with patch("transformers.AutoModel.from_pretrained", return_value=fake_model) as from_pretrained:
            model = load_model(
                "OpenGVLab/InternVL3_5-1B-Instruct",
                "fp32",
                trust_remote_code=True,
                revision=PINNED_REVISION,
                local_files_only=True,
            )

        self.assertIs(model, fake_model)
        kwargs = from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["revision"], PINNED_REVISION)
        self.assertEqual(kwargs["code_revision"], PINNED_REVISION)
        self.assertTrue(kwargs["local_files_only"])
        self.assertFalse(model.config.vision_config.use_flash_attn)
        self.assertFalse(model.config.vision_config.use_fa3)
        self.assertFalse(model.vision_model.config.use_flash_attn)
        self.assertFalse(model.vision_model.config.use_fa3)


if __name__ == "__main__":
    unittest.main()
