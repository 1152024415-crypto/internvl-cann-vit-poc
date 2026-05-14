import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class HarmonyOSDemoCannDeploymentTests(unittest.TestCase):
    def test_harmonyos_ui_uses_explicit_cann_model_lifecycle_promises(self) -> None:
        source = read_text("demo/entry/src/main/ets/pages/Index.ets")

        self.assertIn("async loadModel(): Promise<void>", source)
        self.assertIn("await testNapi.loadModel", source)
        self.assertIn("async unloadModel(): Promise<void>", source)
        self.assertIn("await testNapi.unloadModel", source)
        self.assertIn("async runOnce(): Promise<void>", source)
        self.assertIn("await testNapi.runOnce", source)
        self.assertIn("async runStability(): Promise<void>", source)
        self.assertIn("await testNapi.runStability", source)
        self.assertIn("@State busy: boolean", source)
        self.assertNotIn("const result = testNapi.runOnce", source)
        self.assertNotIn("const result = testNapi.runStability", source)

    def test_harmonyos_ui_awaits_native_validation_promises(self) -> None:
        native = read_text("demo/entry/src/main/cpp/napi_init.cpp")

        self.assertIn("napi_create_async_work", native)
        self.assertIn("napi_queue_async_work", native)
        self.assertIn("napi_resolve_deferred", native)
        self.assertIn("AsyncOperation::LoadModel", native)
        self.assertIn("AsyncOperation::RunOnce", native)
        self.assertIn("AsyncOperation::RunStability", native)

    def test_harmonyos_napi_exports_promise_based_cann_api(self) -> None:
        typings = read_text("demo/entry/src/main/cpp/types/libentry/Index.d.ts")

        self.assertIn("export interface ModelStatusResult", typings)
        self.assertIn("export interface OfficialSmokeResult", typings)
        self.assertIn("loadModel: (resourceManager: object) => Promise<ModelStatusResult>", typings)
        self.assertIn("unloadModel: () => Promise<ModelStatusResult>", typings)
        self.assertIn(
            "runOfficialSmoke: (resourceManager: object) => Promise<OfficialSmokeResult>",
            typings,
        )
        self.assertIn("runOnce: (resourceManager: object, caseName: string) => Promise<RunResult>", typings)
        self.assertIn(
            "runStability: (resourceManager: object, caseName: string, repeatCount: number) "
            "=> Promise<StabilityResult>",
            typings,
        )

    def test_native_validator_follows_official_load_compile_once_run_many_flow(self) -> None:
        source = read_text("demo/entry/src/main/cpp/validation/nn_runtime_validator.cpp")

        self.assertIn("ModelStatusResult LoadModel", source)
        self.assertIn("ModelStatusResult UnloadModel", source)
        self.assertIn("OH_NNCompilation_ConstructWithOfflineModelBuffer", source)
        self.assertIn("OH_NNCompilation_Build", source)
        self.assertIn("OH_NNExecutor_Construct", source)
        self.assertIn("HIAI_F", source)
        self.assertIn("model_not_loaded", source)
        self.assertNotIn("RunOfflineModel", source)

    def test_native_validator_logs_model_build_diagnostics(self) -> None:
        source = read_text("demo/entry/src/main/cpp/validation/nn_runtime_validator.cpp")

        self.assertIn("load_model_read_om_done", source)
        self.assertIn("om_bytes=%{public}zu", source)
        self.assertIn("selected_device=%{public}s", source)
        self.assertIn("ReturnCodeToInt", source)
        self.assertIn("OH_NNCompilation_Build failed code=", source)

    def test_official_squeezenet_smoke_path_is_packaged_and_exported(self) -> None:
        official_om = ROOT / "demo/entry/src/main/resources/rawfile/official_squeezenet_hiai.om"
        self.assertTrue(official_om.exists())
        self.assertGreater(official_om.stat().st_size, 2_000_000)
        self.assertLess(official_om.stat().st_size, 3_000_000)
        self.assertEqual(
            hashlib.sha256(official_om.read_bytes()).hexdigest().upper(),
            "533B84458D7694174A220D3AA8B984B1F0021FB7D3997A1CC2732AA3B51C7AD3",
        )

        validator = read_text("demo/entry/src/main/cpp/validation/nn_runtime_validator.cpp")
        self.assertIn('kOfficialSampleModelFile = "official_squeezenet_hiai.om"', validator)
        self.assertIn("OfficialSmokeResult RunOfficialSmoke", validator)
        self.assertIn("official_smoke_build_start", validator)
        self.assertIn("OH_NNExecutor_RunSync", validator)

        native = read_text("demo/entry/src/main/cpp/napi_init.cpp")
        self.assertIn("AsyncOperation::RunOfficialSmoke", native)
        self.assertIn('{"runOfficialSmoke"', native)

        ui = read_text("demo/entry/src/main/ets/pages/Index.ets")
        self.assertIn("async runOfficialSmoke(): Promise<void>", ui)
        self.assertIn("await testNapi.runOfficialSmoke", ui)
        self.assertIn("Official Smoke", ui)

    def test_official_sample_images_and_classification_path_are_packaged(self) -> None:
        official_cup = ROOT / "demo/entry/src/main/resources/base/media/official_cup.jpg"
        official_guitar = ROOT / "demo/entry/src/main/resources/base/media/official_guitar.jpg"
        official_labels = ROOT / "demo/entry/src/main/resources/rawfile/official_labels_caffe.txt"
        official_cup_input = ROOT / "demo/entry/src/main/resources/rawfile/official_cup_bgr_227.bin"
        official_guitar_input = ROOT / "demo/entry/src/main/resources/rawfile/official_guitar_bgr_227.bin"

        self.assertEqual(official_cup.stat().st_size, 24_566)
        self.assertEqual(official_guitar.stat().st_size, 965_265)
        self.assertEqual(official_labels.stat().st_size, 22_673)
        self.assertEqual(official_cup_input.stat().st_size, 154_587)
        self.assertEqual(official_guitar_input.stat().st_size, 154_587)

        labels = official_labels.read_text(encoding="utf-8")
        self.assertIn("acoustic guitar", labels)
        self.assertIn("cup", labels)

        typings = read_text("demo/entry/src/main/cpp/types/libentry/Index.d.ts")
        self.assertIn("export interface OfficialClassificationResult", typings)
        self.assertIn(
            "runOfficialClassification: (resourceManager: object, imageName: string) "
            "=> Promise<OfficialClassificationResult>",
            typings,
        )

        validator = read_text("demo/entry/src/main/cpp/validation/nn_runtime_validator.cpp")
        self.assertIn("OfficialClassificationResult RunOfficialClassification", validator)
        self.assertIn('kOfficialLabelsFile = "official_labels_caffe.txt"', validator)
        self.assertIn("official_guitar", validator)
        self.assertIn("official_cup", validator)
        self.assertIn("TopKOfficialLabels", validator)

        ui = read_text("demo/entry/src/main/ets/pages/Index.ets")
        self.assertIn("$r('app.media.official_guitar')", ui)
        self.assertIn("$r('app.media.official_cup')", ui)
        self.assertIn("Official Classify", ui)
        self.assertIn("await testNapi.runOfficialClassification", ui)
