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
        self.assertIn("loadModel: (resourceManager: object) => Promise<ModelStatusResult>", typings)
        self.assertIn("unloadModel: () => Promise<ModelStatusResult>", typings)
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
