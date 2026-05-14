from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class CannOmgNpuPartitionTests(unittest.TestCase):
    def test_conversion_script_rejects_cpu_only_om_by_default(self) -> None:
        script = read_text("scripts/convert_onnx_to_om.sh")

        self.assertIn("REQUIRE_NPU_PARTITION", script)
        self.assertIn("partition type NPU:0", script)
        self.assertIn("CPU-only OM is not acceptable", script)
        self.assertIn(".omg.log", script)
        self.assertIn("LAST_PARTITION_LINE", script)
        self.assertIn("tail -n 1", script)
        self.assertIn("OMG_PLATFORM", script)
        self.assertIn("--platform", script)
        self.assertIn("kirin9030", script)
        self.assertIn("AI_NPUCL", script)
        self.assertIn("CPUCL", script)
        self.assertIn("NPU-targeted host conversion", script)

    def test_platform_plugin_import_script_installs_into_tools_platform(self) -> None:
        script = read_text("scripts/install_cann_platform_plugin_to_wsl.ps1")

        self.assertIn("tools/platform", script)
        self.assertIn("kirin", script)
        self.assertIn("PluginPackagePath", script)
        self.assertIn("CANN platform plugin import", script)
        self.assertIn("Invoke-WslChecked", script)
        self.assertIn("$LASTEXITCODE", script)

    def test_wsl_helper_scripts_fail_when_wsl_fails(self) -> None:
        scripts = [
            "scripts/check_wsl_omg.ps1",
            "scripts/convert_wsl_onnx_to_om.ps1",
            "scripts/copy_wsl_om_to_windows.ps1",
            "scripts/install_cann_platform_plugin_to_wsl.ps1",
            "scripts/install_cann_tools_to_wsl.ps1",
            "scripts/prepare_wsl_stage3.ps1",
        ]

        for relative_path in scripts:
            with self.subTest(script=relative_path):
                script = read_text(relative_path)
                self.assertIn("Invoke-WslChecked", script)
                self.assertIn("$LASTEXITCODE", script)
                self.assertIn("throw", script)

    def test_wsl_conversion_copies_omg_log_back_to_windows(self) -> None:
        script = read_text("scripts/convert_wsl_onnx_to_om.ps1")

        self.assertIn(".omg.log", script)
        self.assertIn("copy to Windows", script)

    def test_docs_record_old_cpu_only_om_and_replacement_evidence(self) -> None:
        docs = read_text("docs/stage-4-vit-projector-chain.md")

        self.assertIn("NPU:0, CPU:1", docs)
        self.assertIn("Release OM was CPU-only", docs)
        self.assertIn("Kirin 9030 platform plugin installed", docs)
        self.assertIn("AI_NPUCL", docs)
        self.assertIn("CANN-specific ONNX", docs)
