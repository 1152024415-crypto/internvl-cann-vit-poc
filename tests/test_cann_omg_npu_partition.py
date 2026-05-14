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

    def test_platform_plugin_import_script_installs_into_tools_platform(self) -> None:
        script = read_text("scripts/install_cann_platform_plugin_to_wsl.ps1")

        self.assertIn("tools/platform", script)
        self.assertIn("kirin", script)
        self.assertIn("PluginPackagePath", script)
        self.assertIn("CANN platform plugin import", script)

    def test_docs_record_current_cpu_only_om_root_cause(self) -> None:
        docs = read_text("docs/stage-4-vit-projector-chain.md")

        self.assertIn("NPU:0, CPU:1", docs)
        self.assertIn("CPU-only OM", docs)
        self.assertIn("matching Kirin platform plugin", docs)
