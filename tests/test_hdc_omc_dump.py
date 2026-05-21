import json
import tempfile
import unittest
from pathlib import Path


class FakeRunner:
    def __init__(self, local_dump_file: str = "tensor.bin"):
        self.calls = []
        self.local_dump_file = local_dump_file

    def run(self, args):
        self.calls.append(list(args))
        command = self.calls[-1]
        if command[-2:] == ["list", "targets"]:
            return "ABC123\n"
        if "shell" in command:
            shell_command = command[-1]
            if "OK_DIR" in shell_command:
                return "OK_DIR\n"
            if "OK_RUNNER" in shell_command:
                return "OK_RUNNER\n"
            if "OK_MODEL" in shell_command:
                return "OK_MODEL\n"
            if "OK_INPUT" in shell_command:
                return "OK_INPUT\n"
            if "OK_DUMP" in shell_command:
                return "OK_DUMP\n"
            return "RUN_OK\n"
        if "file" in command and "recv" in command:
            local_dir = Path(command[-1])
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / self.local_dump_file).write_bytes(b"\x00\x00\x80?\x00\x00\x00@")
            return "FileTransfer finish\n"
        return ""


class DumpEnableFailureRunner(FakeRunner):
    def run(self, args):
        self.calls.append(list(args))
        command = self.calls[-1]
        if "shell" in command:
            shell_command = command[-1]
            if "OK_INPUT" in shell_command:
                return "OK_INPUT\n"
            if "model_run_tool_internal" in shell_command:
                return "[ERROR] Enable dump failed.\n"
            if "data_proc_tool_internal" in shell_command:
                return "data proc attempted\n"
            if "mkdir -p" in shell_command:
                return ""
            if "OK_DUMP" in shell_command:
                return "MISSING_DUMP\n"
        return ""


class MissingDeviceRunner(FakeRunner):
    def run(self, args):
        self.calls.append(list(args))
        command = self.calls[-1]
        if command[-2:] == ["list", "targets"]:
            return "ABC123\n"
        if "shell" in command:
            shell_command = command[-1]
            if "if test -d" in shell_command and "MISSING_DIR" in shell_command:
                return "MISSING_DIR\n"
            if "if test -f" in shell_command and "MISSING_RUNNER" in shell_command:
                return "MISSING_RUNNER\n"
            if "if test -f" in shell_command and "MISSING_MODEL" in shell_command:
                return "MISSING_MODEL\n"
            raise RuntimeError(f"nonzero legacy check: {shell_command}")
        return ""


class MissingInputRunner(FakeRunner):
    def run(self, args):
        self.calls.append(list(args))
        command = self.calls[-1]
        if "shell" in command:
            shell_command = command[-1]
            if "if test -f" in shell_command and "MISSING_INPUT" in shell_command:
                return "MISSING_INPUT\n"
            raise RuntimeError(f"nonzero legacy check: {shell_command}")
        return ""


class HdcOmcDumpTests(unittest.TestCase):
    def test_case_id_validation_rejects_unsafe_strings(self):
        from internvl_vit_poc.hdc_omc_dump import validate_case_id

        for case_id in ["../dog", "dog;rm", "dog cat", "dog$(id)", ""]:
            with self.subTest(case_id=case_id):
                with self.assertRaises(ValueError):
                    validate_case_id(case_id)

    def test_hdc_command_with_target_inserts_target_before_subcommand(self):
        from internvl_vit_poc.hdc_omc_dump import build_hdc_command

        self.assertEqual(
            build_hdc_command("hdc.exe", "SERIAL1", ["shell", "echo OK"]),
            ["hdc.exe", "-t", "SERIAL1", "shell", "echo OK"],
        )

    def test_dump_one_with_fake_runner_pulls_and_writes_inventory(self):
        from internvl_vit_poc.hdc_omc_dump import dump_one

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_root = root / "omc"
            local_dir = local_root / "dog"
            local_dir.mkdir(parents=True)
            (local_dir / "stale.bin").write_bytes(b"stale")
            inventory_json = local_root / "dog_inventory.json"
            inventory_csv = local_root / "dog_inventory.csv"
            runner = FakeRunner()

            result = dump_one(
                case_id="dog",
                hdc_path="hdc.exe",
                target="ABC123",
                local_dir=local_dir,
                inventory_json=inventory_json,
                inventory_csv=inventory_csv,
                local_root=local_root,
                runner=runner,
            )

            self.assertEqual(result["file_count"], 1)
            self.assertEqual(result["local_dir"], str(local_dir))
            self.assertTrue(inventory_json.exists())
            self.assertTrue(inventory_csv.exists())
            self.assertEqual(json.loads(inventory_json.read_text(encoding="utf-8"))["count"], 1)
            self.assertFalse((local_dir / "stale.bin").exists())
            self.assertIn("tensor.bin", inventory_csv.read_text(encoding="utf-8"))
            self.assertIn(
                [
                    "hdc.exe",
                    "-t",
                    "ABC123",
                    "file",
                    "recv",
                    "/data/local/tmp/internvl_om_compare/dump_dog",
                    str(local_dir),
                ],
                runner.calls,
            )
            shell_commands = [call[-1] for call in runner.calls if "shell" in call]
            self.assertTrue(any("if test -f /data/local/tmp/internvl_om_compare/inputs/dog_pixel_values_fp32.bin" in c for c in shell_commands))
            self.assertTrue(any("./model_run_tool_internal --model=/data/local/tmp/internvl_om_compare/split_internvit_v6_exclude_conv_int8_8.omc" in c for c in shell_commands))

    def test_run_one_with_fake_runner_pulls_output_and_writes_inventory(self):
        from internvl_vit_poc.hdc_omc_dump import run_one

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_root = root / "layer_debug"
            runner = FakeRunner(local_dump_file="output_0.bin")

            result = run_one(
                case_id="dog",
                hdc_path="hdc.exe",
                target="ABC123",
                local_root=local_root,
                runner=runner,
            )

            self.assertEqual(result["mode"], "run")
            self.assertEqual(result["file_count"], 1)
            self.assertEqual(result["local_dir"], str(local_root / "run" / "dog"))
            self.assertTrue((local_root / "run" / "dog_inventory.json").exists())
            shell_commands = [call[-1] for call in runner.calls if "shell" in call]
            model_commands = [command for command in shell_commands if "./model_run_tool_internal" in command]
            self.assertEqual(len(model_commands), 1)
            self.assertIn("--output_dir=/data/local/tmp/internvl_om_compare/run_dog", model_commands[0])
            self.assertNotIn("--enable_item", model_commands[0])
            self.assertTrue(any("mkdir -p /data/local/tmp/internvl_om_compare/run_dog" in c for c in shell_commands))
            self.assertIn(
                [
                    "hdc.exe",
                    "-t",
                    "ABC123",
                    "file",
                    "recv",
                    "/data/local/tmp/internvl_om_compare/run_dog",
                    str(local_root / "run" / "dog"),
                ],
                runner.calls,
            )

    def test_profile_one_uses_enable_item_1_runs_data_proc_and_pulls_proc(self):
        from internvl_vit_poc.hdc_omc_dump import profile_one

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_root = root / "layer_debug"
            runner = FakeRunner(local_dump_file="model.csv")

            result = profile_one(
                case_id="dog",
                hdc_path="hdc.exe",
                target="ABC123",
                local_root=local_root,
                runner=runner,
            )

            self.assertEqual(result["mode"], "profile")
            self.assertEqual(result["local_dir"], str(local_root / "profile" / "dog"))
            self.assertEqual(result["local_proc_dir"], str(local_root / "profile" / "dog_proc"))
            self.assertIn("proc_inventory", result)
            self.assertEqual(result["proc_inventory"]["count"], 1)
            shell_commands = [call[-1] for call in runner.calls if "shell" in call]
            model_commands = [command for command in shell_commands if "./model_run_tool_internal" in command]
            self.assertEqual(len(model_commands), 1)
            self.assertIn("--enable_item=1", model_commands[0])
            self.assertIn("--output_dir=/data/local/tmp/internvl_om_compare/profile_dog", model_commands[0])
            data_proc_commands = [command for command in shell_commands if "data_proc_tool_internal" in command]
            self.assertEqual(len(data_proc_commands), 1)
            self.assertIn("--result_path=/data/local/tmp/internvl_om_compare/profile_proc_dog", data_proc_commands[0])
            self.assertIn(
                [
                    "hdc.exe",
                    "-t",
                    "ABC123",
                    "file",
                    "recv",
                    "/data/local/tmp/internvl_om_compare/profile_proc_dog",
                    str(local_root / "profile" / "dog_proc"),
                ],
                runner.calls,
            )

    def test_dump_one_enable_failure_includes_run_and_data_proc_output(self):
        from internvl_vit_poc.hdc_omc_dump import dump_one

        with tempfile.TemporaryDirectory() as tmp:
            local_root = Path(tmp) / "layer_debug"

            with self.assertRaisesRegex(RuntimeError, "Enable dump failed") as caught:
                dump_one(
                    case_id="dog",
                    hdc_path="hdc.exe",
                    target="ABC123",
                    local_root=local_root,
                    runner=DumpEnableFailureRunner(),
                )

            message = str(caught.exception)
            self.assertIn("[ERROR] Enable dump failed.", message)
            self.assertIn("data proc attempted", message)
            self.assertIn("MISSING_DUMP", message)

    def test_doctor_with_fake_runner_checks_device_paths_and_parses_targets(self):
        from internvl_vit_poc.hdc_omc_dump import doctor

        runner = FakeRunner()

        result = doctor(hdc_path="hdc.exe", target="ABC123", runner=runner)

        self.assertEqual(result["hdc_path"], "hdc.exe")
        self.assertEqual(result["targets"], ["ABC123"])
        self.assertEqual(
            result["checks"],
            {"device_dir": True, "runner": True, "model": True},
        )
        shell_commands = [call[-1] for call in runner.calls if "shell" in call]
        self.assertIn(
            "if test -d /data/local/tmp/internvl_om_compare; then echo OK_DIR; else echo MISSING_DIR; fi",
            shell_commands,
        )
        self.assertIn(
            "if test -f /data/local/tmp/internvl_om_compare/model_run_tool_internal; then echo OK_RUNNER; else echo MISSING_RUNNER; fi",
            shell_commands,
        )
        self.assertIn(
            "if test -f /data/local/tmp/internvl_om_compare/split_internvit_v6_exclude_conv_int8_8.omc; then echo OK_MODEL; else echo MISSING_MODEL; fi",
            shell_commands,
        )

    def test_doctor_missing_device_paths_return_false_checks(self):
        from internvl_vit_poc.hdc_omc_dump import doctor

        result = doctor(hdc_path="hdc.exe", target="ABC123", runner=MissingDeviceRunner())

        self.assertEqual(
            result["checks"],
            {"device_dir": False, "runner": False, "model": False},
        )
        self.assertIn("MISSING_DIR", result["outputs"]["device_dir"])
        self.assertIn("MISSING_RUNNER", result["outputs"]["runner"])
        self.assertIn("MISSING_MODEL", result["outputs"]["model"])

    def test_dump_one_missing_input_raises_file_not_found(self):
        from internvl_vit_poc.hdc_omc_dump import dump_one

        with self.assertRaisesRegex(FileNotFoundError, "Device input does not exist"):
            dump_one(case_id="dog", hdc_path="hdc.exe", runner=MissingInputRunner())

    def test_prepare_local_dump_dir_rejects_paths_outside_allowed_root(self):
        from internvl_vit_poc.hdc_omc_dump import prepare_local_dump_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed_root = root / "allowed"
            outside = root / "outside"

            with self.assertRaisesRegex(ValueError, "Local dump dir must stay under"):
                prepare_local_dump_dir(outside, allowed_root=allowed_root)

    def test_direct_hdc_scripts_exist_and_use_missing_markers(self):
        root = Path(__file__).resolve().parents[1]
        script_names = [
            "hdc_omc_doctor.bat",
            "hdc_omc_dump_one.bat",
            "hdc_omc_doctor.sh",
            "hdc_omc_dump_one.sh",
        ]

        for script_name in script_names:
            with self.subTest(script_name=script_name):
                script = root / "scripts" / script_name
                text = script.read_text(encoding="utf-8")
                self.assertIn("MISSING_", text)
                self.assertIn("HDC_TARGET", text)

        dump_bat = (root / "scripts" / "hdc_omc_dump_one.bat").read_text(encoding="utf-8")
        dump_sh = (root / "scripts" / "hdc_omc_dump_one.sh").read_text(encoding="utf-8")
        self.assertIn("rmdir /S /Q", dump_bat)
        self.assertIn("rm -rf \"$LOCAL_DIR\"", dump_sh)
        self.assertIn("compare_omc_layers.py inventory", dump_bat)
        self.assertIn("compare_omc_layers.py inventory", dump_sh)

    def test_posix_direct_hdc_scripts_preserve_logged_hdc_shell_failures(self):
        root = Path(__file__).resolve().parents[1]
        script_names = [
            "hdc_omc_run_one.sh",
            "hdc_omc_profile_one.sh",
            "hdc_omc_dump_one.sh",
        ]

        for script_name in script_names:
            with self.subTest(script_name=script_name):
                text = (root / "scripts" / script_name).read_text(encoding="utf-8")
                self.assertIn("hdc_shell_log()", text)
                self.assertIn("status_file=", text)
                self.assertIn("tee_status=$?", text)
                self.assertIn("cmd_status=$(cat \"$status_file\")", text)
                self.assertIn("if [ \"$cmd_status\" -ne 0 ]; then", text)
                self.assertIn("return \"$cmd_status\"", text)

    def test_dump_batch_surfaces_logged_runner_and_data_proc_failures(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts" / "hdc_omc_dump_one.bat").read_text(encoding="utf-8")

        runner_command = (
            "\"%HDC%\" %TARGET_ARGS% shell \"cd %DEVICE_DIR%; chmod +x %RUNNER_NAME%; "
            "export LD_LIBRARY_PATH=.; rm -rf dump_%CASE_ID%; mkdir -p %DEVICE_DUMP%; "
            "./%RUNNER_NAME% --model=%DEVICE_DIR%/%MODEL_NAME% --input=%DEVICE_INPUT% "
            "--output_dir=%DEVICE_DUMP% --enable_item=2 --times=1\" > \"%RUN_LOG%\" 2>&1"
        )
        data_proc_command = (
            "\"%HDC%\" %TARGET_ARGS% shell \"cd %DEVICE_DIR%; chmod +x data_proc_tool_internal "
            "2>/dev/null || true; export LD_LIBRARY_PATH=.; rm -rf dump_proc_%CASE_ID%; "
            "mkdir -p %DEVICE_PROC%; ./data_proc_tool_internal --result_path=%DEVICE_PROC%\" "
            "> \"%DATA_PROC_LOG%\" 2>&1"
        )
        self.assertIn(runner_command, text)
        self.assertIn(runner_command + "\nset \"RUN_STATUS=%ERRORLEVEL%\"", text)
        self.assertIn(data_proc_command, text)
        self.assertIn(data_proc_command + "\nset \"DATA_PROC_STATUS=%ERRORLEVEL%\"", text)
        self.assertIn("if not \"%RUN_STATUS%\"==\"0\" exit /b %RUN_STATUS%", text)
        self.assertIn("if not \"%DATA_PROC_STATUS%\"==\"0\" exit /b %DATA_PROC_STATUS%", text)


if __name__ == "__main__":
    unittest.main()
