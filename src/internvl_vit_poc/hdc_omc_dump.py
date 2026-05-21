from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from internvl_vit_poc.omc_dump_align import inventory_dump_files, save_inventory_csv

DEFAULT_HDC_PATH = Path(r"D:\software\deveco\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe")
DEFAULT_DEVICE_DIR = "/data/local/tmp/internvl_om_compare"
DEFAULT_MODEL_NAME = "split_internvit_v6_exclude_conv_int8_8.omc"
DEFAULT_RUNNER_NAME = "model_run_tool_internal"
DEFAULT_DATA_PROC_NAME = "data_proc_tool_internal"
DEFAULT_LOCAL_ROOT = Path("quantized_v6/layer_debug")
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class CommandRunner(Protocol):
    def run(self, args: Sequence[str]) -> str:
        ...


class SubprocessRunner:
    def run(self, args: Sequence[str]) -> str:
        completed = subprocess.run(args, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Command failed ({completed.returncode}): {list(args)}\n{message}")
        return completed.stdout


def validate_case_id(case_id: str) -> str:
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError(f"Unsafe case_id {case_id!r}; expected [A-Za-z0-9_.-]+")
    return case_id


def build_hdc_command(hdc_path: str | Path, target: str | None, subcommand: Sequence[str]) -> list[str]:
    command = [str(hdc_path)]
    if target:
        command.extend(["-t", target])
    command.extend(str(part) for part in subcommand)
    return command


def find_hdc_path(hdc: str | Path | None = None) -> str:
    if hdc:
        return str(hdc)

    path_hdc = shutil.which("hdc") or shutil.which("hdc.exe")
    if path_hdc:
        return path_hdc

    if DEFAULT_HDC_PATH.exists():
        return str(DEFAULT_HDC_PATH)

    raise FileNotFoundError(
        "Could not find hdc. Pass --hdc, add hdc to PATH, or install DevEco Studio at the default SDK path."
    )


def parse_targets(output: str) -> list[str]:
    targets = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped and not stripped.lower().startswith("empty"):
            targets.append(stripped.split()[0])
    return targets


def _run_hdc(
    runner: CommandRunner,
    hdc_path: str | Path,
    target: str | None,
    subcommand: Sequence[str],
) -> str:
    return runner.run(build_hdc_command(hdc_path, target, subcommand))


def _check_shell_token(
    runner: CommandRunner,
    hdc_path: str | Path,
    target: str | None,
    shell_command: str,
    token: str,
) -> tuple[bool, str]:
    output = _run_hdc(runner, hdc_path, target, ["shell", shell_command])
    return token in output.split(), output


def _check_shell_command(test_op: str, path: str, ok_token: str, missing_token: str) -> str:
    return f"if test {test_op} {shlex.quote(path)}; then echo {ok_token}; else echo {missing_token}; fi"


def _device_path(device_dir: str, *parts: str) -> str:
    return "/".join([device_dir.rstrip("/"), *(part.strip("/") for part in parts)])


def _default_local_dir(mode: str, case_id: str) -> Path:
    return DEFAULT_LOCAL_ROOT / mode / case_id


def _default_inventory_json(mode: str, case_id: str) -> Path:
    return DEFAULT_LOCAL_ROOT / mode / f"{case_id}_inventory.json"


def _default_inventory_csv(mode: str, case_id: str) -> Path:
    return DEFAULT_LOCAL_ROOT / mode / f"{case_id}_inventory.csv"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def prepare_local_dump_dir(path: str | Path, *, allowed_root: str | Path = DEFAULT_LOCAL_ROOT) -> Path:
    local_path = Path(path)
    resolved = local_path.resolve()
    resolved_allowed_root = Path(allowed_root).resolve()
    if not _is_relative_to(resolved, resolved_allowed_root):
        raise ValueError(f"Local dump dir must stay under {resolved_allowed_root}: {local_path}")
    if resolved == resolved.parent:
        raise ValueError(f"Refusing to remove filesystem root as local dump dir: {local_path}")
    if local_path.exists():
        if not local_path.is_dir():
            raise NotADirectoryError(f"Local dump path exists but is not a directory: {local_path}")
        shutil.rmtree(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return local_path


def _write_inventory(
    local_dir: str | Path,
    inventory_json: str | Path,
    inventory_csv: str | Path,
) -> dict[str, object]:
    inventory = inventory_dump_files(local_dir)
    inventory_json_path = Path(inventory_json)
    inventory_json_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_json_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    save_inventory_csv(inventory_csv, inventory["records"])
    return inventory


def _run_hdc_capture(
    runner: CommandRunner,
    hdc_path: str | Path,
    target: str | None,
    subcommand: Sequence[str],
) -> tuple[str, str | None]:
    try:
        return _run_hdc(runner, hdc_path, target, subcommand), None
    except RuntimeError as exc:
        return str(exc), str(exc)


def _model_run_command(
    *,
    device_dir: str,
    model_name: str,
    runner_name: str,
    model_path: str,
    input_path: str,
    output_dir: str,
    output_dir_name: str,
    enable_item: int | None,
) -> str:
    enable_arg = "" if enable_item is None else f" --enable_item={enable_item}"
    return (
        f"cd {shlex.quote(device_dir)}; "
        f"chmod +x {shlex.quote(runner_name)}; "
        "export LD_LIBRARY_PATH=.; "
        f"rm -rf {shlex.quote(output_dir_name)}; "
        f"mkdir -p {shlex.quote(output_dir)}; "
        f"./{shlex.quote(runner_name)} "
        f"--model={shlex.quote(model_path)} "
        f"--input={shlex.quote(input_path)} "
        f"--output_dir={shlex.quote(output_dir)}"
        f"{enable_arg} --times=1"
    )


def _data_proc_command(*, device_dir: str, data_proc_name: str, result_path: str, result_dir_name: str) -> str:
    return (
        f"cd {shlex.quote(device_dir)}; "
        f"chmod +x {shlex.quote(data_proc_name)} 2>/dev/null || true; "
        "export LD_LIBRARY_PATH=.; "
        f"rm -rf {shlex.quote(result_dir_name)}; "
        f"mkdir -p {shlex.quote(result_path)}; "
        f"./{shlex.quote(data_proc_name)} --result_path={shlex.quote(result_path)}"
    )


def _run_one_case(
    *,
    mode: str,
    case_id: str,
    hdc_path: str | Path | None = None,
    target: str | None = None,
    device_dir: str = DEFAULT_DEVICE_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    runner_name: str = DEFAULT_RUNNER_NAME,
    data_proc_name: str = DEFAULT_DATA_PROC_NAME,
    device_input: str | None = None,
    device_output_name: str | None = None,
    device_proc_name: str | None = None,
    local_dir: str | Path | None = None,
    local_proc_dir: str | Path | None = None,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    inventory_json: str | Path | None = None,
    inventory_csv: str | Path | None = None,
    proc_inventory_json: str | Path | None = None,
    proc_inventory_csv: str | Path | None = None,
    enable_item: int | None = None,
    run_data_proc: bool = False,
    require_files: bool = True,
    runner: CommandRunner | None = None,
) -> dict[str, object]:
    safe_case_id = validate_case_id(case_id)
    resolved_hdc = find_hdc_path(hdc_path)
    command_runner = runner or SubprocessRunner()

    input_path = device_input or _device_path(device_dir, "inputs", f"{safe_case_id}_pixel_values_fp32.bin")
    model_path = _device_path(device_dir, model_name)
    output_dir_name = device_output_name or f"{mode}_{safe_case_id}"
    output_dir = _device_path(device_dir, output_dir_name)
    proc_dir_name = device_proc_name or f"{mode}_proc_{safe_case_id}"
    proc_dir = _device_path(device_dir, proc_dir_name)
    local_root_path = Path(local_root)
    default_local_dir = local_root_path / mode / safe_case_id
    local_output_dir = Path(local_dir) if local_dir is not None else default_local_dir
    inventory_json_path = (
        Path(inventory_json)
        if inventory_json is not None
        else local_root_path / mode / f"{safe_case_id}_inventory.json"
    )
    inventory_csv_path = (
        Path(inventory_csv)
        if inventory_csv is not None
        else local_root_path / mode / f"{safe_case_id}_inventory.csv"
    )

    ok_input, input_output = _check_shell_token(
        command_runner,
        resolved_hdc,
        target,
        _check_shell_command("-f", input_path, "OK_INPUT", "MISSING_INPUT"),
        "OK_INPUT",
    )
    if not ok_input:
        raise FileNotFoundError(f"Device input does not exist: {input_path}; check output: {input_output.strip()}")

    mkdir_output = _run_hdc(
        command_runner,
        resolved_hdc,
        target,
        ["shell", f"mkdir -p {shlex.quote(output_dir)}"],
    )
    run_command = _model_run_command(
        device_dir=device_dir,
        model_name=model_name,
        runner_name=runner_name,
        model_path=model_path,
        input_path=input_path,
        output_dir=output_dir,
        output_dir_name=output_dir_name,
        enable_item=enable_item,
    )
    run_output, run_error = _run_hdc_capture(command_runner, resolved_hdc, target, ["shell", run_command])

    data_proc_output = ""
    data_proc_error = None
    data_proc_command = None
    if run_data_proc:
        data_proc_mkdir_output = _run_hdc(
            command_runner,
            resolved_hdc,
            target,
            ["shell", f"mkdir -p {shlex.quote(proc_dir)}"],
        )
        data_proc_command = _data_proc_command(
            device_dir=device_dir,
            data_proc_name=data_proc_name,
            result_path=proc_dir,
            result_dir_name=proc_dir_name,
        )
        data_proc_output, data_proc_error = _run_hdc_capture(
            command_runner,
            resolved_hdc,
            target,
            ["shell", data_proc_command],
        )
    else:
        data_proc_mkdir_output = ""

    ok_output, output_check = _check_shell_token(
        command_runner,
        resolved_hdc,
        target,
        _check_shell_command("-d", output_dir, "OK_DUMP", "MISSING_DUMP"),
        "OK_DUMP",
    )

    dump_enable_failed = mode == "dump" and "Enable dump failed" in run_output
    if run_error or data_proc_error or not ok_output or dump_enable_failed:
        if mode == "dump":
            details = {
                "mode": mode,
                "case_id": safe_case_id,
                "device_output_dir": output_dir,
                "device_proc_dir": proc_dir,
                "outputs": {
                    "input_check": input_output,
                    "mkdir_output": mkdir_output,
                    "run": run_output,
                    "data_proc_mkdir": data_proc_mkdir_output,
                    "data_proc": data_proc_output,
                    "output_check": output_check,
                },
            }
            raise RuntimeError(f"dump-one failed with captured outputs:\n{json.dumps(details, indent=2, ensure_ascii=False)}")
        if run_error:
            raise RuntimeError(run_output)
        if data_proc_error:
            raise RuntimeError(data_proc_output)
        raise FileNotFoundError(f"Device output directory was not created: {output_dir}; check output: {output_check.strip()}")

    prepare_local_dump_dir(local_output_dir, allowed_root=local_root_path)
    recv_output = _run_hdc(
        command_runner,
        resolved_hdc,
        target,
        ["file", "recv", output_dir, str(local_output_dir)],
    )
    inventory = _write_inventory(local_output_dir, inventory_json_path, inventory_csv_path)

    result: dict[str, object] = {
        "mode": mode,
        "case_id": safe_case_id,
        "hdc_path": resolved_hdc,
        "target": target,
        "device_input": input_path,
        "device_output_dir": output_dir,
        "local_dir": str(local_output_dir),
        "inventory_json": str(inventory_json_path),
        "inventory_csv": str(inventory_csv_path),
        "file_count": inventory["count"],
        "outputs": {
            "input_check": input_output,
            "mkdir_output": mkdir_output,
            "run": run_output,
            "output_check": output_check,
            "recv": recv_output,
        },
        "run_command": run_command,
    }

    if require_files and inventory["count"] == 0:
        if mode == "dump":
            details = {
                "mode": mode,
                "case_id": safe_case_id,
                "device_output_dir": output_dir,
                "device_proc_dir": proc_dir,
                "local_dir": str(local_output_dir),
                "outputs": {
                    "input_check": input_output,
                    "mkdir_output": mkdir_output,
                    "run": run_output,
                    "data_proc_mkdir": data_proc_mkdir_output,
                    "data_proc": data_proc_output,
                    "output_check": output_check,
                    "recv": recv_output,
                },
            }
            raise RuntimeError(f"dump-one produced no pulled files:\n{json.dumps(details, indent=2, ensure_ascii=False)}")
        raise RuntimeError(f"{mode}-one produced no files in {local_output_dir}; run output:\n{run_output}")

    if run_data_proc:
        local_proc_path = Path(local_proc_dir) if local_proc_dir is not None else local_root_path / mode / f"{safe_case_id}_proc"
        proc_inventory_json_path = (
            Path(proc_inventory_json)
            if proc_inventory_json is not None
            else local_root_path / mode / f"{safe_case_id}_proc_inventory.json"
        )
        proc_inventory_csv_path = (
            Path(proc_inventory_csv)
            if proc_inventory_csv is not None
            else local_root_path / mode / f"{safe_case_id}_proc_inventory.csv"
        )
        prepare_local_dump_dir(local_proc_path, allowed_root=local_root_path)
        proc_recv_output = _run_hdc(
            command_runner,
            resolved_hdc,
            target,
            ["file", "recv", proc_dir, str(local_proc_path)],
        )
        proc_inventory = _write_inventory(local_proc_path, proc_inventory_json_path, proc_inventory_csv_path)
        result.update(
            {
                "device_proc_dir": proc_dir,
                "local_proc_dir": str(local_proc_path),
                "proc_inventory_json": str(proc_inventory_json_path),
                "proc_inventory_csv": str(proc_inventory_csv_path),
                "proc_file_count": proc_inventory["count"],
                "proc_inventory": proc_inventory,
                "data_proc_command": data_proc_command,
            }
        )
        result["outputs"].update(
            {
                "data_proc_mkdir": data_proc_mkdir_output,
                "data_proc": data_proc_output,
                "proc_recv": proc_recv_output,
            }
        )

    return result


def doctor(
    *,
    hdc_path: str | Path | None = None,
    target: str | None = None,
    device_dir: str = DEFAULT_DEVICE_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    runner_name: str = DEFAULT_RUNNER_NAME,
    runner: CommandRunner | None = None,
) -> dict[str, object]:
    resolved_hdc = find_hdc_path(hdc_path)
    command_runner = runner or SubprocessRunner()

    list_output = _run_hdc(command_runner, resolved_hdc, None, ["list", "targets"])
    targets = parse_targets(list_output)
    if not targets:
        raise RuntimeError("No hdc targets found.")

    ok_dir, dir_output = _check_shell_token(
        command_runner,
        resolved_hdc,
        target,
        _check_shell_command("-d", device_dir, "OK_DIR", "MISSING_DIR"),
        "OK_DIR",
    )
    ok_runner, runner_output = _check_shell_token(
        command_runner,
        resolved_hdc,
        target,
        _check_shell_command("-f", _device_path(device_dir, runner_name), "OK_RUNNER", "MISSING_RUNNER"),
        "OK_RUNNER",
    )
    ok_model, model_output = _check_shell_token(
        command_runner,
        resolved_hdc,
        target,
        _check_shell_command("-f", _device_path(device_dir, model_name), "OK_MODEL", "MISSING_MODEL"),
        "OK_MODEL",
    )

    return {
        "hdc_path": resolved_hdc,
        "target": target,
        "targets": targets,
        "checks": {
            "device_dir": ok_dir,
            "runner": ok_runner,
            "model": ok_model,
        },
        "outputs": {
            "list_targets": list_output,
            "device_dir": dir_output,
            "runner": runner_output,
            "model": model_output,
        },
    }


def dump_one(
    *,
    case_id: str,
    hdc_path: str | Path | None = None,
    target: str | None = None,
    device_dir: str = DEFAULT_DEVICE_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    runner_name: str = DEFAULT_RUNNER_NAME,
    data_proc_name: str = DEFAULT_DATA_PROC_NAME,
    device_input: str | None = None,
    local_dir: str | Path | None = None,
    local_proc_dir: str | Path | None = None,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    inventory_json: str | Path | None = None,
    inventory_csv: str | Path | None = None,
    proc_inventory_json: str | Path | None = None,
    proc_inventory_csv: str | Path | None = None,
    runner: CommandRunner | None = None,
) -> dict[str, object]:
    return _run_one_case(
        mode="dump",
        case_id=case_id,
        hdc_path=hdc_path,
        target=target,
        device_dir=device_dir,
        model_name=model_name,
        runner_name=runner_name,
        data_proc_name=data_proc_name,
        device_input=device_input,
        local_dir=local_dir,
        local_proc_dir=local_proc_dir,
        local_root=local_root,
        inventory_json=inventory_json,
        inventory_csv=inventory_csv,
        proc_inventory_json=proc_inventory_json,
        proc_inventory_csv=proc_inventory_csv,
        enable_item=2,
        run_data_proc=True,
        runner=runner,
    )


def run_one(
    *,
    case_id: str,
    hdc_path: str | Path | None = None,
    target: str | None = None,
    device_dir: str = DEFAULT_DEVICE_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    runner_name: str = DEFAULT_RUNNER_NAME,
    device_input: str | None = None,
    local_dir: str | Path | None = None,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    inventory_json: str | Path | None = None,
    inventory_csv: str | Path | None = None,
    runner: CommandRunner | None = None,
) -> dict[str, object]:
    return _run_one_case(
        mode="run",
        case_id=case_id,
        hdc_path=hdc_path,
        target=target,
        device_dir=device_dir,
        model_name=model_name,
        runner_name=runner_name,
        device_input=device_input,
        local_dir=local_dir,
        local_root=local_root,
        inventory_json=inventory_json,
        inventory_csv=inventory_csv,
        enable_item=None,
        run_data_proc=False,
        runner=runner,
    )


def profile_one(
    *,
    case_id: str,
    hdc_path: str | Path | None = None,
    target: str | None = None,
    device_dir: str = DEFAULT_DEVICE_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    runner_name: str = DEFAULT_RUNNER_NAME,
    data_proc_name: str = DEFAULT_DATA_PROC_NAME,
    device_input: str | None = None,
    local_dir: str | Path | None = None,
    local_proc_dir: str | Path | None = None,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    inventory_json: str | Path | None = None,
    inventory_csv: str | Path | None = None,
    proc_inventory_json: str | Path | None = None,
    proc_inventory_csv: str | Path | None = None,
    runner: CommandRunner | None = None,
) -> dict[str, object]:
    return _run_one_case(
        mode="profile",
        case_id=case_id,
        hdc_path=hdc_path,
        target=target,
        device_dir=device_dir,
        model_name=model_name,
        runner_name=runner_name,
        data_proc_name=data_proc_name,
        device_input=device_input,
        local_dir=local_dir,
        local_proc_dir=local_proc_dir,
        local_root=local_root,
        inventory_json=inventory_json,
        inventory_csv=inventory_csv,
        proc_inventory_json=proc_inventory_json,
        proc_inventory_csv=proc_inventory_csv,
        enable_item=1,
        run_data_proc=True,
        runner=runner,
    )


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hdc", help="Path to hdc executable.")
    parser.add_argument("--target", help="Optional hdc target serial.")
    parser.add_argument("--device-dir", default=DEFAULT_DEVICE_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--runner-name", default=DEFAULT_RUNNER_NAME)


def _add_one_case_options(parser: argparse.ArgumentParser) -> None:
    _add_common_options(parser)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--device-input")
    parser.add_argument("--local-dir")
    parser.add_argument("--local-root", default=str(DEFAULT_LOCAL_ROOT))
    parser.add_argument("--inventory-json")
    parser.add_argument("--inventory-csv")


def _add_data_proc_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-proc-name", default=DEFAULT_DATA_PROC_NAME)
    parser.add_argument("--local-proc-dir")
    parser.add_argument("--proc-inventory-json")
    parser.add_argument("--proc-inventory-csv")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate one-case HDC OMC run/profile/dump modes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check hdc, target, and device-side files.")
    _add_common_options(doctor_parser)

    run_parser = subparsers.add_parser("run-one", help="Run ordinary inference, pull output_0, and inventory files.")
    _add_one_case_options(run_parser)

    profile_parser = subparsers.add_parser("profile-one", help="Run profiling, process results, pull output and CSV dirs.")
    _add_one_case_options(profile_parser)
    _add_data_proc_options(profile_parser)

    dump_parser = subparsers.add_parser("dump-one", help="Run one OMC dump attempt with captured model/data_proc output.")
    _add_one_case_options(dump_parser)
    _add_data_proc_options(dump_parser)

    return parser.parse_args(argv)


def run(args: argparse.Namespace, runner: CommandRunner | None = None) -> dict[str, object]:
    common = {
        "hdc_path": args.hdc,
        "target": args.target,
        "device_dir": args.device_dir,
        "model_name": args.model_name,
        "runner_name": args.runner_name,
        "runner": runner,
    }
    if args.command == "doctor":
        return doctor(**common)
    if args.command == "run-one":
        return run_one(
            case_id=args.case_id,
            device_input=args.device_input,
            local_dir=args.local_dir,
            local_root=args.local_root,
            inventory_json=args.inventory_json,
            inventory_csv=args.inventory_csv,
            **common,
        )
    if args.command == "profile-one":
        return profile_one(
            case_id=args.case_id,
            data_proc_name=args.data_proc_name,
            device_input=args.device_input,
            local_dir=args.local_dir,
            local_proc_dir=args.local_proc_dir,
            local_root=args.local_root,
            inventory_json=args.inventory_json,
            inventory_csv=args.inventory_csv,
            proc_inventory_json=args.proc_inventory_json,
            proc_inventory_csv=args.proc_inventory_csv,
            **common,
        )
    if args.command == "dump-one":
        return dump_one(
            case_id=args.case_id,
            data_proc_name=args.data_proc_name,
            device_input=args.device_input,
            local_dir=args.local_dir,
            local_proc_dir=args.local_proc_dir,
            local_root=args.local_root,
            inventory_json=args.inventory_json,
            inventory_csv=args.inventory_csv,
            proc_inventory_json=args.proc_inventory_json,
            proc_inventory_csv=args.proc_inventory_csv,
            **common,
        )
    raise SystemExit(f"Unknown command: {args.command}")


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
