# API 22 Dump Validation

Date: 2026-05-21

Device build observed:

- Kernel: `HongMeng Kernel 1.11.0`
- Software: `DEL 6.0.0.271(HISlog)`
- API: `22`
- Device work dir: `/data/local/tmp/internvl_om_compare`

## Result

Plain OMC inference works on this firmware.

- V1: success
- V2 with `--direct_from_file=true`: success
- ROMC: success
- All three output files have the same SHA256:
  `e4437f3e10501b63214a94ceb64138d8099fcda0d72e61c94f962932ee694d9b`

Profiling works.

- `--enable_item=1` succeeds.
- `data_proc_tool_internal` generates:
  - `*_model.csv`
  - `*_op.csv`
- The op CSV contains NPU op names such as `/vision_model/encoder/layers.0/...`, task ids, device type, and per-op time.

Dump still fails.

- Command shape: `--modelmanager_itf=ROMC --enable_item=2 --times=1`
- The model loads and input filling succeeds.
- Failure occurs at run:
  `Inference: running model split_internvit_v6_exclude_conv_int8_8 failed`
- Return code: `RUN_EXIT=255`
- No dump file is produced.
- `data_proc_tool_internal` reports `No profiling/dump file`.
- `hiaiserver` fd count jumps from 34 to 7556 after the dump attempt.

After the failed dump attempt, the device was rebooted. `hiaiserver` returned to a clean fd count around 33.

## Current Interpretation

This firmware does not fix the CANN dump failure for this OMC. The failure is not normal OMC execution, because non-dump inference works across V1, V2, and ROMC. The failure is specifically tied to `--enable_item=2` dump mode, and it leaves `hiaiserver` in a polluted fd-leak state.

For the next debugging step, use profiling output as a stable source of OMC op names and execution order, but do not rely on `model_run_tool_internal --enable_item=2` for layer tensors on this firmware.

