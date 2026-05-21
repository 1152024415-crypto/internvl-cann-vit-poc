# Dump Status 1 Diagnostic Logs

Captured from phone-side `model_run_tool_internal --enable_item=2` on:

```text
Device: 5ZU0125529000402
Build: DEL 6.0.0.271
API: 26
Model: split_internvit_v6_exclude_conv_int8_8.omc
Input: 000000018463_pixel_values_fp32.bin
```

## What To Read First

```text
02_dump_command_stdout.txt
```

Shows the user-facing failure:

```text
Dump is enabled.
Inference: loading model split_internvit_v6_exclude_conv_int8_8 succeeded.
Inference: filling input tensors succeeded.
Run model failed. status:1.
No profiling/dump file.
```

```text
05_hilog_filtered.txt
```

Main system log to inspect. Key lines:

```text
DirectEngine Model Execute failed: errCode=0x30
rtClientExecuteModel: errno = 0x56080030
rtModelExecute failed: 0x56080030
DumpBinaryFile: write op output file failed
model maintaince failed, error = 0x56060004
FdLeakInfo: hiaiserver fd handler leaked, fdcnt:7567
FdLeakDumpState: leakType: dmabuf not support
```

```text
03_fd_after_and_limits.txt
```

Resource evidence after the failed dump:

```text
after_fd=7555
Max open files = 32768
many fd entries point to anon_inode:dmabuf
```

```text
faultlog/cppcrash-hiaiserver-0-19700306072839123.log
faultlog/crashjsonstack-12259-5527719123
```

Native crash evidence:

```text
Process name: hiaiserver
Reason: Signal:SIGSEGV(SEGV_MAPERR)
#00 /vendor/lib64/libai_infra_rpc_server.so
```

## Interpretation

This is not just a generic `status:1`.

The concrete failure chain is:

```text
enable_item=2 starts dump
model load succeeds
input fill succeeds
rtModelExecute fails with 0x56080030
dump output write fails
hiaiserver leaks thousands of dmabuf fds
faultlog records hiaiserver native crash in libai_infra_rpc_server.so
```

Normal inference without `--enable_item=2` succeeds, so the current debugging
path should avoid full runtime dump and use debug multi-output OMC instead.

## Device Cleanup

After this failed dump, the phone was rebooted. The final clean state is in:

```text
09_after_reboot_fd.txt
```

Expected clean state:

```text
fd_count=33
```
