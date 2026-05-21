@echo off
setlocal

if not defined HDC set "HDC=D:\software\deveco\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe"
if not defined DEVICE_DIR set "DEVICE_DIR=/data/local/tmp/internvl_om_compare"
if not defined MODEL_NAME set "MODEL_NAME=split_internvit_v6_exclude_conv_int8_8.omc"
if not defined RUNNER_NAME set "RUNNER_NAME=model_run_tool_internal"

powershell -NoProfile -Command "$ok = $env:DEVICE_DIR -match '^[A-Za-z0-9_./-]+$' -and $env:MODEL_NAME -match '^[A-Za-z0-9_.-]+$' -and $env:RUNNER_NAME -match '^[A-Za-z0-9_.-]+$' -and (!$env:HDC_TARGET -or $env:HDC_TARGET -match '^[A-Za-z0-9_.:-]+$'); if ($ok) { exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo Unsafe DEVICE_DIR, MODEL_NAME, RUNNER_NAME, or HDC_TARGET
  exit /b 2
)

set "TARGET_ARGS="
if defined HDC_TARGET set "TARGET_ARGS=-t %HDC_TARGET%"

"%HDC%" list targets
"%HDC%" %TARGET_ARGS% shell "if test -d %DEVICE_DIR%; then echo OK_DIR; else echo MISSING_DIR; fi"
"%HDC%" %TARGET_ARGS% shell "if test -f %DEVICE_DIR%/%RUNNER_NAME%; then echo OK_RUNNER; else echo MISSING_RUNNER; fi"
"%HDC%" %TARGET_ARGS% shell "if test -f %DEVICE_DIR%/%MODEL_NAME%; then echo OK_MODEL; else echo MISSING_MODEL; fi"
