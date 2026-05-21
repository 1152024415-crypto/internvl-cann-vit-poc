@echo off
setlocal

if not defined CASE_ID (
  echo Usage: set CASE_ID=000000018463 then run %~nx0
  exit /b 2
)

if not defined HDC set "HDC=D:\software\deveco\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe"
if not defined DEVICE_DIR set "DEVICE_DIR=/data/local/tmp/internvl_om_compare"
if not defined MODEL_NAME set "MODEL_NAME=split_internvit_v6_exclude_conv_int8_8.omc"
if not defined RUNNER_NAME set "RUNNER_NAME=model_run_tool_internal"

powershell -NoProfile -Command "$ok = $env:CASE_ID -match '^[A-Za-z0-9_.-]+$' -and $env:DEVICE_DIR -match '^[A-Za-z0-9_./-]+$' -and $env:MODEL_NAME -match '^[A-Za-z0-9_.-]+$' -and $env:RUNNER_NAME -match '^[A-Za-z0-9_.-]+$' -and (!$env:HDC_TARGET -or $env:HDC_TARGET -match '^[A-Za-z0-9_.:-]+$'); if ($ok) { exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo Unsafe CASE_ID, DEVICE_DIR, MODEL_NAME, RUNNER_NAME, or HDC_TARGET
  exit /b 2
)

set "TARGET_ARGS="
if defined HDC_TARGET set "TARGET_ARGS=-t %HDC_TARGET%"

set "DEVICE_INPUT=%DEVICE_DIR%/inputs/%CASE_ID%_pixel_values_fp32.bin"
set "DEVICE_RUN=%DEVICE_DIR%/run_%CASE_ID%"
set "LOCAL_DIR=quantized_v6\layer_debug\run\%CASE_ID%"
set "RUN_LOG=quantized_v6\layer_debug\run\%CASE_ID%_run.log"

if not exist "quantized_v6\layer_debug\run" mkdir "quantized_v6\layer_debug\run"
"%HDC%" %TARGET_ARGS% shell "mkdir -p %DEVICE_RUN%"
"%HDC%" %TARGET_ARGS% shell "cd %DEVICE_DIR%; chmod +x %RUNNER_NAME%; export LD_LIBRARY_PATH=.; rm -rf run_%CASE_ID%; mkdir -p %DEVICE_RUN%; ./%RUNNER_NAME% --model=%DEVICE_DIR%/%MODEL_NAME% --input=%DEVICE_INPUT% --output_dir=%DEVICE_RUN% --times=1" > "%RUN_LOG%" 2>&1
if errorlevel 1 set "RUN_FAILED=1"
type "%RUN_LOG%"
if defined RUN_FAILED exit /b 1

if exist "%LOCAL_DIR%" rmdir /S /Q "%LOCAL_DIR%"
"%HDC%" %TARGET_ARGS% file recv "%DEVICE_RUN%" "%LOCAL_DIR%"
