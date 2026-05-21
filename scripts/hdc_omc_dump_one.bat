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
set "DEVICE_DUMP=%DEVICE_DIR%/dump_%CASE_ID%"
set "DEVICE_PROC=%DEVICE_DIR%/dump_proc_%CASE_ID%"
set "LOCAL_DIR=quantized_v6\layer_debug\omc\%CASE_ID%"
set "LOCAL_PROC_DIR=quantized_v6\layer_debug\omc\%CASE_ID%_proc"
set "INVENTORY_JSON=quantized_v6\layer_debug\omc\%CASE_ID%_inventory.json"
set "INVENTORY_CSV=quantized_v6\layer_debug\omc\%CASE_ID%_inventory.csv"
set "RUN_LOG=quantized_v6\layer_debug\omc\%CASE_ID%_run.log"
set "DATA_PROC_LOG=quantized_v6\layer_debug\omc\%CASE_ID%_data_proc.log"

"%HDC%" %TARGET_ARGS% shell "if test -f %DEVICE_INPUT%; then echo OK_INPUT; else echo MISSING_INPUT; fi" | findstr /C:"OK_INPUT" >nul
if errorlevel 1 (
  echo Missing device input: %DEVICE_INPUT%
  exit /b 1
)

if not exist "quantized_v6\layer_debug\omc" mkdir "quantized_v6\layer_debug\omc"
"%HDC%" %TARGET_ARGS% shell "mkdir -p %DEVICE_DUMP%"
"%HDC%" %TARGET_ARGS% shell "cd %DEVICE_DIR%; chmod +x %RUNNER_NAME%; export LD_LIBRARY_PATH=.; rm -rf dump_%CASE_ID%; mkdir -p %DEVICE_DUMP%; ./%RUNNER_NAME% --model=%DEVICE_DIR%/%MODEL_NAME% --input=%DEVICE_INPUT% --output_dir=%DEVICE_DUMP% --enable_item=2 --times=1" > "%RUN_LOG%" 2>&1
set "RUN_STATUS=%ERRORLEVEL%"
type "%RUN_LOG%"
if not "%RUN_STATUS%"=="0" exit /b %RUN_STATUS%
"%HDC%" %TARGET_ARGS% shell "mkdir -p %DEVICE_PROC%"
"%HDC%" %TARGET_ARGS% shell "cd %DEVICE_DIR%; chmod +x data_proc_tool_internal 2>/dev/null || true; export LD_LIBRARY_PATH=.; rm -rf dump_proc_%CASE_ID%; mkdir -p %DEVICE_PROC%; ./data_proc_tool_internal --result_path=%DEVICE_PROC%" > "%DATA_PROC_LOG%" 2>&1
set "DATA_PROC_STATUS=%ERRORLEVEL%"
type "%DATA_PROC_LOG%"
if not "%DATA_PROC_STATUS%"=="0" exit /b %DATA_PROC_STATUS%

"%HDC%" %TARGET_ARGS% shell "if test -d %DEVICE_DUMP%; then echo OK_DUMP; else echo MISSING_DUMP; fi" | findstr /C:"OK_DUMP" >nul
if errorlevel 1 (
  echo Missing device dump: %DEVICE_DUMP%
  exit /b 1
)

if exist "%LOCAL_DIR%" rmdir /S /Q "%LOCAL_DIR%"
if exist "%LOCAL_PROC_DIR%" rmdir /S /Q "%LOCAL_PROC_DIR%"
"%HDC%" %TARGET_ARGS% file recv "%DEVICE_DUMP%" "%LOCAL_DIR%"
if errorlevel 1 exit /b 1
"%HDC%" %TARGET_ARGS% file recv "%DEVICE_PROC%" "%LOCAL_PROC_DIR%"

uv run --no-sync python scripts/compare_omc_layers.py inventory --dump-dir "%LOCAL_DIR%" --output-json "%INVENTORY_JSON%" --output-csv "%INVENTORY_CSV%"
