$ErrorActionPreference = "Stop"

$Distro = "Ubuntu-22.04"
$WslProject = "/root/internvl-cann-vit-poc"
$WinProject = "D:\proj\internvl-cann-vit-poc"
$LogPath = "C:\Users\11520\wsl-stage3-prepare.log"

function Invoke-WslChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    & wsl @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

Start-Transcript -Path $LogPath -Force

Write-Host "=== Stage 3 WSL preparation started: $(Get-Date -Format o) ==="
Write-Host "Distro: $Distro"
Write-Host "Windows project: $WinProject"
Write-Host "WSL project: $WslProject"

Write-Host "`n--- WSL status ---"
Invoke-WslChecked -Description "WSL status check" -Arguments @("-d", $Distro, "--", "bash", "-lc", "whoami && uname -a && cat /etc/os-release | head -n 5")

Write-Host "`n--- Create WSL project directories and copy ONNX/script ---"
$cmd = @'
set -euo pipefail
mkdir -p '__WSL_PROJECT__/artifacts/onnx' '__WSL_PROJECT__/artifacts/om' '__WSL_PROJECT__/scripts' '__WSL_PROJECT__/docs'
cp '/mnt/d/proj/internvl-cann-vit-poc/artifacts/onnx/internvl3_5_vit_fp32_opset18.onnx' '__WSL_PROJECT__/artifacts/onnx/'
cp '/mnt/d/proj/internvl-cann-vit-poc/artifacts/onnx/internvl3_5_vit_fp32_opset18.onnx.metadata.json' '__WSL_PROJECT__/artifacts/onnx/'
cp '/mnt/d/proj/internvl-cann-vit-poc/artifacts/onnx/internvl3_5_vit_fp32_opset18_staticpos.onnx' '__WSL_PROJECT__/artifacts/onnx/'
cp '/mnt/d/proj/internvl-cann-vit-poc/artifacts/onnx/internvl3_5_vit_fp32_opset18_staticpos.onnx.metadata.json' '__WSL_PROJECT__/artifacts/onnx/'
cp '/mnt/d/proj/internvl-cann-vit-poc/scripts/convert_onnx_to_om.sh' '__WSL_PROJECT__/scripts/'
cp '/mnt/d/proj/internvl-cann-vit-poc/docs/stage-3-onnx-to-om.md' '__WSL_PROJECT__/docs/' || true
chmod +x '__WSL_PROJECT__/scripts/convert_onnx_to_om.sh'
ls -lh '__WSL_PROJECT__/artifacts/onnx' '__WSL_PROJECT__/scripts'
'@
$cmd = $cmd.Replace("__WSL_PROJECT__", $WslProject)
Invoke-WslChecked -Description "WSL stage 3 file preparation" -Arguments @("-d", $Distro, "--", "bash", "-lc", $cmd)

Write-Host "`n--- Check OMG availability ---"
$omgCheck = @'
set +e
echo "PATH=$PATH"
command -v omg
echo "command -v omg exit: $?"
find /root /opt /usr/local -path '*tools_omg/omg' -type f 2>/dev/null | head -n 20
'@
Invoke-WslChecked -Description "WSL OMG availability check" -Arguments @("-d", $Distro, "--", "bash", "-lc", $omgCheck)

Write-Host "=== Stage 3 WSL preparation finished: $(Get-Date -Format o) ==="
Stop-Transcript
