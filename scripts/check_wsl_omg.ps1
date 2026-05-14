$ErrorActionPreference = "Continue"
$Distro = "Ubuntu-22.04"
$LogPath = "C:\Users\11520\wsl-omg-check.log"

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

Write-Host "=== OMG check started: $(Get-Date -Format o) ==="
$cmd = @'
set -euo pipefail
mkdir -p /root/internvl-cann-vit-poc/scripts
cp /mnt/d/proj/internvl-cann-vit-poc/scripts/convert_onnx_to_om.sh /root/internvl-cann-vit-poc/scripts/
cp /mnt/d/proj/internvl-cann-vit-poc/scripts/check_wsl_omg.sh /root/internvl-cann-vit-poc/scripts/
chmod +x /root/internvl-cann-vit-poc/scripts/convert_onnx_to_om.sh
chmod +x /root/internvl-cann-vit-poc/scripts/check_wsl_omg.sh
/root/internvl-cann-vit-poc/scripts/check_wsl_omg.sh
'@

Invoke-WslChecked -Description "WSL OMG check" -Arguments @("-d", $Distro, "--", "bash", "-lc", $cmd)

Write-Host "=== OMG check finished: $(Get-Date -Format o) ==="
Stop-Transcript
