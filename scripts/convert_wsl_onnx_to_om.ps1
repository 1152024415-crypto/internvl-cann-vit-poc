param(
    [string]$OnnxFile = "D:\proj\internvl-cann-vit-poc\artifacts\onnx\internvl3_5_vit_projector_fp32_opset18_staticpos.onnx",
    [string]$OutputName = "internvl3_5_vit_projector_fp32_opset18_staticpos",
    [string]$Distro = "Ubuntu-22.04",
    [string]$LogPath = "C:\Users\11520\wsl-onnx-to-om.log"
)

$ErrorActionPreference = "Stop"

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$WindowsPath)

    $resolved = (Resolve-Path -LiteralPath $WindowsPath).Path
    if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Only local drive paths are supported: $resolved"
    }

    $drive = $Matches[1].ToLowerInvariant()
    $rest = $Matches[2] -replace '\\', '/'
    return "/mnt/$drive/$rest"
}

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

try {
    $projectRoot = "D:\proj\internvl-cann-vit-poc"
    $wslProject = "/root/internvl-cann-vit-poc"
    $winOmDir = Join-Path $projectRoot "artifacts\om"

    $resolvedOnnx = (Resolve-Path -LiteralPath $OnnxFile).Path
    $onnxName = Split-Path -Path $resolvedOnnx -Leaf
    $metadataPath = "$resolvedOnnx.metadata.json"
    $wslOnnx = Convert-ToWslPath -WindowsPath $resolvedOnnx
    $wslMetadata = if (Test-Path -LiteralPath $metadataPath) {
        Convert-ToWslPath -WindowsPath $metadataPath
    } else {
        ""
    }

    New-Item -ItemType Directory -Force -Path $winOmDir | Out-Null

    Write-Host "=== ONNX to OM started: $(Get-Date -Format o) ==="
    Write-Host "Distro: $Distro"
    Write-Host "ONNX: $resolvedOnnx"
    Write-Host "Output name: $OutputName"
    Write-Host "Windows OM dir: $winOmDir"

    $cmd = @'
set -euo pipefail

WSL_PROJECT="__WSL_PROJECT__"
ONNX_SRC="__WSL_ONNX__"
METADATA_SRC="__WSL_METADATA__"
ONNX_NAME="__ONNX_NAME__"
OUTPUT_NAME="__OUTPUT_NAME__"
WIN_PROJECT="/mnt/d/proj/internvl-cann-vit-poc"

mkdir -p "${WSL_PROJECT}/artifacts/onnx" "${WSL_PROJECT}/artifacts/om" "${WSL_PROJECT}/scripts" "${WIN_PROJECT}/artifacts/om"
cp "${ONNX_SRC}" "${WSL_PROJECT}/artifacts/onnx/${ONNX_NAME}"
if [ -n "${METADATA_SRC}" ] && [ -f "${METADATA_SRC}" ]; then
  cp "${METADATA_SRC}" "${WSL_PROJECT}/artifacts/onnx/${ONNX_NAME}.metadata.json"
fi
cp "${WIN_PROJECT}/scripts/convert_onnx_to_om.sh" "${WSL_PROJECT}/scripts/"
chmod +x "${WSL_PROJECT}/scripts/convert_onnx_to_om.sh"

echo "--- input ---"
ls -lh "${WSL_PROJECT}/artifacts/onnx/${ONNX_NAME}"
echo "--- convert ---"
cd "${WSL_PROJECT}"
./scripts/convert_onnx_to_om.sh "${WSL_PROJECT}/artifacts/onnx/${ONNX_NAME}" "${WSL_PROJECT}/artifacts/om/${OUTPUT_NAME}"
echo "--- WSL output ---"
ls -lh "${WSL_PROJECT}/artifacts/om/${OUTPUT_NAME}.om"
echo "--- copy to Windows ---"
cp "${WSL_PROJECT}/artifacts/om/${OUTPUT_NAME}.om" "${WIN_PROJECT}/artifacts/om/"
ls -lh "${WIN_PROJECT}/artifacts/om/${OUTPUT_NAME}.om"
'@
    $cmd = $cmd.Replace("__WSL_PROJECT__", $wslProject)
    $cmd = $cmd.Replace("__WSL_ONNX__", $wslOnnx)
    $cmd = $cmd.Replace("__WSL_METADATA__", $wslMetadata)
    $cmd = $cmd.Replace("__ONNX_NAME__", $onnxName)
    $cmd = $cmd.Replace("__OUTPUT_NAME__", $OutputName)

    $scriptRoot = Join-Path $env:USERPROFILE "wsl-cann-tools"
    New-Item -ItemType Directory -Force -Path $scriptRoot | Out-Null
    $scriptPath = Join-Path $scriptRoot "convert-onnx-to-om.sh"
    $linuxScript = $cmd -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($scriptPath, $linuxScript, [System.Text.UTF8Encoding]::new($false))
    $wslScriptPath = Convert-ToWslPath -WindowsPath $scriptPath
    Write-Host "WSL conversion script: $wslScriptPath"

    Invoke-WslChecked -Description "WSL ONNX to OM conversion" -Arguments @("-d", $Distro, "--", "bash", $wslScriptPath)
    Write-Host "=== ONNX to OM finished: $(Get-Date -Format o) ==="
}
finally {
    Stop-Transcript
}
