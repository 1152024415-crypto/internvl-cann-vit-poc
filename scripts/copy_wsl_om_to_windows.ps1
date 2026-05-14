$ErrorActionPreference = "Stop"

$Distro = "Ubuntu-22.04"
$WslOm = "/root/internvl-cann-vit-poc/artifacts/om/internvl3_5_vit_fp32_opset18_staticpos.om"
$WinOmDir = "D:\proj\internvl-cann-vit-poc\artifacts\om"
$WinOmWslDir = "/mnt/d/proj/internvl-cann-vit-poc/artifacts/om"
$LogPath = "C:\Users\11520\wsl-om-copy.log"

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
    Write-Host "=== OM copy started: $(Get-Date -Format o) ==="
    Write-Host "Distro: $Distro"
    Write-Host "WSL OM: $WslOm"
    Write-Host "Windows OM dir: $WinOmDir"

    New-Item -ItemType Directory -Force -Path $WinOmDir | Out-Null

    $cmd = @'
set -euo pipefail
echo "--- source ---"
ls -lh "__WSL_OM__"
echo "--- target before ---"
mkdir -p "__WIN_OM_WSL_DIR__"
ls -lh "__WIN_OM_WSL_DIR__"
echo "--- copy ---"
cp "__WSL_OM__" "__WIN_OM_WSL_DIR__/"
echo "--- target after ---"
ls -lh "__WIN_OM_WSL_DIR__"
'@
    $cmd = $cmd.Replace("__WSL_OM__", $WslOm)
    $cmd = $cmd.Replace("__WIN_OM_WSL_DIR__", $WinOmWslDir)

    $scriptRoot = Join-Path $env:USERPROFILE "wsl-cann-tools"
    New-Item -ItemType Directory -Force -Path $scriptRoot | Out-Null
    $scriptPath = Join-Path $scriptRoot "copy-om-to-windows.sh"
    $linuxScript = $cmd -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($scriptPath, $linuxScript, [System.Text.UTF8Encoding]::new($false))
    $wslScriptPath = Convert-ToWslPath -WindowsPath $scriptPath
    Write-Host "WSL copy script: $wslScriptPath"

    Invoke-WslChecked -Description "WSL OM copy" -Arguments @("-d", $Distro, "--", "bash", $wslScriptPath)
    Write-Host "=== OM copy finished: $(Get-Date -Format o) ==="
}
finally {
    Stop-Transcript
}
