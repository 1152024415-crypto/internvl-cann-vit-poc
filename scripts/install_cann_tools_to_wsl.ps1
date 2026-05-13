param(
    [Parameter(Mandatory = $true)]
    [string]$PackagePath,

    [string]$Distro = "Ubuntu-22.04",
    [string]$InstallRoot = "/root/cann-kit",
    [string]$LogPath = "C:\Users\11520\wsl-cann-tools-install.log"
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

Start-Transcript -Path $LogPath -Force

try {
    $resolvedPackage = (Resolve-Path -LiteralPath $PackagePath).Path
    Write-Host "=== CANN tools import started: $(Get-Date -Format o) ==="
    Write-Host "Package: $resolvedPackage"
    Write-Host "Distro: $Distro"
    Write-Host "Install root: $InstallRoot"

    $sourceForWsl = $resolvedPackage
    if ([System.IO.Path]::GetExtension($resolvedPackage).ToLowerInvariant() -eq ".zip") {
        $stamp = Get-Date -Format "yyyyMMddHHmmss"
        $expandRoot = Join-Path $env:USERPROFILE "wsl-cann-tools\expanded-$stamp"
        New-Item -ItemType Directory -Force -Path $expandRoot | Out-Null
        Write-Host "Expanding zip to: $expandRoot"
        Expand-Archive -LiteralPath $resolvedPackage -DestinationPath $expandRoot -Force
        $sourceForWsl = $expandRoot
    }

    $wslSource = Convert-ToWslPath -WindowsPath $sourceForWsl
    $installScript = @'
set -euo pipefail
mkdir -p "__INSTALL_ROOT__"

if [ -d "__WSL_SOURCE__" ]; then
  cp -a "__WSL_SOURCE__"/. "__INSTALL_ROOT__"/
elif [ -f "__WSL_SOURCE__" ]; then
  case "__WSL_SOURCE__" in
    *.tar.gz|*.tgz)
      tar -xzf "__WSL_SOURCE__" -C "__INSTALL_ROOT__"
      ;;
    *.tar)
      tar -xf "__WSL_SOURCE__" -C "__INSTALL_ROOT__"
      ;;
    *)
      echo "Unsupported package file. Use .zip, .tar.gz, .tgz, or .tar." >&2
      exit 2
      ;;
  esac
else
  echo "Source not visible inside WSL: __WSL_SOURCE__" >&2
  exit 3
fi

echo "--- imported files ---"
find "__INSTALL_ROOT__" -path '*/tools_omg/omg' -type f -print | head -n 20

OMG_PATH="$(find "__INSTALL_ROOT__" -path '*/tools_omg/omg' -type f -print | head -n 1 || true)"
if [ -z "$OMG_PATH" ]; then
  echo "OMG not found after import. Check whether this is the CANN Kit Tools package." >&2
  exit 4
fi

chmod +x "$OMG_PATH"
echo "OMG_PATH=$OMG_PATH"
'@

    $installScript = $installScript.Replace("__INSTALL_ROOT__", $InstallRoot)
    $installScript = $installScript.Replace("__WSL_SOURCE__", $wslSource)

    $scriptRoot = Join-Path $env:USERPROFILE "wsl-cann-tools"
    New-Item -ItemType Directory -Force -Path $scriptRoot | Out-Null
    $scriptPath = Join-Path $scriptRoot "install-cann-tools.sh"
    $linuxScript = $installScript -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($scriptPath, $linuxScript, [System.Text.UTF8Encoding]::new($false))
    $wslScriptPath = Convert-ToWslPath -WindowsPath $scriptPath
    Write-Host "WSL install script: $wslScriptPath"

    wsl -d $Distro -- bash $wslScriptPath

    Write-Host "=== CANN tools import finished: $(Get-Date -Format o) ==="
}
finally {
    Stop-Transcript
}
