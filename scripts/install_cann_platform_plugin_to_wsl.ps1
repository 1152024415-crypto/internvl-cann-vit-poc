param(
    [Parameter(Mandatory = $true)]
    [string[]]$PluginPackagePath,

    [string]$Distro = "Ubuntu-22.04",
    [string]$InstallRoot = "/root/cann-kit",
    [string]$LogPath = "C:\Users\11520\wsl-cann-platform-plugin-install.log"
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

function Expand-PackageIfNeeded {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $extension = [System.IO.Path]::GetExtension($resolved).ToLowerInvariant()
    if ($extension -eq ".zip") {
        $safeName = [System.IO.Path]::GetFileNameWithoutExtension($resolved) -replace '[^A-Za-z0-9_.-]', '_'
        $stamp = Get-Date -Format "yyyyMMddHHmmss"
        $expandRoot = Join-Path $env:USERPROFILE "wsl-cann-tools\platform-$safeName-$stamp"
        New-Item -ItemType Directory -Force -Path $expandRoot | Out-Null
        Write-Host "Expanding plugin zip to: $expandRoot"
        Expand-Archive -LiteralPath $resolved -DestinationPath $expandRoot -Force
        return $expandRoot
    }

    return $resolved
}

Start-Transcript -Path $LogPath -Force

try {
    Write-Host "=== CANN platform plugin import started: $(Get-Date -Format o) ==="
    Write-Host "Distro: $Distro"
    Write-Host "Install root: $InstallRoot"

    $wslSources = @()
    foreach ($path in $PluginPackagePath) {
        $source = Expand-PackageIfNeeded -Path $path
        Write-Host "Plugin source: $source"
        $wslSources += Convert-ToWslPath -WindowsPath $source
    }

    $sourceList = ($wslSources | ForEach-Object { "'$_'" }) -join " "
    $installScript = @'
set -euo pipefail

INSTALL_ROOT="__INSTALL_ROOT__"
PLATFORM_ROOT="${INSTALL_ROOT}/tools/platform"
mkdir -p "${PLATFORM_ROOT}"

copy_plugin_dir() {
  local dir="$1"
  local name
  name="$(basename "$dir")"
  if [[ "$name" != kirin* ]]; then
    echo "skip non-kirin plugin candidate: $dir"
    return 0
  fi
  echo "install platform plugin: ${name}"
  rm -rf "${PLATFORM_ROOT}/${name}"
  cp -a "$dir" "${PLATFORM_ROOT}/${name}"
}

for src in __SOURCES__; do
  if [[ -d "${src}/tools/platform" ]]; then
    for plugin in "${src}/tools/platform"/kirin*; do
      [[ -d "$plugin" ]] && copy_plugin_dir "$plugin"
    done
  elif [[ -d "${src}/platform" ]]; then
    for plugin in "${src}/platform"/kirin*; do
      [[ -d "$plugin" ]] && copy_plugin_dir "$plugin"
    done
  elif compgen -G "${src}/kirin*" >/dev/null; then
    for plugin in "${src}"/kirin*; do
      [[ -d "$plugin" ]] && copy_plugin_dir "$plugin"
    done
  elif [[ "$(basename "$src")" == kirin* ]]; then
    copy_plugin_dir "$src"
  elif [[ -f "$src" ]]; then
    tmpdir="$(mktemp -d)"
    case "$src" in
      *.tar.gz|*.tgz)
        tar -xzf "$src" -C "$tmpdir"
        ;;
      *.tar)
        tar -xf "$src" -C "$tmpdir"
        ;;
      *)
        echo "Unsupported plugin package file: $src" >&2
        exit 2
        ;;
    esac
    for plugin in "$tmpdir"/kirin* "$tmpdir"/platform/kirin* "$tmpdir"/tools/platform/kirin*; do
      [[ -d "$plugin" ]] && copy_plugin_dir "$plugin"
    done
  else
    echo "Plugin source not visible inside WSL: $src" >&2
    exit 3
  fi
done

echo "--- installed platform plugins ---"
find "${PLATFORM_ROOT}" -maxdepth 2 -type d -name 'kirin*' -print | sort
echo "--- key NPU libraries ---"
find "${PLATFORM_ROOT}" -type f \( -name 'libai_npucore_itf.so' -o -name 'libcustom_op.so' -o -name '*NPU*' \) -print | head -n 50
'@

    $installScript = $installScript.Replace("__INSTALL_ROOT__", $InstallRoot)
    $installScript = $installScript.Replace("__SOURCES__", $sourceList)

    $scriptRoot = Join-Path $env:USERPROFILE "wsl-cann-tools"
    New-Item -ItemType Directory -Force -Path $scriptRoot | Out-Null
    $scriptPath = Join-Path $scriptRoot "install-cann-platform-plugin.sh"
    $linuxScript = $installScript -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($scriptPath, $linuxScript, [System.Text.UTF8Encoding]::new($false))
    $wslScriptPath = Convert-ToWslPath -WindowsPath $scriptPath
    Write-Host "WSL platform plugin install script: $wslScriptPath"

    wsl -d $Distro -- bash $wslScriptPath
    Write-Host "=== CANN platform plugin import finished: $(Get-Date -Format o) ==="
}
finally {
    Stop-Transcript
}
