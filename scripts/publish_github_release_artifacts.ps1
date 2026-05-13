param(
    [string]$Owner = "1152024415-crypto",
    [string]$Repo = "internvl-cann-vit-poc",
    [string]$Tag = "v0.1.0-artifacts",
    [string]$LogPath = "C:\Users\11520\github-release-artifacts.log"
)

$ErrorActionPreference = "Stop"

function Get-GitHubToken {
    $credentialInput = "protocol=https`nhost=github.com`n`n"
    $credentialOutput = $credentialInput | git credential fill
    $passwordLine = $credentialOutput | Where-Object { $_ -like "password=*" } | Select-Object -First 1
    if (-not $passwordLine) {
        throw "No GitHub token was returned by Git Credential Manager. Log in to GitHub and retry."
    }
    return $passwordLine.Substring("password=".Length)
}

function Invoke-GitHubJson {
    param(
        [Parameter(Mandatory = $true)][string]$Method,
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][hashtable]$Headers,
        [object]$Body = $null
    )

    if ($null -eq $Body) {
        return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $Headers
    }

    $json = $Body | ConvertTo-Json -Depth 8
    return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $Headers -ContentType "application/json" -Body $json
}

function Get-FileSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Upload-ReleaseAsset {
    param(
        [Parameter(Mandatory = $true)][string]$UploadBaseUrl,
        [Parameter(Mandatory = $true)][hashtable]$Headers,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $escapedName = [System.Uri]::EscapeDataString($Name)
    $uploadUri = "$UploadBaseUrl?name=$escapedName"
    $uploadHeaders = $Headers.Clone()
    $uploadHeaders["Content-Type"] = "application/octet-stream"

    Write-Host "Uploading: $Name"
    return Invoke-RestMethod `
        -Method Post `
        -Uri $uploadUri `
        -Headers $uploadHeaders `
        -InFile $Path `
        -ContentType "application/octet-stream"
}

Start-Transcript -Path $LogPath -Force

try {
    $projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
    $manifestDir = Join-Path $projectRoot "artifacts\release"
    New-Item -ItemType Directory -Force -Path $manifestDir | Out-Null

    $assets = @(
        @{
            Name = "internvl3_5_vit_projector_fp32_opset18_staticpos.onnx"
            Path = Join-Path $projectRoot "artifacts\onnx\internvl3_5_vit_projector_fp32_opset18_staticpos.onnx"
            Description = "ViT + projector ONNX, fp32, opset 18, static 1x3x448x448 input."
        },
        @{
            Name = "internvl3_5_vit_projector_fp32_opset18_staticpos.onnx.metadata.json"
            Path = Join-Path $projectRoot "artifacts\onnx\internvl3_5_vit_projector_fp32_opset18_staticpos.onnx.metadata.json"
            Description = "Metadata for the ViT + projector ONNX artifact."
        },
        @{
            Name = "internvl3_5_vit_projector_fp32_opset18_staticpos.om"
            Path = Join-Path $projectRoot "artifacts\om\internvl3_5_vit_projector_fp32_opset18_staticpos.om"
            Description = "CANN Kit OMG output for the ViT + projector ONNX artifact."
        },
        @{
            Name = "onnx-projector-dog-output.pt"
            Path = Join-Path $projectRoot "artifacts\onnx-projector-dog-output.pt"
            Description = "ONNX Runtime visual_tokens output for dog.jpg."
        },
        @{
            Name = "onnx-projector-cat-output.pt"
            Path = Join-Path $projectRoot "artifacts\onnx-projector-cat-output.pt"
            Description = "ONNX Runtime visual_tokens output for cat.jpg."
        },
        @{
            Name = "pytorch-projector-dog-baseline_output.pt"
            Path = Join-Path $projectRoot "artifacts\baseline-projector-dog\baseline_output.pt"
            Description = "PyTorch projector baseline tensor for dog.jpg."
        },
        @{
            Name = "pytorch-projector-cat-baseline_output.pt"
            Path = Join-Path $projectRoot "artifacts\baseline-projector-cat\baseline_output.pt"
            Description = "PyTorch projector baseline tensor for cat.jpg."
        }
    )

    foreach ($asset in $assets) {
        if (-not (Test-Path -LiteralPath $asset.Path)) {
            throw "Missing release asset: $($asset.Path)"
        }
    }

    $manifestAssets = foreach ($asset in $assets) {
        $item = Get-Item -LiteralPath $asset.Path
        [ordered]@{
            name = $asset.Name
            local_path = $asset.Path.Replace($projectRoot, "").TrimStart("\")
            bytes = [int64]$item.Length
            sha256 = Get-FileSha256 -Path $asset.Path
            description = $asset.Description
        }
    }

    $manifest = [ordered]@{
        repository = "https://github.com/$Owner/$Repo"
        tag = $Tag
        created_from_commit = (git -C $projectRoot rev-parse HEAD).Trim()
        note = "DDK-tools zip is intentionally not uploaded because it should be downloaded from Huawei's official site."
        assets = $manifestAssets
    }

    $manifestPath = Join-Path $manifestDir "artifacts-manifest.json"
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $manifestPath -Encoding utf8

    $shaTextPath = Join-Path $manifestDir "artifacts-manifest.sha256.txt"
    $manifestAssets | ForEach-Object { "$($_.sha256)  $($_.name)" } | Set-Content -Path $shaTextPath -Encoding ascii

    $assets += @{
        Name = "artifacts-manifest.json"
        Path = $manifestPath
        Description = "Release artifact manifest with sizes and SHA256 hashes."
    }
    $assets += @{
        Name = "artifacts-manifest.sha256.txt"
        Path = $shaTextPath
        Description = "SHA256 checksum list for release assets."
    }

    Write-Host "=== GitHub release artifact publish started: $(Get-Date -Format o) ==="
    Write-Host "Repository: $Owner/$Repo"
    Write-Host "Tag: $Tag"
    Write-Host "Assets: $($assets.Count)"

    $token = Get-GitHubToken
    $headers = @{
        "Authorization" = "Bearer $token"
        "Accept" = "application/vnd.github+json"
        "X-GitHub-Api-Version" = "2022-11-28"
        "User-Agent" = "internvl-cann-vit-poc-release-publisher"
    }

    $release = $null
    try {
        $release = Invoke-GitHubJson -Method Get -Uri "https://api.github.com/repos/$Owner/$Repo/releases/tags/$Tag" -Headers $headers
        Write-Host "Using existing release id=$($release.id)"
    }
    catch {
        if ($_.Exception.Response.StatusCode.value__ -ne 404) {
            throw
        }
        Write-Host "Creating release $Tag"
        $body = @{
            tag_name = $Tag
            target_commitish = "main"
            name = "ViT + projector artifacts $Tag"
            body = @"
Artifacts for the fixed-shape InternVL3.5 ViT + projector PoC.

Included:
- projector ONNX
- projector OM
- ONNX Runtime output tensors
- PyTorch baseline tensors
- SHA256 manifest

Not included:
- Huawei DDK/CANN tools package. Download it from Huawei's official site and verify the SHA256 recorded in docs.
"@
            draft = $false
            prerelease = $true
        }
        $release = Invoke-GitHubJson -Method Post -Uri "https://api.github.com/repos/$Owner/$Repo/releases" -Headers $headers -Body $body
        Write-Host "Created release id=$($release.id)"
    }

    $uploadBaseUrl = ($release.upload_url -replace "\{\?name,label\}", "")
    $existingAssets = Invoke-GitHubJson -Method Get -Uri "https://api.github.com/repos/$Owner/$Repo/releases/$($release.id)/assets?per_page=100" -Headers $headers
    $existingNames = @{}
    foreach ($existing in $existingAssets) {
        $existingNames[$existing.name] = $true
    }

    foreach ($asset in $assets) {
        if ($existingNames.ContainsKey($asset.Name)) {
            Write-Host "Skipping existing asset: $($asset.Name)"
            continue
        }
        Upload-ReleaseAsset -UploadBaseUrl $uploadBaseUrl -Headers $headers -Path $asset.Path -Name $asset.Name | Out-Null
    }

    Write-Host "Release URL: $($release.html_url)"
    Write-Host "=== GitHub release artifact publish finished: $(Get-Date -Format o) ==="
}
finally {
    Stop-Transcript
}
