# Build ERA-<version>.msi from an existing PyInstaller output (dist\ERA).
#
#   pyinstaller ERA.spec --noconfirm
#   powershell -File installer\build_msi.ps1 -Version 2.0.0
#
# Requires WiX Toolset 3.x (heat/candle/light).  If it is not on PATH, the
# official portable binaries are downloaded into installer\wix.
param(
    [string]$Version = "2.0.0",
    [string]$DistDir = "dist\ERA",
    [string]$OutDir = "dist"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path $DistDir\ERA.exe)) {
    throw "$DistDir\ERA.exe not found: run 'pyinstaller ERA.spec --noconfirm' first."
}

# Locate WiX 3
$wixBin = $null
if (Get-Command candle.exe -ErrorAction SilentlyContinue) {
    $wixBin = Split-Path (Get-Command candle.exe).Source
} elseif ($env:WIX -and (Test-Path "$env:WIX\bin\candle.exe")) {
    $wixBin = "$env:WIX\bin"
} else {
    $wixBin = Join-Path $PSScriptRoot "wix"
    if (-not (Test-Path "$wixBin\candle.exe")) {
        Write-Host "WiX not found, downloading portable binaries..."
        $zip = Join-Path $env:TEMP "wix314-binaries.zip"
        Invoke-WebRequest -Uri "https://github.com/wixtoolset/wix3/releases/download/wix3141rtm/wix314-binaries.zip" -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath $wixBin -Force
        Remove-Item $zip
    }
}
Write-Host "Using WiX from $wixBin"

$obj = Join-Path $root "build\msi"
New-Item -ItemType Directory -Force -Path $obj | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Harvest dist\ERA (ERA.exe + _internal\...) into a component group.
& "$wixBin\heat.exe" dir $DistDir -cg ERAFiles -dr INSTALLFOLDER -srd -sreg -scom `
    -gg -g1 -sfrag -ke -var var.SourceDir -out "$obj\Files.wxs"
if ($LASTEXITCODE) { throw "heat failed" }

& "$wixBin\candle.exe" -nologo -arch x64 "-dVersion=$Version" "-dSourceDir=$DistDir" `
    -ext WixUIExtension -out "$obj\" installer\ERA.wxs "$obj\Files.wxs"
if ($LASTEXITCODE) { throw "candle failed" }

$msi = Join-Path $OutDir "ERA-$Version-windows-x64.msi"
# ICE errors that are noise for a harvested, per-machine file tree:
#   ICE60: unversioned files without language; ICE69: cross-component
#   shortcut reference (intended).
& "$wixBin\light.exe" -nologo -ext WixUIExtension -sice:ICE60 -sice:ICE69 `
    -out $msi "$obj\ERA.wixobj" "$obj\Files.wixobj"
if ($LASTEXITCODE) { throw "light failed" }

Write-Host "Built $msi ($([math]::Round((Get-Item $msi).Length / 1MB)) MB)"
