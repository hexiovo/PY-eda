$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$portable = Join-Path $projectRoot 'dist\v0.3.1\EDA_Toolkit-v0.3.1'
$portableZip = Join-Path $projectRoot 'dist\EDA_Toolkit-0.3.1-Windows-x64.zip'
$skillZip = Join-Path $projectRoot 'dist\eda-study-config-0.3.1.zip'
$checksums = Join-Path $projectRoot 'dist\SHA256SUMS.txt'

if (-not (Test-Path -LiteralPath $portable -PathType Container)) {
    throw "Current release assets are incomplete; build and verify v0.3.1 first: $portable"
}
foreach ($required in @((Join-Path $portable 'release_manifest.json'), $portableZip, $skillZip, $checksums)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Current release assets are incomplete; build and verify v0.3.1 first: $required"
    }
}
$manifest = Get-Content -LiteralPath (Join-Path $portable 'release_manifest.json') -Raw | ConvertFrom-Json
if ($manifest.version -ne '0.3.1') { throw 'Portable manifest version is not v0.3.1.' }
$checksumText = Get-Content -LiteralPath $checksums -Raw
if ($checksumText -notmatch [regex]::Escape((Split-Path -Leaf $portableZip)) -or
    $checksumText -notmatch [regex]::Escape((Split-Path -Leaf $skillZip))) {
    throw 'SHA256SUMS.txt does not list both current release archives.'
}

$allowlist = @(
    'artifacts', '.property-tests', 'examples\current_synthetic_demo\results',
    'examples\synthetic_demo\results', 'build\bdist.win-amd64', 'build\common-metrics',
    'build\eda', 'build\feasibility', 'build\feasibility-spec', 'build\feasibility-work',
    'build\lib', 'build\ui-refresh', 'build\ui-refresh-r2', 'build\ui-refresh-r3',
    'build\ui-refresh-r3-final', 'build\ui-refresh-r4-final', 'build\ui-refresh-r4-onefile',
    'build\v0.2.0', 'build\v0.3.0', 'build\v0.3.1',
    'dist\EDA_Toolkit', 'dist\ui-refresh', 'dist\ui-refresh-r2', 'dist\ui-refresh-r2-candidate',
    'dist\ui-refresh-r3', 'dist\ui-refresh-r3-final', 'dist\ui-refresh-r4-final',
    'dist\ui-refresh-r4-onefile', 'dist\v0.2', 'dist\v0.3',
    'dist\EDA_Toolkit-0.1.0-Windows-x64.zip', 'dist\EDA_Toolkit-0.2.0-Windows-x64-r2.zip',
    'dist\EDA_Toolkit-0.2.0-Windows-x64-r3.zip', 'dist\EDA_Toolkit-0.2.0-Windows-x64.zip',
    'dist\EDA_Toolkit-0.3.0-Windows-x64.zip', 'build_support\onefile',
    'build_support\__pycache__', 'scripts\__pycache__', 'src\eda_toolkit\__pycache__',
    'src\eda_toolkit\ui\__pycache__', 'src\eda_toolkit\resources\__pycache__',
    'tests\__pycache__', '.pytest_cache', 'skills\eda-study-config\scripts\__pycache__'
)
$failures = [System.Collections.Generic.List[string]]::new()
foreach ($relative in $allowlist) {
    $candidate = Join-Path $projectRoot $relative
    if (-not (Test-Path -LiteralPath $candidate)) { continue }
    $item = Get-Item -LiteralPath $candidate -Force
    $resolved = [System.IO.Path]::GetFullPath($item.FullName)
    if (-not $resolved.StartsWith($projectRoot + [System.IO.Path]::DirectorySeparatorChar,
                                  [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup target escapes the project: $resolved"
    }
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to follow a junction or symbolic link: $resolved"
    }
    try {
        Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction Stop
        Write-Host "Removed $relative"
    } catch {
        $failures.Add("$relative — $($_.Exception.Message)")
    }
}
if ($failures.Count -gt 0) {
    Write-Warning 'Some old files could not be removed. The script did not stop or close any application.'
    $failures | ForEach-Object { Write-Warning $_ }
    exit 1
}
Write-Host 'Old build artifacts, reports, and local caches were removed. Current v0.3.1 release files were preserved.'
