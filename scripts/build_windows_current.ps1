param([switch]$Clean)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$releaseVersion = '0.3.0'
$distRoot = Join-Path $projectRoot 'dist\v0.3'
$bundle = Join-Path $distRoot "EDA_Toolkit-v$releaseVersion"
$workRoot = Join-Path $projectRoot 'build\v0.3.0\pyinstaller'
$archive = Join-Path $projectRoot "dist\EDA_Toolkit-$releaseVersion-Windows-x64.zip"
$inventory = Join-Path $projectRoot 'scripts\release_inventory.py'
$spec = Join-Path $projectRoot 'build_support\eda.spec'

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw 'Project .venv is missing; refusing to use a global Python installation.'
}
if ((Test-Path -LiteralPath $bundle) -or (Test-Path -LiteralPath $archive)) {
    throw "Refusing to overwrite existing v$releaseVersion artifacts: $bundle or $archive"
}
if ((Test-Path -LiteralPath (Join-Path $projectRoot 'dist\EDA_Toolkit')) -or
    (Test-Path -LiteralPath (Join-Path $projectRoot 'dist\EDA_Toolkit-0.1.0-Windows-x64.zip'))) {
    Write-Host 'The existing v0.1 release artifacts were found and will be left untouched.'
}

Push-Location -LiteralPath $projectRoot
$oldReleaseVersion = $env:EDA_RELEASE_VERSION
try {
    # Refresh only this project's editable metadata with already-installed build tools.
    & $pythonExe -m pip install --editable . --no-deps --no-build-isolation
    if ($LASTEXITCODE -ne 0) { throw 'Could not refresh project-local editable metadata.' }

    # This phase writes the versioned build_info and current third-party inventory.
    & $pythonExe $inventory --version $releaseVersion --source-scope 'package_python_except_ui_v1' --build-dir (Join-Path $projectRoot 'build')
    if ($LASTEXITCODE -ne 0) { throw 'Version, source fingerprint, or installed metadata validation failed.' }

    $env:EDA_RELEASE_VERSION = $releaseVersion
    $buildArgs = @('-m', 'PyInstaller', '--noconfirm', '--distpath', $distRoot,
                   '--workpath', $workRoot, $spec)
    if ($Clean) { $buildArgs += '--clean' }
    & $pythonExe @buildArgs
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller v0.3 build failed; inspect the versioned build directory.' }
    if (-not (Test-Path -LiteralPath (Join-Path $bundle 'eda_cli.exe') -PathType Leaf) -or
        -not (Test-Path -LiteralPath (Join-Path $bundle 'eda_gui.exe') -PathType Leaf)) {
        throw "PyInstaller did not create the expected dual-EXE directory: $bundle"
    }

    # Rechecks the source and version snapshot before copying versioned documents and SBOM.
    & $pythonExe $inventory --version $releaseVersion --source-scope 'package_python_except_ui_v1' --build-dir (Join-Path $projectRoot 'build') --bundle $bundle
    if ($LASTEXITCODE -ne 0) { throw 'The source changed or inventory copy failed after the v0.3 executable build.' }
    Write-Host "Built isolated bundle: $bundle"
    Write-Host "Archive is created only by scripts/finalize_driver_release.py: $archive"
} finally {
    $env:EDA_RELEASE_VERSION = $oldReleaseVersion
    Pop-Location
}
