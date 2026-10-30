param([switch]$Clean)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$releaseVersion = '0.3.1'
$distRoot = Join-Path $projectRoot 'dist\v0.3.1'
$bundle = Join-Path $distRoot "EDA_Toolkit-v$releaseVersion"
$workRoot = Join-Path $projectRoot 'build\v0.3.1\pyinstaller'
$archive = Join-Path $projectRoot "dist\EDA_Toolkit-$releaseVersion-Windows-x64.zip"
$inventory = Join-Path $projectRoot 'scripts\release_inventory.py'
$finalize = Join-Path $projectRoot 'scripts\finalize_driver_release.py'
$helpBuilder = Join-Path $projectRoot 'scripts\build_help.py'
$spec = Join-Path $projectRoot 'build_support\eda.spec'

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw 'Project .venv is missing; refusing to use a global Python installation.'
}
if ((Test-Path -LiteralPath $bundle) -or (Test-Path -LiteralPath $archive)) {
    throw "Refusing to overwrite existing v$releaseVersion artifacts: $bundle or $archive"
}
Push-Location -LiteralPath $projectRoot
$oldReleaseVersion = $env:EDA_RELEASE_VERSION
try {
    # Refresh only this project's editable metadata with already-installed build tools.
    & $pythonExe -m pip install --editable . --no-deps --no-build-isolation
    if ($LASTEXITCODE -ne 0) { throw 'Could not refresh project-local editable metadata.' }

    # Rebuild the in-app guide and current screenshots from maintained source copy.
    & $pythonExe -X utf8 $helpBuilder
    if ($LASTEXITCODE -ne 0) { throw 'The Chinese user guide could not be rebuilt.' }
    $screenshotSource = Join-Path $projectRoot 'artifacts\verification\v0.3.1\help-screenshots'
    $screenshotTarget = Join-Path $projectRoot 'docs\screenshots'
    $screenshotNames = @('01-startup.png','02-import-mapping.png','03-qc-history.png',
                         '04-config.png','05-preprocess.png','06-run.png','07-result.png',
                         '08-export.png','09-help-menu.png')
    foreach ($name in $screenshotNames) {
        $source = Join-Path $screenshotSource $name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "The user guide screenshot is missing: $source"
        }
        Copy-Item -LiteralPath $source -Destination (Join-Path $screenshotTarget $name) -Force
    }

    # This phase writes the versioned build_info and current third-party inventory.
    & $pythonExe $inventory --version $releaseVersion --source-scope 'package_python_except_ui_v1' --build-dir (Join-Path $projectRoot 'build')
    if ($LASTEXITCODE -ne 0) { throw 'Version, source fingerprint, or installed metadata validation failed.' }

    $env:EDA_RELEASE_VERSION = $releaseVersion
    $buildArgs = @('-m', 'PyInstaller', '--noconfirm', '--distpath', $distRoot,
                   '--workpath', $workRoot, $spec)
    if ($Clean) { $buildArgs += '--clean' }
    & $pythonExe @buildArgs
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed; inspect the versioned build directory.' }
    if (-not (Test-Path -LiteralPath (Join-Path $bundle 'eda_cli.exe') -PathType Leaf) -or
        -not (Test-Path -LiteralPath (Join-Path $bundle 'eda_gui.exe') -PathType Leaf)) {
        throw "PyInstaller did not create the expected dual-EXE directory: $bundle"
    }

    # Rechecks the source and version snapshot before copying current documents and SBOM.
    & $pythonExe $inventory --version $releaseVersion --source-scope 'package_python_except_ui_v1' --build-dir (Join-Path $projectRoot 'build') --bundle $bundle
    if ($LASTEXITCODE -ne 0) { throw 'The source changed or release metadata could not be copied after the executable build.' }

    # Create the portable ZIP, skill ZIP, release notes, and checksums from the finished bundle.
    & $pythonExe $finalize --version $releaseVersion --bundle $bundle
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the release archives and checksums.' }
    Write-Host "Built portable release: $archive"
} finally {
    $env:EDA_RELEASE_VERSION = $oldReleaseVersion
    Pop-Location
}
