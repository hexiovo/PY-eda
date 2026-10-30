[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$skillName = 'eda-study-config'
$source = Join-Path $PSScriptRoot $skillName
if (-not (Test-Path -LiteralPath (Join-Path $source 'SKILL.md') -PathType Leaf)) {
    throw "找不到技能目录：$source。请从包含 eda-study-config 子目录的源码或发布包运行此脚本。"
}

$skillsRoot = Join-Path $env:USERPROFILE '.codex\skills'
$destination = Join-Path $skillsRoot $skillName

if (Test-Path -LiteralPath $destination) {
    if (-not $Force) {
        throw "技能已存在：$destination。确认要更新时，请重新运行并添加 -Force。"
    }
    if (-not (Test-Path -LiteralPath $destination -PathType Container)) {
        throw "目标路径已存在且不是目录：$destination"
    }
}

New-Item -ItemType Directory -Path $skillsRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $destination -PathType Container)) {
    New-Item -ItemType Directory -Path $destination | Out-Null
}

Copy-Item -Path (Join-Path $source '*') -Destination $destination -Recurse -Force
Write-Host "技能已安装到：$destination"
