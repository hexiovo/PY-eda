"""Create the portable Windows archive, configuration-skill archive, and checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tomllib
import zipfile

PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_portable_release(version: str, bundle: Path) -> Path:
    expected_root = (PROJECT / "dist" / f"v{version}").resolve()
    if not bundle.is_relative_to(expected_root) or bundle.name != f"EDA_Toolkit-v{version}":
        raise ValueError(f"Release bundle path/version mismatch: {bundle}")
    for executable in ("eda_gui.exe", "eda_cli.exe"):
        if not (bundle / executable).is_file():
            raise FileNotFoundError(f"Missing executable: {bundle / executable}")
    info_path = bundle / "_internal" / "eda_toolkit" / "resources" / "build_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing packaged build information: {info_path}")
    if json.loads(info_path.read_text(encoding="utf-8")) .get("version") != version:
        raise ValueError("Packaged version information does not match the release version.")

    manifest_path = bundle / "release_manifest.json"
    archive = PROJECT / "dist" / f"EDA_Toolkit-{version}-Windows-x64.zip"
    if manifest_path.exists() or archive.exists():
        raise FileExistsError("Refusing to overwrite an existing release manifest or archive.")
    files = [
        {"path": path.relative_to(bundle).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(bundle.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "schema_version": 1,
        "version": version,
        "platform": "Windows x64",
        "executables": {
            name: {"sha256": sha256(bundle / name), "size": (bundle / name).stat().st_size}
            for name in ("eda_gui.exe", "eda_cli.exe")
        },
        "file_count_excluding_manifest": len(files),
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
            output.write(path, Path(bundle.name) / path.relative_to(bundle))
    verify_portable_archive(archive, bundle.name, manifest)
    return archive


def verify_portable_archive(archive: Path, root_name: str, manifest: dict) -> None:
    with zipfile.ZipFile(archive) as source:
        names = set(source.namelist())
        expected = {f"{root_name}/{item['path']}" for item in manifest["files"]}
        expected.add(f"{root_name}/release_manifest.json")
        if names != expected:
            raise ValueError("Portable archive members do not match release_manifest.json.")
        for item in manifest["files"]:
            member = f"{root_name}/{item['path']}"
            with source.open(member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"Archive hash mismatch: {member}")
        archived_manifest = json.loads(source.read(f"{root_name}/release_manifest.json"))
        if archived_manifest != manifest:
            raise ValueError("Archived release manifest differs from the source manifest.")


def package_skill(version: str) -> Path:
    skill_root = PROJECT / "skills" / "eda-study-config"
    readme = PROJECT / "skills" / "README.md"
    installer_candidates = [PROJECT / "skills" / "install_skill.ps1",
                            PROJECT / "skills" / "install_eda_study_config.ps1"]
    installer = next((path for path in installer_candidates if path.is_file()), None)
    if not (skill_root / "SKILL.md").is_file() or not readme.is_file() or installer is None:
        raise FileNotFoundError("The EDA study-config skill, README, or installation script is missing.")

    archive = PROJECT / "dist" / f"eda-study-config-{version}.zip"
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing skill archive: {archive}")
    skip_names = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    skill_files = [path for path in sorted(skill_root.rglob("*"))
                   if path.is_file()
                   and not any(part in skip_names for part in path.relative_to(skill_root).parts)
                   and path.suffix.lower() not in {".pyc", ".pyo"}
                   and path.name != "validation-summary.json"]
    if not skill_files:
        raise ValueError("The skill archive would be empty.")

    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        output.write(readme, "README.md")
        output.write(installer, "install_skill.ps1")
        for path in skill_files:
            output.write(path, Path("eda-study-config") / path.relative_to(skill_root))
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if "eda-study-config/SKILL.md" not in names or any(
                "__pycache__" in name or name.endswith((".pyc", "validation-summary.json")) for name in names):
            raise ValueError("The skill archive contains missing or temporary files.")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    version = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if args.version != version:
        raise ValueError(f"Requested version {args.version} does not match pyproject.toml ({version}).")
    bundle = args.bundle.resolve()
    portable = write_portable_release(args.version, bundle)
    skill = package_skill(args.version)

    release_notes = PROJECT / "docs" / f"发布说明-v{args.version}.md"
    if not release_notes.is_file():
        raise FileNotFoundError(f"Release notes are missing: {release_notes}")
    notes_copy = PROJECT / "dist" / release_notes.name
    shutil.copy2(release_notes, notes_copy)
    checksum_path = PROJECT / "dist" / "SHA256SUMS.txt"
    lines = [f"{sha256(path)}  {path.name}" for path in (portable, skill)]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps({
        "version": args.version,
        "portable_archive": str(portable),
        "skill_archive": str(skill),
        "release_notes": str(notes_copy),
        "checksums": str(checksum_path),
        "portable_sha256": sha256(portable),
        "skill_sha256": sha256(skill),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
