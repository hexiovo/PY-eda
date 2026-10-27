"""Bind successful verification to bundle hashes and verify the resulting ZIP."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from eda_toolkit.core_export import sha256_file


def main():
    bundle = PROJECT / "dist/v0.3/EDA_Toolkit-v0.3.0"
    evidence = PROJECT / "artifacts/verification/driver-v0.3.0"
    archive = PROJECT / "dist/EDA_Toolkit-0.3.0-Windows-x64.zip"
    if archive.exists() or (bundle / "release_manifest.json").exists():
        raise FileExistsError("Refusing to overwrite finalized release")
    report = json.loads((evidence / "frozen-verification.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed" and all(case["returncode"] == 0 for case in report["cases"])
    for name, digest in report["executables"].items():
        assert sha256_file(bundle / name) == digest, name
    suite = ET.parse(evidence / "full-suite.xml").getroot().find("testsuite")
    assert int(suite.attrib["tests"]) >= 209
    assert all(int(suite.attrib.get(key,0)) == 0 for key in ("failures", "errors", "skipped"))
    validation = bundle / "validation"
    validation.mkdir(exist_ok=True)
    for name in ("frozen-verification.json", "full-suite.xml", "legacy-exact.json"):
        shutil.copy2(evidence / name, validation / name)
    shutil.copy2(PROJECT / "docs/release-validation-v0.3.0.md", bundle / "docs/release-validation-v0.3.0.md")
    manifest = {"version":"0.3.0", "verification": report,
                "files": [{"path":p.relative_to(bundle).as_posix(), "size":p.stat().st_size,
                           "sha256":sha256_file(p)} for p in sorted(bundle.rglob("*")) if p.is_file()]}
    (bundle / "release_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():
                z.write(p, Path(bundle.name) / p.relative_to(bundle))
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == len(manifest["files"]) + 1
        for item in manifest["files"]:
            with z.open(bundle.name + "/" + item["path"]) as f:
                digest=hashlib.file_digest(f,"sha256").hexdigest()
            assert digest == item["sha256"], item["path"]
        assert z.read(bundle.name + "/release_manifest.json") == (bundle / "release_manifest.json").read_bytes()
    summary={"status":"passed","archive":str(archive),"sha256":sha256_file(archive),
             "bytes":archive.stat().st_size,"bundle_files":len(manifest["files"])+1}
    (evidence / "archive-verification.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
