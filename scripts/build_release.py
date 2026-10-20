"""Build a deterministic combined release ZIP from this verified checkout."""
import argparse
import hashlib
from pathlib import Path
import zipfile

from check_artifacts import ROOT, check, is_artifact


def build(target):
    if target.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Write the ZIP outside the source checkout')
    check()
    paths = sorted(p for p in ROOT.rglob('*') if is_artifact(p))
    paths.append(ROOT / 'ARTIFACT_MANIFEST.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(paths):
            info = zipfile.ZipInfo('masif/' + path.relative_to(ROOT).as_posix(),
                                   date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert len(archive.infolist()) == len(paths)
        for info in archive.infolist():
            relative = info.filename.removeprefix('masif/')
            assert archive.read(info) == (ROOT / relative).read_bytes()
    return hashlib.sha256(target.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print('SHA256', build(args.output), args.output)
