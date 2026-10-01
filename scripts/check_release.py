"""Bind release tags and checksums to the wheel and source archive being shipped."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.10 contributor environments.
    import tomli as tomllib

from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def check_release(directory: str | Path, *, version: str, tag: str | None = None,
                  require_tag: bool = False, verify: bool = False) -> Path:
    """Check exact distribution identity, then write or verify portable checksums.

    This does not sign or publish. The release workflow attests the checked wheel,
    source archive and checksum manifest using its GitHub OIDC identity.
    """
    if require_tag and tag is None:
        raise ValueError('A published release tag is required.')
    if tag is not None and tag not in {version, 'v' + version}:
        raise ValueError(f'Release tag {tag!r} must match package version {version!r}.')
    directory = Path(directory)
    wheels, archives = sorted(directory.glob('*.whl')), sorted(directory.glob('*.tar.gz'))
    if len(wheels) != 1 or len(archives) != 1:
        raise ValueError('Build exactly one current wheel and source archive before release.')
    artifacts = [*wheels, *archives]
    allowed = {*artifacts, directory / 'SHA256SUMS'}
    if any(path not in allowed for path in directory.iterdir()):
        raise ValueError('The release directory contains unexpected files; build into a clean dist directory.')
    if any(path.is_symlink() or not path.is_file() for path in artifacts):
        raise ValueError('Release artifacts must be ordinary files inside the distribution directory.')
    wheel_name, wheel_version, _, _ = parse_wheel_filename(wheels[0].name)
    source_name, source_version = parse_sdist_filename(archives[0].name)
    if wheel_name != 'wrangle' or source_name != 'wrangle' or wheel_version != Version(version) or source_version != Version(version):
        raise ValueError('The wheel and source archive must both match the Wrangle package version.')
    expected = ''.join(f'{_digest(path)}  {path.name}\n' for path in sorted(artifacts))
    manifest = directory / 'SHA256SUMS'
    if manifest.is_symlink():
        raise ValueError('The checksum manifest must be an ordinary file in the distribution directory.')
    if verify:
        if not manifest.is_file() or manifest.read_text(encoding='utf-8') != expected:
            raise ValueError('Release artifacts or their checksum manifest changed; rebuild and verify before publication.')
    else:
        manifest.write_text(expected, encoding='utf-8')
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'dist')
    parser.add_argument('--tag', default=os.environ.get('RELEASE_TAG'))
    parser.add_argument('--require-tag', action='store_true')
    parser.add_argument('--verify', action='store_true')
    arguments = parser.parse_args()
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    try:
        manifest = check_release(arguments.directory, version=project['version'], tag=arguments.tag,
                                 require_tag=arguments.require_tag, verify=arguments.verify)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + '\n')
    print('Verified release artifacts: ' + str(manifest))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
