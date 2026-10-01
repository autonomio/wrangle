"""Compare wheel and source-archive bytes from two builds in the same environment."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run() -> None:
    environment = {**os.environ, 'SOURCE_DATE_EPOCH': os.environ.get('SOURCE_DATE_EPOCH', '946684800'),
                   'PYTHONHASHSEED': '0'}
    with tempfile.TemporaryDirectory(prefix='wrangle-repeat-build-') as temporary:
        builds = []
        for name in ('first', 'second'):
            destination = Path(temporary) / name
            subprocess.run([sys.executable, '-m', 'build', '--no-isolation', '--outdir', str(destination), str(ROOT)],
                           check=True, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            artifacts = sorted(destination.iterdir())
            if len(artifacts) != 2 or sum(path.suffix == '.whl' for path in artifacts) != 1 or sum(path.name.endswith('.tar.gz') for path in artifacts) != 1:
                raise RuntimeError('Each build must produce exactly one wheel and one source archive.')
            builds.append({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts})
        if builds[0] != builds[1]:
            raise RuntimeError('Build outputs differ byte-for-byte; do not claim a reproducible release.')
        for name, digest in builds[0].items():
            print(f'{digest}  {name}')
        print('Two builds produced identical wheel and source-archive bytes in this environment.')


if __name__ == '__main__':
    try:
        run()
    except subprocess.CalledProcessError as error:
        sys.stderr.write(error.stdout.decode('utf-8', errors='replace') if isinstance(error.stdout, bytes) else error.stdout or '')
        raise
