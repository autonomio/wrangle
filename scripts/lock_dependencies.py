"""Regenerate reviewed, hash-locked CI and scientific-runtime dependency plans.

Install requirements-ci.txt first, then run this script with --upgrade to refresh
transitive dependencies. Python package dependency ranges remain the public API.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
UV_VERSION = "0.12.22"
HASH_LINE = re.compile(r"    --hash=sha256:[0-9a-f]{64}(?: \\\s*)?$")


def definitions() -> dict[str, list[str]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    runtime = [*project["dependencies"], *project["optional-dependencies"]["excel"]]
    minimum = [
        "polars==1.34.0", "ruamel.yaml==0.18.6",
        *[value for value in runtime if not value.startswith(("polars", "ruamel.yaml"))],
    ]
    return {
        "requirements-ci.txt": [
            *project["optional-dependencies"]["dev"],
            "pip==26.2.1", f"uv=={UV_VERSION}", "editables==0.5",
        ],
        "requirements-audit.txt": runtime,
        "requirements-minimum.txt": minimum,
        "requirements-fuzz.txt": ["atheris==3.1.0"],
    }


def intent_digest(name: str, requirements: list[str]) -> str:
    target = "python==3.13\nplatform==x86_64-unknown-linux-gnu" if name == "requirements-fuzz.txt" else "python>=3.10\nuniversal"
    intent = f"uv=={UV_VERSION}\n{target}\n{name}\n" + "\n".join(requirements)
    if name == "requirements-ci.txt":
        intent += "\n" + (ROOT / "requirements-ci.in").read_text()
    return hashlib.sha256(intent.encode()).hexdigest()


def check_lock(name: str, requirements: list[str]) -> None:
    path = ROOT / name
    lines = path.read_text().splitlines()
    expected = f"# Intent SHA256: {intent_digest(name, requirements)}"
    if not lines or lines[0] != expected:
        raise SystemExit(f"{name} is stale; run python scripts/lock_dependencies.py")
    pinned: dict[str, list[Requirement]] = {}
    for index, line in enumerate(lines):
        if not line or line.startswith("#") or HASH_LINE.fullmatch(line):
            continue
        if not re.fullmatch(r"[a-z0-9_.-]+==[^ ;]+(?: ; .+)? \\", line):
            raise SystemExit(f"{name}:{index + 1} has a non-exact requirement")
        if index + 1 == len(lines) or not HASH_LINE.fullmatch(lines[index + 1]):
            raise SystemExit(f"{name}:{index + 1} lacks a SHA256 artifact hash")
        requirement = Requirement(line.removesuffix(" \\"))
        pinned.setdefault(canonicalize_name(requirement.name), []).append(requirement)
    if not pinned:
        raise SystemExit(f"{name} contains no locked dependencies")
    expected_requirements = list(requirements)
    if name == "requirements-ci.txt":
        expected_requirements.extend(
            line for line in (ROOT / "requirements-ci.in").read_text().splitlines()
            if line and not line.startswith("#")
        )
    for specification in expected_requirements:
        expected_requirement = Requirement(specification)
        candidates = pinned.get(canonicalize_name(expected_requirement.name), [])
        if not candidates or any(
            next(iter(candidate.specifier)).version not in expected_requirement.specifier
            for candidate in candidates
        ):
            raise SystemExit(f"{name} does not satisfy reviewed requirement {specification}")
    print(f"{name}: {len(pinned)} packages, exact versions, SHA256 hashes, current intent")


def generate(name: str, requirements: list[str], *, upgrade: bool) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("Install the committed requirements-ci.txt to obtain the lock generator.")
    version = subprocess.check_output([uv, "--version"], text=True).split()[1]
    if version != UV_VERSION:
        raise SystemExit(f"Lock generation requires uv {UV_VERSION}; found {version}.")
    with tempfile.TemporaryDirectory(prefix="wrangle-lock-") as temporary:
        source = Path(temporary) / "requirements.in"
        source.write_text("\n".join(requirements) + "\n")
        command = [
            uv, "pip", "compile", str(source), "--generate-hashes", "--no-annotate",
            "--no-header", "--python-version", "3.13" if name == "requirements-fuzz.txt" else "3.10",
            "--default-index", "https://pypi.org/simple",
            "--no-config", "--output-file", str(ROOT / name),
        ]
        if name == "requirements-fuzz.txt":
            command.extend(["--python-platform", "x86_64-unknown-linux-gnu"])
        else:
            command.append("--universal")
        if name == "requirements-ci.txt":
            command.extend(["--constraint", str(ROOT / "requirements-ci.in")])
        if upgrade:
            command.append("--upgrade")
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    path = ROOT / name
    path.write_text(
        f"# Intent SHA256: {intent_digest(name, requirements)}\n"
        "# Generated by python scripts/lock_dependencies.py; do not edit artifact hashes.\n"
        "# Install with python -m pip install --require-hashes --only-binary=:all: -r FILE\n"
        + path.read_text()
    )
    check_lock(name, requirements)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check intent freshness and exact hash pins")
    parser.add_argument("--upgrade", action="store_true", help="Resolve newer compatible transitive packages")
    args = parser.parse_args()
    for name, requirements in definitions().items():
        if args.check:
            check_lock(name, requirements)
        else:
            generate(name, requirements, upgrade=args.upgrade)


if __name__ == "__main__":
    main()
