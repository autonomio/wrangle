# Contributing

Use [issues](https://github.com/autonomio/wrangle/issues) for reproducible bugs,
research workflows and proposals. Use pull requests against `master` for changes.
Follow the [code of conduct](CODE_OF_CONDUCT.md), [governance](GOVERNANCE.md) and
[security policy](SECURITY.md). Report vulnerabilities privately.

## Development environment

Python 3.10 or later and Git are required. From a checkout:

```sh
python -m venv .venv
# POSIX: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --require-hashes --only-binary=:all: -r requirements-ci.txt -r requirements-audit.txt
python -m pip install --no-deps --no-build-isolation -e '.[dev,excel]'
python -m pip check
python scripts/lock_dependencies.py --check
python -m pytest -W error --cov=wrangle --cov-report=term-missing --cov-report=xml --cov-fail-under=80
python -m ruff check wrangle scripts
python -m bandit -c pyproject.toml -r wrangle scripts/check_release.py -q
python -m pip_audit --require-hashes -r requirements-audit.txt -r requirements-ci.txt
python scripts/build_catalog.py --check
python -m build --no-isolation
python scripts/check_reproducible_build.py
python scripts/verify_distribution.py
```

If an operation or manual definition changes, run `python scripts/build_catalog.py`
and commit its generated documents with the source change. CI executes the
installed-package workflows and checks source/build/distribution consistency.
Use a new result directory when preparing fixtures; never change real study files.

## Acceptance requirements

A change must solve a concrete researcher problem. State the observation unit,
keys, units, missingness, exclusions, ordering/cardinality and randomness it
changes. Do not infer scientific meaning or replace a statistical method to make
a check pass. All data transformations and checks use native Polars expressions
and lazy plans; Python handles intent, validation grammar, files and receipts.

Every major new capability MUST include meaningful automated tests for its
behavior and failure boundaries. Every fixed bug should have a regression test;
at least half of fixes over a six-month period MUST have one. Reversible wording
changes do not require tests that merely restate their implementation. Runtime
statement coverage MUST remain at least 80%; do not exclude production code to
increase the percentage. Treat runtime warnings as errors.

Use PEP 8 for Python, with the deliberate compactness exceptions configured in
`pyproject.toml`. Ruff enforces the configured layout, imports and correctness
rules. Prefer descriptive names, explicit error codes and small operation
contracts. Preserve public exports and scientific behavior. Do not introduce
arbitrary callbacks, executable YAML, hidden schema inference, silent memory
fallbacks, or automatic loss/expansion of observations.

Describe the problem, resulting behavior, scientific choices and actual validation
in the pull request. Include documentation for changed interfaces and recovery.
Keep source, generated catalog, examples and migration notes consistent. Do not
include real participant data, credentials, compiled files or unrelated changes.

Every pull request requires one independent human approval, code-owner review
and approval after the last push. `.github/CODEOWNERS` names the maintainers;
authors cannot approve their own changes. Tests, quality, CodeQL and the bounded
`fuzz` check must pass on an up-to-date branch; administrators have no bypass. A reviewer checks the science
contract, source behavior, tests and compatibility. Resolve findings before merge;
do not suppress a finding merely to pass CI. Agent analysis supplements human review. Security
findings need a documented correction or specific reviewed non-exploitability
justification. Releases use the [verification process](wrangle/docs/security/releases.md).

Contributions use the existing MIT license. A separate CLA or compulsory DCO
sign-off is not currently imposed; do not sign a legal assertion for someone else.

## Dependency and parser maintenance

CI uses complete hash-locked dependencies while the public package retains its
supported runtime ranges. `requirements-ci.in` records the reviewed tool pins;
`pyproject.toml` defines runtime and optional dependencies. After changing either,
run `python scripts/lock_dependencies.py --upgrade` with the locked uv generator
and review the changed versions and artifact hashes. `--check` verifies intent
and exact pins without resolving packages over the network. Latest and supported
minimum Polars/YAML plans are separate; never silently replace the minimum test.

The [fuzzing guide](wrangle/docs/security/fuzzing.md) explains bounded Atheris
execution, corpus retention and recovery. The [Scorecard controls](wrangle/docs/security/scorecard.md)
distinguish enforced safeguards from historical measurements and proposed scores.
