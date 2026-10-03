# Security assurance case

## Claim and scope

Given a trusted interpreter/dependency environment, authorized local files and
explicit scientific decisions, Wrangle rejects unsupported execution inputs,
checks the declared data contracts, and publishes a consistent result only after
success. The argument covers the native Polars 1.x source and its documented
CLI/Python interface; it does not certify the scientific truth of a protocol.

## Assets, threats and trust boundaries

Assets are original study files, observation identity, measurement meaning,
prepared tables, provenance evidence and released software. Potentially hostile
inputs include YAML/mappings, source contents and filenames, column descriptions,
previously saved bundles and proposed dependencies/source changes.

| Boundary / threat | Required control | Concrete evidence |
| --- | --- | --- |
| Recipe text to executable operation; injection or ambiguous intent | Strict scalars, nesting bound, registered operations, no executable tags/aliases/callbacks | `_protocol.py`, `_core.py:require_native`, `_catalog.py:resolve`; `tests/test_yaml_protocol.py`, `tests/test_recipe_api.py` |
| Scientific answers to protocol; missing decisions or contradictory intent | Pending decision gate and recorded-choice consistency | `_start.py:validate_decisions`, `_api.py`; `tests/test_start.py`, `tests/test_decision_gate.py`, `tests/test_start_cli.py` |
| Source files to captured data; mutation, wrong format or lost identifiers | Explicit parser/type/sheet options, text CSV identifiers, snapshot and stability checks | `_sources.py`; `tests/test_source_contracts.py`, `tests/test_disk_sources.py` |
| Native operation to observations; unnoticed loss/expansion, imprecise numbers, ambiguous matches | Keys and transitions, reason/expansion declarations, typed/domain checks, native join cardinality/order | `_execution.py`, `_contracts.py`, `_recipe_*.py`; `tests/test_observation_contracts.py`, `tests/test_preparation_regressions.py`, `tests/test_research_checks.py` |
| Prepared bundle to reused source; corrupted or inconsistent evidence | Canonical recipe and logical/physical hashes; verify report and evidence before reuse | `_sources.py:_parent`, `_storage.py`; `tests/test_yaml_bundles.py`, `tests/test_disk_sources.py` |
| Checked result to filesystem; partial result, overwrite or competing writer | Exclusive destination lock and native atomic no-replace publication; cleanup on failure | `_publication.py`, `_storage.py`; `tests/test_publication.py`, `tests/test_human_cli.py`, `tests/test_start.py` |
| Source text to HTML/terminal; markup or terminal-control injection | HTML escaping and explicit terminal control escaping | `_presentation.py`; `tests/test_human_cli.py`, `tests/test_start_cli.py` |
| Source/dependencies to release; vulnerable dependency or substituted artifact | Pinned workflow actions, limited token rights, dependency audit, static analysis, tests, artifact attestation and identity verification | `.github/workflows/`, `.github/dependabot.yml`, `pyproject.toml`, `scripts/verify_distribution.py`; [release verification](releases.md) |

Paths without `tests/` or `.github/` are package-relative; tests and workflows are
checkout-relative. A test demonstrates the behavior it exercises, not the absence
of every vulnerability. Consult the actual scan/coverage/build results for the
specific commit; a workflow definition alone is not proof it has run successfully.

## Design principles and common weaknesses

Allowlisted declarative operations reduce attack surface and avoid converting
research data into code. Fail-closed parsing and dataset checks are complete
mediation points for the supported interfaces. Ordinary Python objects are copied
into a finite primitive grammar rather than trusted as arbitrary executable values.
Least-privilege workflow permissions and private vulnerability reporting limit
release and disclosure exposure. Native hashing and atomic publication provide
independent checks around execution and filesystem boundaries.

Executable deserialization and command injection are countered by the grammar
and registered dispatch, rather than sanitizing arbitrary programs. HTML/terminal
injection is countered at presentation. Snapshot checks address input races;
exclusive publication addresses overwrite races. Typed native operations and
explicit scientific checks counter truncation, nonfinite values, invalid identity,
cardinality and unintended observation loss. Dependency alerts and static tools
counter known component and Python implementation weaknesses.

No authentication credentials or private cryptographic keys are processed by the
preparation API. Snapshot hashes use Python's standard SHA-256 implementation;
scientific sampling seeds are reproducibility choices, not security randomness.
Release signing is delegated to established OIDC/Sigstore infrastructure.

## Residual risks and assurance maintenance

A local adversary with permission to alter source, environment and all evidence
can construct a new self-consistent result. Native decoder defects, compromised
upstream packages and denial of service remain possible. Receipts are not digital
signatures of researcher data. Resource isolation, participant-data protection,
scientific authorization remains a researcher decision; the required code review
is performed by bit-mis under the project owner's automated review policy.

Update this argument when an interface, parser, source format, trust boundary or
release mechanism changes. Include relevant adversarial regressions, scan findings
and real build/coverage evidence. Track open findings through issues or private
advisories; never replace an unresolved human or release claim with an assumption
in the [OpenSSF evidence record](openssf-evidence.yaml).
