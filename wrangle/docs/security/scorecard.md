# Scorecard controls and limits

The [live OpenSSF Scorecard](https://scorecard.dev/viewer/?uri=github.com/autonomio/wrangle)
is measured separately from the [Best Practices Silver badge](https://www.bestpractices.dev/en/projects/15135).
Check its timestamp and scanned commit. A proposed workflow, local result or
projected score is not evidence that the public score has changed.

The 2026-10-02 07:07 UTC published scan measured **8.7**, with signed releases,
hash-pinned dependencies and fuzzing at 10/10, and branch protection at 8/10.
The scanned commit was `9571e6d61af66a4be86cb9df3916d85d2da433f1`.
The live service is authoritative when this dated snapshot becomes stale.

## Controls

- `master` requires successful tests, quality, CodeQL and fuzzing checks on an
  up-to-date branch, one approval after the last push and dismissal of stale
  approvals. No administrator bypass, force push or branch deletion is allowed.
  The owner designates `bit-mis` as the sole reviewer and code owner. Its automated
  approval satisfies the review policy; no separate human approval is required.
  This policy does not claim independent human review.
- CI and release dependencies are exact versions with SHA-256 hashes. Installs
  reject missing or mismatched hashes and use wheels; the local checkout installs
  without fetching dependencies or isolated build tools. Package consumers retain
  supported version ranges in `pyproject.toml`.
- Bounded Atheris jobs exercise strict YAML and declarative expressions against
  synthetic fixtures. Unexpected exceptions and changed observations fail the run;
  permitted protocol errors remain permitted. Crash inputs are retained for triage.
- Release identity verification precedes provenance export. `provenance.intoto.jsonl`
  contains the unchanged signed DSSE envelope from `attestation.json`, preserving
  the signed payload and signatures. Keep the full Sigstore bundle for trust and
  certificate verification. See [release verification](releases.md).

The [v1.0.0 release](https://github.com/autonomio/wrangle/releases/tag/v1.0.0)'s JSONL asset exports its already verified provenance; it does
not replace artifacts, change the release commit or create a new attestation.
Scorecard v5.5 recognizes `.intoto.jsonl`; its filename detection is not itself
cryptographic verification. Wrangle separately verifies repository, workflow,
commit, tag, trusted issuer and every released subject before publication.

## Historical measurements

Scorecard uses recent default-branch changes for code-review, CI and SAST evidence.
Legacy unreviewed or unchecked changes remain in that history until genuine new
work replaces them in the evaluation window. Never manufacture commits, approvals,
passing checks or unrelated activity to increase a score.

One bit-mis approval is the current review requirement; no second reviewer is
required. Fuzzing adds adversarial coverage, not a proof that a parser
has no defects. PyPI publication follows explicit release authorization and verification; it is
not enabled merely to obtain packaging points.
