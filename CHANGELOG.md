# Release notes

## 1.0.1 — Unreleased

- Preserve long measurement-field names, significant repeated spaces and escaped
  Unicode characters when saving YAML protocols; text retains its exact values.
- Add bounded continuous Atheris fuzzing for strict YAML and native expressions,
  with retained corpora and an ordinary regression for the serialization defect.
- Install complete CI, release, optional Excel and minimum-version dependency
  plans with verified SHA-256 hashes; retain supported ranges for package users.
- Require independent human approval, code-owner review and approval after the
  last push, alongside existing up-to-date test/security checks and no bypass.
- Export unchanged verified release provenance in standard in-toto JSONL form.

## 1.0.0

Wrangle becomes native Polars data preparation for scientific researchers and
agents. This is a breaking change from the legacy pandas/NumPy package; follow
[the migration guide](wrangle/docs/migration.md) before upgrading.

- Observe files with `inspect`; prepare them with a compact YAML protocol and
  explicit scientific decisions. CLI and Python share checks and receipts.
- Use guided `start` for real files: keep partial answers, require observation
  identity, supplied units, missingness, metadata matching and approved QC.
- Enforce join cardinality/order, numeric domains, observation transitions,
  exclusions and complete output contracts using native Polars plans.
- Reuse frozen reference parameters and verified preparation bundles. Disk mode
  retains source/intermediate snapshots and full file-backed evidence.
- Retain checked data, YAML, readable reports and hash receipts atomically in a
  new directory. Failed checks publish no result.
- Ship generated operation contracts, a short agent entrypoint and executable
  human/agent workflows verified from the installed package.
- Keep reports as exact UTF-8/LF bytes, package timezone data and use portable
  evidence paths so prepared bundles can move between operating systems. CLI
  output preserves Unicode scientific labels and paths under legacy Windows locales.
- Add security/contribution policies, measured test coverage, dependency/static
  analysis and a workflow for signed release-artifact provenance.

Model building and recipe callbacks are outside scope. Hash receipts are
consistency evidence, not author signatures or proof of scientific correctness.
Publication and verified assets are listed on the
[GitHub releases page](https://github.com/autonomio/wrangle/releases).
OpenSSF attainment is tracked separately on the
[badge page](https://www.bestpractices.dev/en/projects/15135).

## 0.7.6 — 2024-04-20

Fixed the remaining legacy pandas `.drop()` call.
[Historical release](https://github.com/autonomio/wrangle/releases/tag/v0.7.6).

## 0.7.5 — 2024-04-20

Completed the move to Hatch packaging with small cleanups.
[Historical release](https://github.com/autonomio/wrangle/releases/tag/v0.7.5).

## 0.7.3 — 2024-03-29 (tagged source)

Replaced the deprecated `sklearn` dependency name with `scikit-learn` and updated
legacy tests. [Source comparison](https://github.com/autonomio/wrangle/compare/v0.7.2...v0.7.3).

## 0.7.2 — 2022-04-21

Fixed shuffling along the wrong axis in `array_random_shuffle`.
[Historical release](https://github.com/autonomio/wrangle/releases/tag/v0.7.2).

## 0.7.1 — 2022-04-17

Fixed an issue in `array_random_shuffle`.
[Historical release](https://github.com/autonomio/wrangle/releases/tag/v0.7.1).

## 0.7 — 2022-04-16

Extended array shuffling for multiple inputs, with small fixes and improvements.
[Historical release](https://github.com/autonomio/wrangle/releases/tag/v0.7).

## 0.6.11 — 2022-02-11

Fixed obsolete Keras imports.
[Historical release](https://github.com/autonomio/wrangle/releases/tag/v0.6.11).

## 0.6.10 — 2022-01-28

Changed legacy multilabel synthetic data generation to pass explicit
`n_samples`, `n_features` and `n_classes` arguments; updated CI. These notes are
reconstructed from the tagged source comparison; the original release body was empty.
[Source comparison](https://github.com/autonomio/wrangle/compare/v0.6.9...v0.6.10).

## 0.6.9 — 2021-09-24

Small autumn cleanup, as described in the original release.
[Historical release](https://github.com/autonomio/wrangle/releases/tag/v0.6.9).
