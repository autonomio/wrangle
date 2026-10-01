# Security policy

Report suspected vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/autonomio/wrangle/security/advisories/new).
Do not post confidential datasets, credentials or exploit details in public issues.
If that channel is unavailable, contact the package maintainer at
mailme@mikkokotila.com with a brief description and a way to reply; arrange a
private exchange before sending sensitive material.

## Supported versions

The current development branch receives security fixes. The native Polars 1.x
line will receive supported releases after publication. Legacy 0.x users should
follow [the migration guide](../migration.md); the former pandas,
NumPy and neural-network interfaces are not maintained by the new engine.
Release notes identify the affected versions and the upgrade path.

## Response and disclosure

The maintainer owns triage, remediation and disclosure:

1. Aim to acknowledge privately within seven calendar days; investigate within
   fourteen days. These are response targets, not a claim about past reports.
2. Reproduce with minimal synthetic input; identify affected versions, native
   dependencies, trust boundaries and impact. Avoid reproductions using private
   researcher data. Restrict draft advisories to the people needed for a fix.
3. Agree on coordinated disclosure with the reporter. For a confirmed issue,
   develop the fix and a regression test privately where needed.
4. Fix exploitable critical issues promptly; do not publish a release with a
   known exploitable critical issue. Target other confirmed medium/high issues
   within sixty days; document any exception, mitigation and follow-up date.
5. Publish the fix, affected/fixed versions, impact, mitigations and any assigned
   CVE in an advisory and release notes. Credit reporters unless they request
   anonymity. Keep an auditable record of acknowledgement and resolution dates.

Dependency alerts and weekly audits are triaged through the same process.
Security suppression requires a specific finding, evidence that the affected
behavior is unreachable or unexploitable, an owner and a review date.

## Guarantees and limits

Read the shipped [security requirements](../security.md),
[assurance case](../security/assurance.md) and
[release verification instructions](../security/releases.md).
Receipts establish consistency of captured inputs, declared protocol and output;
they do not authenticate a dataset's author or prove scientific correctness.
