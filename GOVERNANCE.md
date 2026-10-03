# Project governance

Wrangle is maintained by Mikko Kotila (`mikkokotila`). Decisions are made through
public issues and pull requests, with private handling for vulnerabilities.
The project prioritizes reproducible data preparation for scientific researchers;
neural-network construction and generic workflow infrastructure are outside scope.

## Roles and decisions

| Role | Responsibilities | Assignment |
| --- | --- | --- |
| Lead maintainer | Scope, compatibility, issue triage, releases, dependency/security response and access review. | Mikko Kotila (`mikkokotila`). |
| Required reviewer | Review scientific contracts, implementation and validation; accept or reject a change with reasons. | `bit-mis`, designated by the project owner. |
| Backup maintainer | Continuity for issue triage, merges, releases and security response within one week. | `EnergyGuy3`, confirmed by the project owner. |
| Contributor | Submit focused changes, scientific assumptions, documentation and meaningful regression tests; address review findings. | The author of a contribution. |
| Security reporter | Report privately, coordinate reproduction/disclosure and choose whether to receive public credit. | The reporter of an issue. |

A maintainer records significant scope or compatibility decisions in an issue or
pull request. The lead maintainer resolves disagreements after considering the
research use case and documented alternatives. Approval never substitutes for
tests or scientific decisions. The project owner designates `bit-mis` as the sole
required reviewer and code owner. Its automated review satisfies the project's
review requirement; a separate human approval is not required. This is an
automated review policy, not independent human review. GitHub requires approval
after the last push and rejects approval by the pull-request author. Changes to
this governance model use the same public review process.

The maintainer may delegate specific review or release responsibilities to an
existing authorized collaborator. New repository or publishing access requires
a separate access decision; a document does not itself grant permissions.

## Continuity

The project owner confirms `EnergyGuy3` as the backup maintainer who can take
over issues, merges and releases within one week if the lead is unavailable.
GitHub administrator access is present. The owner also identifies EnergyGuy3 as
the project's secure-development expertise contact. These are owner attestations;
repository permissions alone are not evidence of availability or expertise.

The release workflow uses organization publishing credentials when configured;
never share their values. Rehearse an issue/merge/release using test infrastructure
and review publishing access and recovery arrangements at least annually and
whenever a maintainer changes. Use individual accounts with multi-factor
authentication; store recovery material privately. Do not share credentials in
repository files, issues, receipts or an OpenSSF questionnaire.
