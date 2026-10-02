<div align="center">
  <br />
  <a href="https://github.com/autonomio"><img src="https://avatars.githubusercontent.com/u/28189776?v=4" alt="Autonomio" width="150" height="150" /></a>
  <br />
</div>
<br />
<div align="center"><b>Wrangle turns research files into checked analysis tables and reusable preparation evidence.</b></div>

<div align="center">
  <a href="#wrangle">Wrangle</a> •
  <a href="#what-wrangle-is-not">Scope</a> •
  <a href="#capabilities">Capabilities</a> •
  <a href="#first-successful-preparation">First Preparation</a> •
  <a href="#choose-the-next-task">Learn More</a>
</div>
<br />
<div align="center">
  <a href="https://www.bestpractices.dev/projects/15135"><img src="https://www.bestpractices.dev/projects/15135/badge" alt="OpenSSF Best Practices" /></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/autonomio/wrangle"><img src="https://api.scorecard.dev/projects/github.com/autonomio/wrangle/badge" alt="OpenSSF Scorecard" /></a>
  <a href="https://pypi.org/project/wrangle/"><img src="https://img.shields.io/pypi/v/wrangle?label=pypi" alt="PyPI version" /></a>
  <a href="wrangle/docs/getting_started.md"><img src="https://img.shields.io/badge/docs-manual-blue" alt="Wrangle documentation" /></a>
  <a href="https://github.com/autonomio/wrangle/actions/workflows/ci-push.yml"><img src="https://github.com/autonomio/wrangle/actions/workflows/ci-push.yml/badge.svg?branch=master&amp;event=push" alt="Master tests and builds" /></a>
</div>

<hr />

<a id="wrangle"></a>

# Wrangle — Scientific data preparation

*Research files prepared through a readable YAML protocol, with checked tables and execution receipts.*

Wrangle brings inspection, cleaning, matching, reshaping, and validation into
one preparation protocol. Declare the research decisions, prepare the table,
and keep the evidence needed to review the result and reuse the protocol.
Researchers, notebooks, and agents use the same native Polars engine.

## What Wrangle Is Not

Wrangle owns data preparation, protocol checks, and retained evidence.
Researchers own observation identity, units, missingness, exclusions, metadata
matches, and statistical choices. Wrangle does not infer those decisions or
certify their scientific suitability. It does not build or fit models.
All transformations and data checks compile to Polars expressions and LazyFrame
plans; Python handles orchestration, files, and receipts.

## Capabilities

| Research task | Supported capability |
| --- | --- |
| Understand incoming files | Inspect columns, missingness, example rows, summaries, and schema changes |
| Prepare instrument measurements | Preserve identifiers; explicitly parse numbers, dates, missing codes, and text |
| Match samples and metadata | Joins with enforced cardinality, ordering, and unmatched-observation policies |
| Combine batches or reshape observations | Concatenation, pivoting, unpivoting, nested fields, and declared output identity |
| Apply a research protocol | Category mappings, derived fields, unit conversions, and recorded exclusions |
| Prepare longitudinal or grouped data | Group summaries, temporal matching, lag, and rolling calculations |
| Reuse reference-batch decisions | Frozen imputation and standardization parameters; declared sampling and partitions |
| Check and retain an analysis table | Output contracts, readable reports, reusable YAML, and hash receipts |

Use the [recipe manual](wrangle/docs/recipes.md#select-the-operation) to choose
a workflow and the [operation catalog](wrangle/docs/operations.json) for exact
arguments, preconditions, and stable failures.

## First successful preparation

This README describes the **1.0 API in this repository**. Historical 0.7 releases
on PyPI use a different interface. Start from the current source checkout:

```sh
git clone https://github.com/autonomio/wrangle.git
cd wrangle
python -m pip install .
```

Use Python 3.10 or later. Installation supplies Polars and the strict YAML parser.
For Excel, install `python -m pip install '.[excel]'` from the checkout and select
a worksheet explicitly. Choose new directories for examples and outputs;
existing destinations are rejected.

Run this complete example in a terminal:

```sh
wrangle example my-study
cd my-study
wrangle inspect samples.csv
wrangle prepare recipe.yaml --source measurements=samples.csv --source metadata=metadata.csv --output prepared
wrangle inspect prepared
```

Expected result: three input samples become two accepted samples, with metadata
attached and mass converted from milligrams to grams:

| sample_id | mass_g | qc | group |
| --- | ---: | --- | --- |
| 001 | 1.0 | pass | control |
| 003 | 3.0 | pass | treated |

Sample `002` is excluded by the example's declared instrument quality rule.
Open `prepared/report.txt` to review the checks and exclusion. The commented
`recipe.yaml` explains the supplied decisions; its quality rule, units, and limits
belong to this example and must be adapted to your study.

### Use the same recipe in Python

From the `my-study` directory:

```python
import wrangle as wr

profile = wr.inspect("samples.csv")
print(profile.summary())
result = wr.prepare(
    {"measurements": "samples.csv", "metadata": "metadata.csv"},
    "recipe.yaml",
    output="prepared-python",
)
print(result.summary())
result.data  # A Polars table with the two accepted samples.
```

`inspect` observes without changing the source. CSV fields start as text,
preserving identifiers such as `001`; the recipe explicitly converts measurements.
`prepare` returns checked `data` and a `receipt`. A one-source recipe can use
`wr.prepare("samples.csv", "recipe.yaml")`; Python callers may also supply a mapping.

### Start with your own files

```sh
wrangle start measurements.csv --metadata metadata.csv --output my-protocol
```

Omit `--metadata metadata.csv` for one file. The guide asks what a row represents,
which fields identify it, how numbers and units should be represented, and what
missing values, exclusions, and metadata matches mean. Enter column numbers or
names; press Enter if you are unsure. Your answers become a commented YAML protocol.

Unanswered decisions block preparation. A complete draft is ready to run checks;
preparation validates the data. Run the command printed by the guide and review
the resulting report. Reuse the same recipe for the next batch.
For Excel, supply `--sheet "Sheet name"` and, when needed,
`--metadata-sheet "Sheet name"`. See [Getting started](wrangle/docs/getting_started.md).

## Retain the evidence

| File | Retained evidence |
| --- | --- |
| `data.parquet` | Checked analysis table |
| `recipe.yaml` | Resolved, reusable preparation protocol |
| `report.txt` | Readable summary of checks, changes, exclusions, and unresolved declarations |
| `receipt.json` | Input and output hashes, protocol identity, checks, exclusions, and fitted parameters |

Python's `output="new-directory"` and the CLI's `--output new-directory` save
these files together. Use `result.write("new-directory")` to save an unsaved
Python result. Sources remain unchanged; existing outputs are never overwritten.
A saved preparation directory can be reused as a source after evidence verification.

The same protocol serves shell scripts and agents:

```sh
wrangle inspect samples.csv --json
wrangle catalog join --json
wrangle prepare recipe.yaml --source measurements=samples.csv --source metadata=metadata.csv --json
```

The CLI is readable by default; `--json` supplies structured results and errors.
Agents use `wrangle start ... --json` for the same questions without prompts and
`--answers answers.yaml` for researcher-approved decisions. Preparation uses
identical checks and evidence in Python and both CLI presentations.
Recipe files are YAML only (`.yaml` or `.yml`); internal receipts and catalogs use
JSON. Comments and formatting do not change protocol identity.

## Risk Boundary

Passing checks establishes conformity to your declared protocol. It does not
establish the scientific correctness of a chosen rule. Hash receipts check
consistency with retained evidence; they do not establish authorship or constitute
an independent audit. Unresolved guided decisions stop execution with
`UNRESOLVED_PROTOCOL`; answer them rather than removing the blockers.

Preparation accepts local CSV, TSV, Parquet, Arrow IPC, NDJSON, explicitly selected
Excel sheets, native Polars tables, and verified preparation bundles. It never
uploads data or accesses the network. Recipes cannot execute arbitrary callbacks.
Read the [security boundaries](wrangle/docs/security.md) and
[recipe contracts and recovery](wrangle/docs/recipes.md).

For large local files, use `--execution disk --output new-directory`, or
`wr.prepare(sources, "recipe.yaml", execution="disk", output="new-directory")`.
Complete input snapshots, intermediates, and evidence stay on disk; `result.data`
is a Polars LazyFrame. Global joins, sorting, and grouped calculations can still
need substantial RAM. Inspection and guided start remain eager.
See [disk execution](wrangle/docs/recipes.md#disk-execution).

## Choose the next task

| Job | Start here |
| --- | --- |
| Prepare your own files with guided decisions | [Getting started](wrangle/docs/getting_started.md) |
| Try the complete human and agent example | [Verified first study](wrangle/docs/human_workflow.py) |
| Keep and reuse a research batch | [Research batch](wrangle/docs/research_batch.py) |
| Join, reshape, or summarize observations | [Preparation workflows](wrangle/docs/preparation_workflows.py) |
| Reuse fitted reference parameters | [Frozen parameters](wrangle/docs/frozen_parameters.py) |
| Prepare large files on disk | [Disk workflow](wrangle/docs/disk_preparation.py) |
| Select an operation and its preconditions | [Operation catalog](wrangle/docs/operations.json) |
| Integrate Python or an agent | [Agent entrypoint](AGENTS.md) and [recipe manual](wrangle/docs/recipes.md) |
| Run commands and recover from errors | [Command line](wrangle/docs/recipes.md#inspection-and-cli) |
| Replace a historical method | [Migration](wrangle/docs/migration.md) |

Installed agents start at `Path(wrangle.__file__).parent / "AGENTS.md"`;
the installed manual and generated operation contracts live beside it under `docs/`.
The catalog is generated from the engine, and shipped examples are executed in
release checks. Stable `WrangleError` codes and details guide recovery without
substituting a scientific method. Dedicated Wrangle agent tools are unnecessary.

<a id="contribute-support-and-cite"></a>

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks, and review.
Propose work through [Wrangle issues](https://github.com/autonomio/wrangle/issues).
[Governance](GOVERNANCE.md), [conduct](CODE_OF_CONDUCT.md), the [roadmap](ROADMAP.md),
and [release notes](CHANGELOG.md) describe project maintenance;
[architecture](wrangle/docs/architecture.md) describes the engine.

## Support

Use [Wrangle issues](https://github.com/autonomio/wrangle/issues) for bug reports,
feature requests, and usage questions. Include a minimal YAML protocol, Wrangle
and Python versions, operating system, and the error code and details.
Use a small shareable fixture when a report needs source data.

## Vulnerabilities

Report suspected vulnerabilities privately through
[GitHub Security Advisories](https://github.com/autonomio/wrangle/security/advisories/new).
Use [SECURITY.md](SECURITY.md) for supported versions and response policy, and
[release verification](wrangle/docs/security/releases.md) to check artifact identity.
[OpenSSF evidence](wrangle/docs/security/openssf-evidence.yaml) records the published
Silver answers, distinguishing measured checks from owner attestations.

## Citations

Identify Wrangle, its version, and the retained preparation protocol and receipt
when citing a research workflow. Link the corresponding
[release](https://github.com/autonomio/wrangle/releases) to identify the software used.
No DOI is supplied.

## License

[MIT License](LICENSE).
