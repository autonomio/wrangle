# Wrangle

Turn research files into a checked analysis table. Keep the preparation protocol
in a readable YAML file, reuse it for the next batch, and see what changed.
Wrangle serves researchers directly and agents working on their behalf.

## Install

From this checkout, using Python 3.10 or later:

```sh
python -m pip install .
```

Wrangle uses Polars for data preparation and a strict YAML parser for recipe files.
Excel support is an optional extra. Wrangle does not build or fit models.

## Start with your own files

Run this in a terminal:

```sh
wrangle start measurements.csv --metadata metadata.csv --output my-protocol
```

The guide asks what a row represents, which fields identify it, how measurement
numbers and units should be represented, and what missing values, exclusions and
metadata matches mean. Enter column numbers or names; press Enter if you are not
sure. Wrangle records your answers and creates a commented `recipe.yaml`.

Unanswered decisions block preparation. A complete draft is ready to run checks;
preparation validates the data. The guide prints the exact preparation
command; run it and review the resulting report. The same recipe can then prepare
the next batch. See [Getting started](wrangle/docs/getting_started.md).

For Excel, install `wrangle[excel]` and supply `--sheet "Sheet name"`; the metadata
file uses `--metadata-sheet "Sheet name"`. Worksheet selection is explicit.
Agents use `wrangle start ... --json` to receive the same questions without prompts
and provide researcher-approved decisions through `--answers answers.yaml`.

## Try a complete example

Run these commands in a terminal:

```sh
wrangle example my-study
cd my-study
wrangle inspect samples.csv
wrangle prepare recipe.yaml --source measurements=samples.csv --source metadata=metadata.csv --output prepared
wrangle inspect prepared
```

The example joins sample metadata, applies a declared instrument quality rule,
and converts milligrams to grams. It starts with three samples and produces:

| sample_id | mass_g | qc | group |
| --- | ---: | --- | --- |
| 001 | 1.0 | pass | control |
| 003 | 3.0 | pass | treated |

Sample `002` is excluded because its instrument quality result is `fail`.
The terminal explains the changes. Open `prepared/report.txt` for the saved
summary, including checks and any unresolved declarations.

The example's `recipe.yaml` includes comments explaining each decision.
Use [Getting started](wrangle/docs/getting_started.md) to adapt it to your files;
its quality rule, units and limits are examples, not recommendations for your study.

## Use the same recipe in Python

From the example directory:

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

For a recipe using only one input file, `wr.prepare("samples.csv", "recipe.yaml")`
is sufficient; that recipe need not declare an `input` source name.
Programmatic Python callers may also supply a mapping instead of a YAML file.

## Your files and decisions

`inspect` reports columns, missing values and example rows without changing the
source. CSV fields start as text, preserving sample identifiers such as `001`.
Your recipe explicitly converts measurement fields to numbers.

`prepare` reads local CSV, TSV, Parquet, Arrow IPC or NDJSON files, native Polars
tables, an explicitly selected Excel sheet, or a verified prepared directory.
Multiple sources are named as in the example. Author recipe files as `.yaml` or
`.yml`; JSON recipe files are not an input format.

You decide what identifies an observation, what missing values mean, which
observations to exclude, how metadata must match, and which units or statistical
methods apply. Wrangle checks those decisions and stops with an explanation when
the data contradict them. A successful check establishes your declared protocol;
it does not establish the scientific correctness of a chosen rule.

## Keep and reuse the result

Passing `output="new-directory"` saves `data.parquet`, `recipe.yaml`, `report.txt`
and `receipt.json` together. Existing directories are never overwritten. The YAML
is the reusable protocol; the report is for reading; the JSON receipt records
inputs, output, checks, exclusions, parameters and hashes for verification.
Sources remain unchanged. Use `result.write("new-directory")` to save an unsaved
Python result, or use a verified preparation directory as a later source.

For large local files, add `--execution disk --output new-directory` to the CLI,
or use `wr.prepare(sources, "recipe.yaml", execution="disk", output="new-directory")`.
Complete inputs and intermediates stay on disk; the result's `data` is a Polars
LazyFrame. Global joins, sorting and grouped calculations can still need substantial
RAM. `inspect` remains eager. See [disk execution](wrangle/docs/recipes.md#disk-execution).

## Researchers and agents share one engine

The CLI prints readable results and recovery guidance by default. Add `--json`
to `start`, `inspect`, `catalog`, `prepare` or `example` for structured output and errors.

```sh
wrangle inspect samples.csv --json
wrangle catalog join --json
wrangle prepare recipe.yaml --source measurements=samples.csv --source metadata=metadata.csv --json
```

Recipes, checks and receipts are identical across Python, human CLI output and
agent CLI output. Internal receipts and the generated operation catalog use JSON;
recipe authoring uses YAML only. Comments and formatting do not change recipe identity.

Agents start at [AGENTS.md](AGENTS.md), also shipped inside the installed package.
The [recipe manual](wrangle/docs/recipes.md) gives exact grammar, boundaries and
recovery; the [operation catalog](wrangle/docs/operations.json) is generated from
the engine. The [human workflow](wrangle/docs/human_workflow.py) verifies the
complete example through both presentations. Dedicated agent tools are unnecessary.
Read [migration](wrangle/docs/migration.md) when upgrading from older Wrangle.

## Development and security

[OpenSSF Best Practices progress](https://www.bestpractices.dev/en/projects/15135)
records the project assessment; badge attainment depends on verified evidence.

[Contributing](CONTRIBUTING.md) covers installation, required checks and review;
[governance](GOVERNANCE.md), [conduct](CODE_OF_CONDUCT.md), the [roadmap](ROADMAP.md)
and [release notes](CHANGELOG.md) explain how the project is maintained.
Report vulnerabilities through [the security policy](SECURITY.md).

The [architecture](wrangle/docs/architecture.md) and
[security boundaries](wrangle/docs/security.md) describe the actual guarantees.
[Release verification](wrangle/docs/security/releases.md) checks artifact identity;
[OpenSSF evidence](wrangle/docs/security/openssf-evidence.yaml) distinguishes
published controls from work still awaiting verification.

Verified examples and the agent manual ship with the package. All transformations
and data checks use native Polars expressions and plans. Python handles files,
orchestration and receipts. Preparation never uploads data or accesses the network.

[MIT License](LICENSE).
