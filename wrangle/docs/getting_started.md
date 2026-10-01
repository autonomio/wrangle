# Prepare your first research dataset

Wrangle turns source files into a checked table using a YAML protocol you keep
and reuse. A terminal guides you through the decisions; Python and agents use
the same protocol and preparation checks.

## Start with your own files

After installing Wrangle, run this in a terminal, replacing the filenames:

```sh
wrangle start measurements.csv --metadata metadata.csv --output my-protocol
```

Leave out `--metadata metadata.csv` if you have one file. `my-protocol` must be
a new directory. For Excel, install `wrangle[excel]` and add `--sheet Measurements`.
If `--metadata` also supplies an Excel file, add `--metadata-sheet Samples`.
Wrangle does not select a worksheet.

The guide shows observed columns and asks:

- What does one row represent, and which columns identify it without repeats?
- Which fields contain measurements, how should numbers be represented, and
  what units were supplied? Decimal numbers use approximate `Float64`; exact
  whole numbers use `Int64`. Native numeric files can keep their existing type.
- Which exact instrument codes mean missing, and are missing values allowed?
- If metadata is supplied, which columns match, and must every observation match?
- Should all rows be retained, or only values passing your approved QC rule?

Choose from the displayed column numbers or names. Use `1` for a dimensionless
measurement. Press Enter when a scientific decision is uncertain. Wrangle does
not fill in an answer from the values it observes.

The new directory contains:

| File | What to do |
| --- | --- |
| `README.md` | Read the questions and copy the exact next command. |
| `answers.yaml` | Keep your answers; `null` marks an unanswered question. |
| `recipe.yaml` | Review the commented protocol compiled from those answers. |

An unanswered required decision blocks preparation with `UNRESOLVED_PROTOCOL`.
Complete `answers.yaml` and use the README command to save a resolved protocol
in a new directory. Do not remove the pending questions to make it run.
When the guide says “ready to prepare”, the choices are complete; the data
have not been validated. Copy the README's `wrangle prepare ...` command to run
the transformations and checks. A successful result includes `report.txt`,
`data.parquet`, `recipe.yaml` and `receipt.json`.

The initial guide handles one observation file, optional metadata, numeric
representation, exact missing codes and one QC keep-values rule. It preserves
observation order and permits at most one metadata match per observation.
It labels supplied units; it does not convert them. Variable descriptions,
scientific ranges, repeated measurements, reshaping, imputation and statistical
methods require further explicit protocol decisions in the [recipe manual](recipes.md).
The guide currently loads files for inspection and can require substantial memory;
a resolved protocol can use disk preparation for supported large-file workflows.

Keep the resolved `recipe.yaml` for the next batch. Bind new file paths to the
same source names and use a new output directory. The same identity, schema,
matching and missingness checks run each time. If a check fails, investigate the
reported data or protocol decision; do not weaken the check merely to finish.

## Agents and noninteractive use

```sh
wrangle start measurements.csv --metadata metadata.csv --output draft --json
# Edit draft/answers.yaml with researcher-confirmed answers before continuing.
wrangle start measurements.csv --metadata metadata.csv --answers draft/answers.yaml --output resolved --json
```

`--json` returns observations, exact questions, permitted choices, schemas and
pending decisions without prompting. Redirected input also never prompts unless
`--interactive` is explicitly requested. An agent asks the researcher for the
same missing answers, writes strict YAML and reruns. `prepare` never prompts.
See [guided answers](recipes.md#guided-start) for the complete answer grammar and
[the verified workflow](start_workflow.py) for expected results and recovery.
That workflow uses synthetic study files; researcher usability trials are separate.

## Try the supplied study

After installing Wrangle, open a terminal and run:

```sh
wrangle example my-study
cd my-study
wrangle inspect samples.csv
```

This creates a new folder containing `samples.csv`, `metadata.csv`, `recipe.yaml`
and a short README. `inspect` shows three samples, column names, missing values
and example rows. It does not change the files or decide what they mean.
Identifiers such as `001` remain text, including their leading zeros.

## Read the example decisions before running

Open `recipe.yaml` in a text editor. Lines beginning with `#` explain the example.
The instructions below them say:

1. A row represents one sample, identified by `sample_id`.
2. Convert `mass_mg` to numbers while keeping sample identifiers as text.
3. Match each sample to one metadata record, preserving the sample-file order.
4. Keep samples only when their instrument QC result is `pass`.
5. Convert milligrams to grams and call the output measurement `mass_g`.
6. Check that required fields are present, masses are nonnegative, and units and
   variable descriptions have been recorded.

The quality rule and limits were supplied for this example. Wrangle does not
recommend them for your study or infer them from the values it observes.

## Prepare and read the result

```sh
wrangle prepare recipe.yaml --source measurements=samples.csv --source metadata=metadata.csv --output prepared
wrangle inspect prepared
```

`measurements` and `metadata` connect the names in the recipe to your files.
The terminal reports the changes and whether the declared checks passed.
The analysis table has two rows:

| sample_id | mass_g | qc | group |
| --- | ---: | --- | --- |
| 001 | 1.0 | pass | control |
| 003 | 3.0 | pass | treated |

Sample `002` was excluded under the declared QC rule; the source still has all
three samples. Open `prepared/report.txt` for the saved explanation.

| Saved file | What it is for |
| --- | --- |
| `report.txt` | Read the changes, checks and unresolved declarations. |
| `data.parquet` | Use the checked table in your analysis environment. |
| `recipe.yaml` | Review and reuse the preparation protocol. |
| `receipt.json` | Verify exact sources, output, exclusions, parameters and hashes. |

JSON is used for recorded evidence, not recipe authoring. Disk execution can
add an `evidence/` directory containing complete exclusion and identity records.
Your analysis environment must support Parquet; Wrangle's Python result is also
a native Polars table.

## Extend the protocol

Keep a copy of your original files. Use `inspect` to find exact column names,
then edit the YAML file to state your protocol. Before running, resolve:

| Decision | Example question |
| --- | --- |
| Observation identity | Is one row a specimen, a visit, or one measurement within a visit? |
| Missingness | Does an empty value mean missing, or does your instrument use a code such as `-999`? |
| Metadata matching | Must every sample match, and can one sample have several metadata records? |
| Quality exclusions | Which predeclared rule permits dropping an observation, and why? |
| Measurement units | What are the source units and the exact approved conversion? |
| Analysis requirements | Which values must be present, and which ranges or categories are allowed? |

Use spaces for indentation, not tabs. Keep identifiers and literal categories
quoted if they could look like numbers or special values: `"001"`, `"null"`.
Use `true` and `false` for deliberate yes/no settings. The keys in the example,
including `on`, retain their ordinary text meanings. A recipe is one YAML mapping;
use explicit values rather than YAML tags, anchors, aliases or merge shortcuts.
JSON files cannot be supplied as recipes. Comments explain the protocol but do
not alter execution or its identity.

Read an operation with `wrangle catalog OPERATION`, for example
`wrangle catalog join`. The [recipe manual](recipes.md) maps common tasks to
operations and gives the complete grammar. An agent may draft a recipe, but it
must ask you for unresolved scientific decisions; it must not guess them.

## Reuse it for the next batch

Keep `recipe.yaml` and bind the next files to the same source names. Always use
a new output directory:

```sh
wrangle prepare recipe.yaml --source measurements=next-samples.csv --source metadata=next-metadata.csv --output prepared-002
```

If a check fails, no result directory is published. Read the explanation and
stable error code, investigate the affected data or unresolved decision, and
rerun. Do not relax a check simply to make the preparation finish. If the study
protocol truly changes, record that decision in a new recipe.

When fitting imputation or standardization on a reference batch, later batches
must reuse its recorded parameters instead of fitting new values. See the
[frozen-parameter workflow](frozen_parameters.py).

## Python and agents

Researchers who use notebooks can run the same YAML protocol:

```python
import wrangle as wr

print(wr.inspect("samples.csv").summary())
result = wr.prepare(
    {"measurements": "samples.csv", "metadata": "metadata.csv"},
    "recipe.yaml",
)
print(result.summary())
result.data
```

For structured command output, agents add `--json`:

```sh
wrangle inspect samples.csv --json
wrangle catalog join --json
wrangle prepare recipe.yaml --source measurements=samples.csv --source metadata=metadata.csv --json
```

Both presentations execute identical checks and return the same evidence.
The [agent entrypoint](../AGENTS.md) gives exact navigation and decision boundaries.
The [verified workflow](human_workflow.py) checks the example through Python,
readable CLI output and structured CLI output on every release.

For large local files, add `--execution disk --output new-directory`; complete
inputs and intermediates stay on disk, although global operations can still need
substantial RAM. Inspection remains eager. Read [disk execution](recipes.md#disk-execution)
before preparing a dataset that does not fit memory.
