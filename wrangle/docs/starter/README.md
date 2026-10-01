# Your first preparation

These are example files, not a study protocol recommended for your research.
`samples.csv` contains three measurements. `metadata.csv` contains their study
groups. `recipe.yaml` explains the supplied identity, matching, quality and unit
conversion decisions in comments.

In this directory, run:

```sh
wrangle inspect samples.csv
wrangle prepare recipe.yaml --source measurements=samples.csv --source metadata=metadata.csv --output prepared
wrangle inspect prepared
```

Expect two samples, in the original measurement order:

| sample_id | mass_g | qc | group |
| --- | ---: | --- | --- |
| 001 | 1.0 | pass | control |
| 003 | 3.0 | pass | treated |

Sample `002` is excluded because its QC result is `fail`. Read
`prepared/report.txt` for the changes and checks; the full evidence is in
`prepared/receipt.json`. `prepared/recipe.yaml` is the reusable protocol and
`prepared/data.parquet` is the checked analysis table.

Choose a new output directory for each run, for example `--output prepared-002`.
Keep the original files. Adapt the recipe only after deciding what your study
requires; missing metadata and duplicate sample identifiers should be investigated,
not bypassed. Use `wrangle catalog OPERATION` to read an operation's arguments.
Agents can add `--json` to any command to receive structured results and errors.
