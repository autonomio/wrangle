# Migration from Wrangle 0.7

Wrangle 1.0 prepares research data entirely through native Polars. The primary
interface is `inspect`/`prepare`; historical operation names remain accessible
for explicit preparation tasks. The catalog is generated from their actual
implementations. This major version changes input/output types and fixes
scientifically consequential behavior; it is not a drop-in pandas replacement.

## Data and outputs

| Previously | Now |
| --- | --- |
| pandas DataFrame/index | Polars DataFrame or LazyFrame; identity is a named column. |
| NumPy feature arrays | Polars tables; Python row lists are also accepted at the boundary. |
| NumPy column arrays | Polars Series. |
| Tensor reshaping | Polars nested-list Series; no neural models are created. |
| Mutable inputs / `destructive=True` | Immutable inputs; `destructive=True` raises `IMMUTABLE_INPUT`. |
| Implicit, global randomness | Explicit seed; legacy sampling defaults to seed=0, recorded by recipes. |
| Python aggregation/parallel callbacks | Native named reducers or pl.Expr; Python UDFs inside expressions/plans are rejected. |
| Printed reports | Readable CLI summaries, `Prepared.summary()` and saved `report.txt`; native Polars tables and dictionaries remain available. |

Convert data at your external application boundary if necessary; Wrangle itself
has no pandas, NumPy, SciPy, statsmodels, scikit-learn, Keras, or TensorFlow dependency.
Lazy preparation operations retain a LazyFrame when supplied one. Reports and
category discovery materialize where necessary; `prepare` deliberately
snapshots input data and verifies each step. Memory execution collects full
tables; disk execution stores checkpoints and exposes a lazy result.

## Recipe files and command-line output

Recipe authoring is YAML only. Rename and convert old recipe files to `.yaml` or
`.yml`; JSON recipe files are rejected. Keep the same scientific decisions rather
than changing the protocol during migration. Python mappings remain supported for
programmatic callers. Receipts and the generated operation catalog remain JSON.

New saved results contain `data.parquet`, `recipe.yaml`, `report.txt` and
`receipt.json`. Existing bundles containing `recipe.json` remain verified,
read-only sources; preparing a new result publishes YAML. Recipe identity is
computed from the normalized protocol, so comments and formatting do not alter
its hash. Retain your original commented recipe for its explanatory text.

CLI commands now print readable output and recovery guidance by default. Existing
automation must explicitly request `--json` to keep structured results/errors:

```sh
wrangle inspect samples.csv --json
wrangle catalog join --json
wrangle prepare recipe.yaml --source data=samples.csv --output batch --json
```

Use `wrangle example my-study` and [Getting started](getting_started.md) for a
complete file-based workflow. Both researchers and agents run the same recipe,
validation and evidence engine.

## Corrected behavior that can change results

- `array_to_kfold` retains every observation, including remainder rows; fold
  counts and feature/label alignment are validated.
- Splits/shuffles are seeded, nonmutating, and preserve feature/label alignment.
- `df_impute_nan` honors the requested method and handles both null and NaN.
  All-missing columns fail. Modes are deterministic; tied modes choose the
  smallest observed value. `mean_by_std` is seeded continuous uniform sampling
  over observed mean ± sample standard deviation, not the old integer draws.
  This optional method is preserved, not recommended as a scientific default.
- Missingness thresholds mean the maximum allowed missing fraction. Dropping
  nan columns at threshold=0 actually drops every column containing missing data.
- Outlier z-score thresholds are honored; IQR uses Q1−threshold×IQR and
  Q3+threshold×IQR. Missing observations follow the operation's documented policy.
- The legacy `col_groupby_pdf` name returns an empirical probability mass
  function; `col_groupby_cdf` returns the ascending cumulative distribution.
  They no longer compute the same cumulative complement. Binary outcome rates
  use observed outcomes, expose counts, and identify inadequate sample size.
- Renaming rejects collisions. Retained columns in scaling/renaming are actually
  preserved. Empty-string filling leaves other column types unchanged.
- `df_add_scorecol` uses the requested metric rather than a hardcoded F1 field.
- Category encodings have deterministic order and optional returned mappings.
  Use `encode` with a persisted explicit mapping across research batches;
  independently inferred labels cannot establish shared scientific meaning.
- `col_corr_ols` is explicitly dummy-only OLS: category coefficients are group
  means. Equal coefficients share a rank. Fit on training observations only;
  `return_mapping=True` exposes category/coefficient/code. This is preprocessing,
  not an automatic selection of a statistical model.
- Pearson/Spearman use pairwise observed values; undefined correlations are null.
  Native exact Kendall tau-b is capped at 10 million valid pairs per coefficient
  before quadratic allocation. Larger exact analysis belongs outside preparation.
- `df_merge` replaces implicit index joins with equal-length positional joining
  or explicit columns; unmatched keys and cardinality are checked. Recipe
  `join` defaults to left/m:1/all-matched, with stable order.
- `df_find_nan` returns `column`, `missing`, `missing_fraction`, `no_nans`, and
  `quality`; `df_count_uniques` returns a one-row table.
- Datetime sequences use declared calendar intervals and start dates;
  datetime generation fails on a mismatch instead of returning an unrelated
  length after printing a warning. Interval resampling's `quarter` is six hours;
  `month` is explicitly thirty days. Calendar sequences use calendar months.
- `read_large_csv` returns exactly the observed requested rows, never fabricated
  zero padding. Default dtype preserves CSV text, including leading-zero IDs.
- Task classification requires an explicit task when values cannot establish
  meaning. Weighted sampling requires explicit weights; distribution-name
  shortcuts no longer invent a sampling scheme.
- Category-label utilities require declared category mappings rather than
  dropping high-cardinality strings or guessing their interpretation.

## Retired modeling helpers

`df_corr_randomforest`, `df_corr_extratrees`, and the four
`create_synth_*_model` factories raise `MODEL_ENGINE_REQUIRED` with migration
instructions. They neither fit models nor substitute another statistic.
Prepare the research table, then use its Polars output in the analysis environment
you selected. Synthetic data generation remains a native, seeded fixture utility.
Network probes remain explicit utilities; importing or preparing data never
initiates a connectivity check or upload.

## Locating any operation

Read `AGENTS.md` and `docs/operations.json` in the installed package directory.
Each catalog entry names the source module, callable, exact argument defaults,
and operation-specific behavior. Some outputs intentionally differ from
legacy formats; follow the docstring rather than assuming pandas indexing or
old NumPy shapes. The complete batch example and contract tests demonstrate
supported research workflows without downloading external test data.
