from .._core import operation
from itertools import combinations
import polars as pl
from .._core import frame, finish, WrangleError
from .df_to_groupby import df_to_groupby
from ._expressions import numeric_only


@operation(returns=('table',), recipe='yes', aggregates=True)
def df_groupby_params(data, metric, dims=4, gram_type='ngram', group_by_func='sum', *, seed=0):
    """Aggregate parameter combinations: contiguous ngrams, fixed skipgrams, or sizes 1..dims."""
    schema = numeric_only(data, [metric])
    features = [name for name in schema if name != metric]
    if not features:
        raise WrangleError("INVALID_OPTION", "At least one parameter column is required.")
    if not isinstance(dims, int) or isinstance(dims, bool) or dims < 1:
        raise WrangleError("INVALID_OPTION", "dims must be a positive integer.")
    if gram_type not in {"ngram", "skipgram", "flexgram", "none"}:
        raise WrangleError("INVALID_OPTION", "gram_type must be ngram, skipgram, flexgram, or none.")
    if gram_type != 'none' and dims > len(features):
        raise WrangleError("INVALID_OPTION", "dims exceeds the number of parameter columns.")
    if gram_type == 'none':
        groups = [(name,) for name in features]
    elif gram_type == 'ngram':
        groups = [tuple(features[i:i + dims]) for i in range(len(features) - dims + 1)]
    elif gram_type == 'skipgram':
        groups = list(combinations(features, dims))
    else:
        groups = [group for size in range(1, dims + 1) for group in combinations(features, size)]
    label = 'parameters'
    if metric == label:
        label = '_parameters'
    expanded = pl.concat([frame(data).select(pl.concat_list([pl.struct(pl.lit(name).alias('column'), pl.col(name).cast(pl.String).alias('value')) for name in group]).alias(label), pl.col(metric)) for group in groups])
    result = df_to_groupby(expanded, label, group_by_func, seed=seed)
    if metric not in result.collect_schema():
        raise WrangleError("INVALID_OPTION", "Aggregation must produce the metric column.")
    return finish(result.sort(metric, descending=True, maintain_order=True), data)
