"""Scientific contracts for native Polars dataframe preparation."""

import math
import polars as pl
import pytest
from polars.testing import assert_frame_equal
import wrangle as wr
from wrangle._core import WrangleError


def materialize(data):
    return data.collect() if isinstance(data, pl.LazyFrame) else data


@pytest.mark.parametrize("lazy", [False, True])
def test_missingness_drop_threshold_and_imputation(lazy):
    original = pl.DataFrame({"sample": ["a", "b", "c", "d"], "x": [1.0, 9.0, None, float('nan')], "complete": [1, 2, 3, 4]})
    data = original.lazy() if lazy else original
    report = materialize(wr.df_find_nan(data))
    assert report.filter(pl.col('column') == 'x')['missing'].item() == 2
    assert report.filter(pl.col('column') == 'x')['missing_fraction'].item() == .5
    assert materialize(wr.df_drop_nanrows(data))['sample'].to_list() == ['a', 'b']
    assert materialize(wr.df_drop_nancols(data)).columns == ['sample', 'complete']
    assert materialize(wr.df_drop_nancols(data, .5)).columns == original.columns
    result = wr.df_impute_nan(data, ['x'], 'median')
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    assert materialize(result)['x'].to_list() == [1.0, 9.0, 5.0, 5.0]
    assert original['x'].null_count() == 1 and original['x'].is_nan().sum() == 1


def test_imputation_uses_requested_method_and_preserves_fractional_values():
    data = pl.DataFrame({'x': [1.25, 1.25, 9.75, None]})
    assert wr.df_impute_nan(data, 'x', 'mean')['x'][-1] == pytest.approx(12.25 / 3)
    assert wr.df_impute_nan(data, 'x', 'median')['x'][-1] == 1.25
    assert wr.df_impute_nan(data, 'x', 'mode')['x'][-1] == 1.25
    assert_frame_equal(wr.df_impute_nan(data, 'x', seed=42), wr.df_impute_nan(data, 'x', seed=42))
    with pytest.raises(WrangleError, match='without observed') as exc:
        wr.df_impute_nan(pl.DataFrame({'x': [None]}, schema={'x': pl.Float64}), 'x', 'mean')
    assert exc.value.code == 'NO_OBSERVATIONS'
    with pytest.raises(WrangleError) as exc:
        wr.df_impute_nan(data, 'x', 'imaginary')
    assert exc.value.code == 'INVALID_OPTION'


@pytest.mark.parametrize("lazy", [False, True])
def test_scaling_preserves_identity_nulls_and_constants(lazy):
    original = pl.DataFrame({'sample': ['a', 'b', 'c'], 'x': [1., 4., None], 'constant': [3, 3, 3], 'target': [4, 9, 16]})
    data = original.lazy() if lazy else original
    for operation in (wr.df_rescale_log, wr.df_rescale_sqrt):
        result = materialize(operation(data, retain_cols='target'))
        assert result.columns == original.columns
        assert result['sample'].equals(original['sample'])
        assert result['target'].equals(original['target'])
        assert result['x'].null_count() == 1
    standardized = materialize(wr.df_rescale_meanzero(data, retain='target'))
    assert standardized['constant'].to_list() == [0., 0., 0.]
    assert standardized['x'].mean() == pytest.approx(0)
    assert standardized['x'].std() == pytest.approx(1)
    with pytest.raises(WrangleError) as exc:
        wr.df_rescale_sqrt(pl.DataFrame({'x': [-.1]}))
    assert exc.value.code == 'INVALID_DOMAIN'
    with pytest.raises(WrangleError) as exc:
        wr.df_rescale_log(pl.DataFrame({'x': [-1.]}))
    assert exc.value.code == 'INVALID_DOMAIN'


def test_score_uses_declared_metric_and_missing_is_not_a_tie():
    data = pl.DataFrame({'accuracy': [1., 2., 3., 4., None]})
    result = wr.df_add_scorecol(data, 'accuracy')
    assert result['score'].to_list() == [-2, 0, 2, 2, None]
    assert data.columns == ['accuracy']


def test_names_preserve_excluded_values_and_reject_collisions():
    data = pl.DataFrame({'Feature One': [1], 'target': [2], 'Feature Two': [3]})
    renamed = wr.df_rename_cols(data, exclude='target')
    assert renamed.columns == ['C0', 'target', 'C1']
    assert renamed['target'].to_list() == [2]
    assert wr.df_clean_colnames(data).columns == ['feature_one', 'target', 'feature_two']
    for callback in (lambda: wr.df_clean_colnames(pl.DataFrame({'a-b': [1], 'a b': [2]})), lambda: wr.df_rename_col(data, 'Feature One', 'target')):
        with pytest.raises(WrangleError) as exc:
            callback()
        assert exc.value.code == 'COLUMN_COLLISION'
    with pytest.raises(WrangleError) as exc:
        wr.df_drop_col(data, 'absent')
    assert exc.value.code == 'UNKNOWN_COLUMN'


def test_text_cleanup_retains_numeric_types_and_mixed_strings():
    data = pl.DataFrame({'label': ['  A ', '  ', None], 'numeric': ['1', '2', None], 'mixed': ['1', 'no', None], 'x': [1, 2, 3]})
    cleaned = wr.df_fill_empty(data, None)
    assert cleaned['label'].to_list() == ['A', None, None]
    assert cleaned['x'].dtype == pl.Int64
    assert wr.df_to_lower(cleaned)['label'].to_list() == ['a', None, None]
    converted = wr.df_to_numeric(cleaned)
    assert converted['numeric'].dtype == pl.Int64
    assert converted['mixed'].dtype == pl.String
    assert converted['numeric'].to_list() == [1, 2, None]


def test_join_validates_cardinality_missing_keys_and_observation_loss():
    left = pl.DataFrame({'sample': ['b', 'a'], 'measurement': [2, 1]})
    right = pl.DataFrame({'sample': ['a', 'b'], 'condition': ['treated', 'control']})
    joined = wr.df_merge(left.lazy(), right.lazy(), on_column='sample')
    assert isinstance(joined, pl.LazyFrame)
    assert joined.collect()['condition'].to_list() == ['control', 'treated']
    with pytest.raises(WrangleError) as exc:
        wr.df_merge(left, pl.concat([right, right.head(1)]), on_column='sample')
    assert exc.value.code == 'JOIN_CARDINALITY'
    with pytest.raises(WrangleError) as exc:
        wr.df_merge(left, right.head(1), on_column='sample')
    assert exc.value.code == 'JOIN_LOSS'
    assert wr.df_merge(left, right.head(1), on_column='sample', how='left', allow_unmatched=True).height == 2
    with pytest.raises(WrangleError) as exc:
        wr.df_merge(left, pl.DataFrame({'sample': ['a', None]}), on_column='sample')
    assert exc.value.code == 'MISSING_KEY'
    with pytest.raises(WrangleError) as exc:
        wr.df_merge(left, pl.DataFrame({'extra': [1]}))
    assert exc.value.code == 'JOIN_LOSS'


def test_resampling_has_equal_strata_reproducibility_and_source_order():
    data = pl.DataFrame({'sample': list(range(20)), 'condition': ['a'] * 12 + ['b'] * 8})
    first = wr.df_resample_stratified(data, 'condition', 5, seed=17)
    assert_frame_equal(first, wr.df_resample_stratified(data.lazy(), 'condition', 5, seed=17).collect())
    assert first.group_by('condition').len()['len'].to_list() == [5, 5]
    assert first['sample'].to_list() == sorted(first['sample'].to_list())
    assert first['sample'].n_unique() == 10
    assert not first.equals(wr.df_resample_stratified(data, 'condition', 5, seed=18))
    with pytest.raises(WrangleError) as exc:
        wr.df_resample_stratified(data, 'condition', 9)
    assert exc.value.code == 'INSUFFICIENT_SAMPLES'
    ids = wr.df_resample_id(data, 'condition')
    assert ids['sample'].to_list() == [0, 12]


def test_encoding_maps_are_sorted_missing_is_explicit_and_onehot_conserves_rows():
    data = pl.DataFrame({'condition': ['z', 'a', None, 'a'], 'target': [1, 2, 3, 4]})
    encoded, maps = wr.df_to_multiclass(data.lazy(), ignore_y='target', return_mapping=True)
    assert encoded.collect()['condition'].to_list() == [1, 0, -1, 0]
    assert maps['condition']['value'].to_list() == ['a', 'z']
    hot, maps = wr.df_to_multilabel(data, ignore_y='target', return_mapping=True)
    assert hot.height == 4
    assert hot.select(pl.sum_horizontal(pl.exclude('target'))).to_series().to_list() == [1] * 4
    assert maps['condition']['value'].to_list() == ['a', 'z', None]
    assert hot['condition__2'].to_list() == [0, 0, 1, 0]
    with pytest.raises(WrangleError) as exc:
        wr.df_to_multilabel(pl.DataFrame({'a': ['x'], 'a__0': [1]}))
    assert exc.value.code == 'COLUMN_COLLISION'


def test_correlations_are_pairwise_complete_and_kendall_corrects_ties():
    data = pl.DataFrame({'x': [1., 1., 2., None], 'y': [1., 2., 3., 999.], 'constant': [5., 5., 5., 5.], 'condition': ['a', 'a', 'b', 'b']})
    coeffs, excluded = wr.df_corr_pearson(data, 'y', excluded_list=True)
    assert coeffs['x'][0] == pytest.approx(math.sqrt(.75))
    assert coeffs['constant'][0] is None
    assert excluded == ['condition']
    matrix = wr.df_corr_any(data, 'kendall')
    assert matrix.filter(pl.col('column') == 'x')['y'].item() == pytest.approx(2 / math.sqrt(6))
    kept = wr.df_drop_weak(data, 'y', .8)
    assert kept.columns == ['x', 'y']


def test_native_groups_order_mode_and_parameter_combinations():
    data = pl.DataFrame({'condition': ['b', 'a', 'b', 'a'], 'batch': ['x', 'x', 'x', 'x'], 'value': [1., 2., 3., 4.]})
    grouped = wr.df_to_groupby(data.lazy(), 'condition', 'mean').collect()
    assert grouped['condition'].to_list() == ['b', 'a']
    assert grouped['value'].to_list() == [2., 3.]
    parameters = wr.df_groupby_params(data, 'value', 2)
    assert parameters.height == 2
    assert parameters['value'].to_list() == [6., 4.]
    assert parameters['parameters'].dtype == pl.List(pl.Struct({'column': pl.String, 'value': pl.String}))
    native = wr.df_to_groupby(data, 'condition', pl.col('value').max().alias('peak'))
    assert native['peak'].to_list() == [3., 4.]


def test_split_restructure_counts_frequencies_and_native_processing():
    data = pl.DataFrame({'first_x': [1, 1], 'last_x': [2, 3], 'target': [4, 5]})
    splits = wr.df_to_dfs(data, ['first_', 'last_'], 'target')
    assert splits['first_'].columns == ['x', 'target']
    x, y = wr.df_to_xy(data, 'target')
    assert x.columns == ['first_x', 'last_x'] and isinstance(y, pl.Series)
    x_lazy, y_lazy = wr.df_to_xy(data.lazy(), 'target')
    assert isinstance(x_lazy, pl.LazyFrame) and isinstance(y_lazy, pl.LazyFrame)
    assert wr.df_count_uniques(data)['first_x'].item() == 1
    assert wr.df_print_values(data)['first_x']['len'].item() == 2
    paired = wr.df_restructure_values(data, 'tuple')
    assert paired['first_x'][0] == {'column': 'first_x', 'value': 1}
    result = wr.df_parallelize_process(data, pl.col('first_x') * 3)
    assert result['first_x'].to_list() == [3, 3]
    with pytest.raises(WrangleError) as exc:
        wr.df_parallelize_process(data, lambda rows: rows)
    assert exc.value.code == 'UNSUPPORTED_CALLBACK'


def test_model_fitting_is_retired_and_destructive_mutation_rejected():
    data = pl.DataFrame({'x': [1], 'target': [2]})
    for operation in (wr.df_corr_randomforest, wr.df_corr_extratrees):
        with pytest.raises(WrangleError) as exc:
            operation(data, 'target')
        assert exc.value.code == 'MODEL_ENGINE_REQUIRED'
    with pytest.raises(WrangleError) as exc:
        wr.df_rename_cols(data, destructive=True)
    assert exc.value.code == 'IMMUTABLE_INPUT'


def test_embedded_callbacks_never_execute():
    invoked = []

    def callback(values):
        invoked.append(True)
        return values

    expression = pl.col("x").map_batches(callback, return_dtype=pl.Int64)
    data = pl.DataFrame({"group": ["a", "a"], "x": [1, 2]})
    for operation in (lambda: wr.df_parallelize_process(data, expression), lambda: wr.df_to_groupby(data, "group", expression)):
        with pytest.raises(WrangleError) as exc:
            operation()
        assert exc.value.code == "UNSUPPORTED_CALLBACK"
    assert invoked == []


def test_ols_mapping_preserves_equal_category_means_and_binary_target():
    data = pl.DataFrame({"condition": ["a", "a", "b", "b", "c", "c"], "value": [1., 3., 2., 2., 4., 6.]})
    ranked, maps = wr.df_corr_ols(data, "value", return_mapping=True)
    assert ranked['condition'].to_list() == [1, 1, 1, 1, 2, 2]
    assert maps['condition']['coefficient'].to_list() == [2., 2., 5.]
    assert wr.df_to_binary(data, 'value')['value'].to_list() == [False, True, False, False, True, True]


def test_entropy_rejects_zero_mass_and_numeric_conversion_needs_evidence():
    with pytest.raises(WrangleError) as exc:
        wr.df_to_groupby(pl.DataFrame({'group': ['a', 'a'], 'x': [0., 0.]}), 'group', 'entropy')
    assert exc.value.code == 'INVALID_DOMAIN'
    assert wr.df_to_numeric(pl.DataFrame({'unknown': [None]}, schema={'unknown': pl.String})).schema['unknown'] == pl.String


def test_native_kendall_fails_before_quadratic_memory_allocation():
    data = pl.DataFrame({'x': list(range(5000)), 'y': list(range(5000))})
    with pytest.raises(WrangleError) as exc:
        wr.df_corr_any(data, 'kendall')
    assert exc.value.code == 'NATIVE_CORRELATION_LIMIT'
    assert exc.value.details['pairs'] == 5000 * 4999 // 2


def test_random_imputation_replays_with_distinct_named_column_streams():
    data = pl.DataFrame({
        'first': [0., 2., None, None, None, None],
        'second': [100., 120., None, None, None, None],
    })
    eager = wr.df_impute_nan(data, ['first', 'second'], 'mean_by_std', seed=41)
    lazy = wr.df_impute_nan(data.lazy(), ['first', 'second'], 'mean_by_std', seed=41).collect()
    assert_frame_equal(eager, lazy)
    assert_frame_equal(eager, wr.df_impute_nan(data, ['first', 'second'], 'mean_by_std', seed=41))
    assert eager.head(2).equals(data.head(2))
    standardized = eager.slice(2).select(
        ((pl.col('first') - 1.) / math.sqrt(2.)).alias('first'),
        ((pl.col('second') - 110.) / math.sqrt(200.)).alias('second'),
    )
    assert standardized['first'].to_list() != pytest.approx(standardized['second'].to_list())
    assert all(-1 <= value <= 1 for value in standardized['first'])
    assert all(-1 <= value <= 1 for value in standardized['second'])
    assert not eager.equals(wr.df_impute_nan(data, ['first', 'second'], 'mean_by_std', seed=42))


@pytest.mark.parametrize('dtype,value', [(pl.Int64, 2**63 - 1), (pl.UInt64, 2**64 - 1)])
def test_integer_group_sums_are_exact_beyond_the_input_range(dtype, value):
    data = pl.DataFrame({'group': ['a', 'a'], 'x': pl.Series([value, value], dtype=dtype)})
    result = wr.df_to_groupby(data, 'group', 'sum')
    assert result['x'].item() == value * 2
    assert result.schema['x'] == pl.Int128
    assert_frame_equal(result, wr.df_to_groupby(data.lazy(), 'group', 'sum').collect())
    parameters = wr.df_groupby_params(data, 'x', 1, group_by_func='sum')
    assert parameters['x'].item() == value * 2


def test_extreme_int128_sums_fail_before_wrapping():
    data = pl.DataFrame({'group': ['a', 'a'], 'x': pl.Series([2**126, 2**126], dtype=pl.Int128)})
    with pytest.raises(WrangleError) as exc:
        wr.df_to_groupby(data, 'group', 'sum')
    assert exc.value.code == 'ARITHMETIC_OVERFLOW'


@pytest.mark.parametrize('mode', ['mean', 'median', 'mean_by_std'])
def test_imputation_refuses_to_change_large_observed_integers(mode):
    data = pl.DataFrame({'x': [2**53, 2**53 + 1, None]})
    with pytest.raises(WrangleError) as exc:
        wr.df_impute_nan(data, 'x', mode)
    assert exc.value.code == 'LOSSY_CAST'
    assert data['x'].to_list() == [2**53, 2**53 + 1, None]


def test_standardization_refuses_to_fabricate_a_constant_from_distinct_integers():
    data = pl.DataFrame({'x': [2**53, 2**53 + 1]})
    for source in (data, data.lazy()):
        with pytest.raises(WrangleError) as exc:
            wr.df_rescale_meanzero(source)
        assert exc.value.code == 'LOSSY_CAST'


def test_kendall_compares_extreme_integers_without_float_rounding_or_subtraction():
    data = pl.DataFrame({'x': [-(2**63), 2**53, 2**53 + 1, 2**63 - 1], 'y': [1, 2, 3, 4]})
    result = wr.df_corr_any(data, 'kendall')
    assert result.filter(pl.col('column') == 'x')['y'].item() == pytest.approx(1.)
    with pytest.raises(WrangleError) as exc:
        wr.df_corr_pearson(data, 'y')
    assert exc.value.code == 'LOSSY_CAST'


@pytest.mark.parametrize('func', ['mean', 'median', 'std', 'entropy'])
def test_float_group_statistics_reject_unrepresentable_observations(func):
    data = pl.DataFrame({'group': ['a', 'a'], 'x': [2**53, 2**53 + 1]})
    with pytest.raises(WrangleError) as exc:
        wr.df_to_groupby(data, 'group', func)
    assert exc.value.code == 'LOSSY_CAST'
