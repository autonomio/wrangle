import polars as pl
import pytest
import wrangle as wr


def test_declared_nested_nan_normalization_preserves_shape_parent_null_and_counts():
    data = pl.DataFrame({'id': [1, 2, 3], 'values': [[float('nan'), 2.0], [], None], 'record': pl.Series([{'x': float('nan')}, {'x': 3.0}, None], dtype=pl.Struct({'x': pl.Float64})), 'array': pl.Series([[float('nan'), 4.0], [5.0, 6.0], None], dtype=pl.Array(pl.Float64, 2))})
    recipe = {'key': 'id', 'steps': [{'op': 'normalize_missing', 'columns': {'values': [], 'record': [], 'array': []}, 'nan': True}]}
    result = wr.prepare(data, recipe)
    assert result.data['values'].to_list() == [[None, 2.0], [], None]
    assert result.data['record'].to_list() == [{'x': None}, {'x': 3.0}, None]
    assert result.data['array'].to_list() == [[None, 4.0], [5.0, 6.0], None]
    assert result.data.schema == data.schema
    assert result.receipt['steps'][0]['replacements'] == {'values': 1, 'record': 1, 'array': 1}
    assert wr.inspect(data)['fields'][1]['nans'] == 1
    recipe['steps'][0]['nan'] = False
    with pytest.raises(wr.WrangleError) as error:
        wr.prepare(data, recipe)
    assert error.value.code == 'NONFINITE_RESULT'
