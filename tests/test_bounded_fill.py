import polars as pl
import pytest
import wrangle as wr


def test_bounded_fill_never_crosses_subject_or_declared_gap():
    data = pl.DataFrame({'id': [1, 2, 3, 4, 5, 6], 'subject': ['a', 'a', 'a', 'a', 'b', 'b'], 'time': [1, 2, 3, 4, 1, 2], 'x': [4, None, None, 8, None, 3]})
    result = wr.prepare(data, {'key': 'id', 'units': {'x': 'mg'}, 'steps': [{'op': 'window', 'by': ['subject'], 'order_by': ['time'], 'ties': 'error', 'metrics': {'forward': {'column': 'x', 'method': 'fill_forward', 'n': 1, 'nulls': 'keep'}, 'backward': {'column': 'x', 'method': 'fill_backward', 'n': 1, 'nulls': 'keep'}}}]})
    assert result.data['forward'].to_list() == [4, 4, None, 8, None, 3]
    assert result.data['backward'].to_list() == [4, None, 8, 8, 3, 3]
    assert result.receipt['units'] == {'x': 'mg', 'forward': 'mg', 'backward': 'mg'}
    assert data['x'].to_list() == [4, None, None, 8, None, 3]
    assert result.data['forward'].dtype == pl.Int64
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(data, {'key': 'id', 'steps': [{'op': 'window', 'by': ['subject'], 'order_by': ['time'], 'ties': 'error', 'metrics': {'fill': {'column': 'x', 'method': 'fill_forward', 'nulls': 'keep'}}}]})
    assert caught.value.code == 'INVALID_OPTION'
