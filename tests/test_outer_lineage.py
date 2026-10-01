import polars as pl
import pytest
import wrangle as wr


def test_full_join_retains_both_sources_identity_and_introduction_evidence():
    sources = {'left': pl.DataFrame({'id': [1, 2], 'x': [10, 20]}), 'right': pl.DataFrame({'id': [2, 3], 'y': [40, 60]})}
    step = {'op': 'join', 'source': 'right', 'on': 'id', 'how': 'full', 'unmatched': {'left': 'keep', 'right': 'keep'}, 'cardinality': '1:1', 'maintain_order': 'left_right', 'key': 'id', 'allow_expand': True}
    result = wr.prepare(sources, {'input': 'left', 'key': 'id', 'steps': [step]})
    assert result.data['id'].to_list() == [1, 2, 3]
    receipt = result.receipt['steps'][0]
    assert receipt['observation_counts']['introduced_observations'] == 1
    assert receipt['observation_transition']['introduced_source'] == 'right'
    assert receipt['parameters']['coalesce'] is True
    assert receipt['parameters']['cardinality'] == '1:1'


def test_balanced_outer_join_cannot_hide_new_observations_or_lost_parents():
    sources = {'left': pl.DataFrame({'id': [1, 2]}), 'right': pl.DataFrame({'id': [3, 4]})}
    step = {'op': 'join', 'source': 'right', 'on': 'id', 'how': 'right', 'unmatched': {'left': 'drop', 'right': 'keep'}, 'cardinality': '1:1', 'maintain_order': 'right_left', 'key': 'id', 'reason': 'Declared metadata universe'}
    with pytest.raises(wr.WrangleError) as error:
        wr.prepare(sources, {'input': 'left', 'key': 'id', 'steps': [step]})
    assert error.value.code == 'UNDECLARED_EXPANSION'
    result = wr.prepare(sources, {'input': 'left', 'key': 'id', 'steps': [{**step, 'allow_expand': True}]})
    assert result.data['id'].to_list() == [3, 4]
    assert result.receipt['steps'][0]['excluded_keys'] == [{'id': 1}, {'id': 2}]


def test_distinct_join_keys_keep_scientific_roles_and_payload_suffixes():
    sources = {'left': pl.DataFrame({'id': [1], 'code': ['a'], 'x': [2]}), 'right': pl.DataFrame({'subject': ['a'], 'x': [3]})}
    result = wr.prepare(sources, {'input': 'left', 'key': 'id', 'source_contracts': {'right': {'units': {'x': 'mg'}, 'descriptions': {'subject': 'External subject identifier'}}}, 'steps': [{'op': 'join', 'source': 'right', 'left_on': 'code', 'right_on': 'subject', 'cardinality': 'm:1', 'unmatched': 'error'}]})
    assert result.data.columns == ['id', 'code', 'x', 'subject', 'x_right']
    assert result.receipt['units'] == {'x_right': 'mg'}
    assert result.receipt['variables']['subject']['description'] == 'External subject identifier'


def test_matching_dynamic_axes_require_matching_physical_units():
    sources = {'left': pl.DataFrame({'id': [1, 2], 'time': [1, 2], 'unit': ['day', 'hour']}), 'right': pl.DataFrame({'time': [1, 2], 'unit': ['hour', 'day'], 'dose': [2, 3]})}
    recipe = {'input': 'left', 'key': 'id', 'units': {'time': '@unit'}, 'source_contracts': {'right': {'units': {'time': '@unit'}}}, 'steps': [{'op': 'join', 'source': 'right', 'on': 'time'}]}
    with pytest.raises(wr.WrangleError) as error:
        wr.prepare(sources, recipe)
    assert error.value.code == 'UNIT_MISMATCH'
    sources['right'] = sources['left'].select('time', 'unit').with_columns(pl.lit(2).alias('dose'))
    recipe['steps'][0]['on'] = ['time', 'unit']
    result = wr.prepare(sources, recipe)
    assert result.receipt['units'] == {'time': '@unit'}
    assert result.data['dose'].to_list() == [2, 2]
