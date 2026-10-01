"""Research contracts: identity, deliberate exclusions, replay, and publication."""
import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import wrangle as wr


def test_research_batch_replay_and_source_preservation(tmp_path):
    measurements = pl.DataFrame({"sample_id": ["001", "002", "003"], "mass_mg": ["1000", "2000", "3000"], "qc": ["pass", "fail", "pass"]})
    metadata = pl.DataFrame({"sample_id": ["003", "001", "002"], "group": ["treated", "control", "control"]})
    before = measurements.clone()
    recipe = {"version": 1, "input": "measurements", "key": "sample_id", "units": {"mass_mg": "mg"}, "steps": [
        {"op": "cast", "columns": {"mass_mg": "Float64"}},
        {"op": "join", "source": "metadata", "on": "sample_id"},
        {"op": "filter", "where": {"eq": [{"col": "qc"}, "pass"]}, "reason": "Instrument quality control"},
        {"op": "convert_unit", "column": "mass_mg", "from_unit": "mg", "to_unit": "g", "factor": 0.001},
        {"op": "rename", "columns": {"mass_mg": "mass_g"}},
    ], "checks": {"required": ["group", "mass_g"], "ranges": {"mass_g": {"min": 0}}, "row_count": {"exact": 2}}}
    sources = {"measurements": measurements, "metadata": metadata}
    first = wr.prepare(sources, recipe, output=tmp_path / "batch")
    second = wr.prepare(sources, recipe)
    assert first.data["sample_id"].to_list() == ["001", "003"]
    assert first.data["mass_g"].to_list() == [1.0, 3.0]
    assert first.data["group"].to_list() == ["control", "treated"]
    assert first.receipt["units"] == {"mass_g": "g"}
    assert first.receipt["steps"][2]["excluded_keys"] == [{"sample_id": "002"}]
    assert first.receipt == second.receipt
    assert_frame_equal(measurements, before)
    assert_frame_equal(pl.read_parquet(tmp_path / "batch" / "data.parquet"), first.data)
    assert json.loads((tmp_path / "batch" / "receipt.json").read_text()) == first.receipt
    with pytest.raises(wr.WrangleError, match="never overwritten"):
        first.write(tmp_path / "batch")


def test_csv_inspection_preserves_leading_zero_identifiers(tmp_path):
    source = tmp_path / "samples.csv"
    source.write_text("sample_id,value\n001,1.5\n002,\n")
    profile = wr.inspect(source)
    assert profile["columns"] == {"sample_id": "String", "value": "String"}
    assert profile["examples"][0]["sample_id"] == "001"
    assert profile["fields"][1]["nulls"] == 1
    assert wr.prepare(source, {"steps": [{"op": "cast", "columns": {"value": "Float64"}}]}).data["value"].to_list() == [1.5, None]


@pytest.mark.parametrize("steps,code", [
    ([{"op": "cast", "columns": {"value": "Int64"}}], "LOSSY_CAST"),
    ([{"op": "filter", "where": {"gt": [{"col": "value"}, 1]}}], "UNDECLARED_LOSS"),
    ([{"op": "derive", "columns": {"root": {"sqrt": -1}}}], "NONFINITE_RESULT"),
    ([{"op": "derive", "columns": {"value": 3}}], "COLUMN_EXISTS"),
    ([{"op": "run_python", "code": "print('unsafe')"}], "UNKNOWN_OPERATION"),
    ([{"op": "derive", "columns": {"result": {"eval": "1+1"}}}], "INVALID_EXPRESSION"),
    ([{"op": "rename", "columns": {"value": "id"}}], "DUPLICATE_COLUMNS"),
])
def test_invalid_transformations_fail_before_publication(tmp_path, steps, code):
    data = pl.DataFrame({"id": ["a", "b"], "value": [0.5, 2.0]})
    output = tmp_path / "must-not-exist"
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(data, {"steps": steps}, output=output)
    assert caught.value.code == code
    assert not output.exists()
    assert data["value"].to_list() == [0.5, 2.0]


def test_unknown_filter_truth_requires_explicit_policy():
    data = pl.DataFrame({"value": [1, None, 3]})
    step = {"op": "filter", "where": {"gt": [{"col": "value"}, 1]}, "reason": "Predefined threshold"}
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(data, {"steps": [step]})
    assert caught.value.code == "UNRESOLVED_FILTER"
    result = wr.prepare(data, {"steps": [{**step, "nulls": "keep"}]})
    assert result.data["value"].to_list() == [None, 3]


@pytest.mark.parametrize("metadata,code", [
    (pl.DataFrame({"id": ["a"], "group": ["control"]}), "UNMATCHED_KEYS"),
    (pl.DataFrame({"id": ["a", "a", "b"], "group": ["control", "treated", "control"]}), "JOIN_CARDINALITY"),
    (pl.DataFrame({"id": ["a", None], "group": ["control", "treated"]}), "NULL_KEY"),
])
def test_join_identity_failures(metadata, code):
    measurements = pl.DataFrame({"id": ["a", "b"], "value": [1, 2]})
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare({"measurements": measurements, "metadata": metadata}, {"input": "measurements", "steps": [{"op": "join", "source": "metadata", "on": "id"}]})
    assert caught.value.code == code


def test_output_contract_reports_violated_measurement_range():
    data = pl.DataFrame({"id": [1, 2], "value": [2, -1]})
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(data, {"key": "id", "checks": {"ranges": {"value": {"min": 0}}}})
    assert caught.value.code == "RANGE_VIOLATION"
    assert caught.value.details["affected_rows"] == 1


def test_category_mapping_is_stable_across_batches():
    recipe = {"steps": [{"op": "encode", "column": "group", "mapping": {"control": 0, "treated": 1}}]}
    assert wr.prepare(pl.DataFrame({"group": ["treated"]}), recipe).data["group"].to_list() == [1]
    with pytest.raises(wr.WrangleError):
        wr.prepare(pl.DataFrame({"group": ["new_group"]}), recipe)


def test_unit_conversion_rejects_protocol_disagreement():
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(pl.DataFrame({"mass": [1.0]}), {"units": {"mass": "kg"}, "steps": [{"op": "convert_unit", "column": "mass", "from_unit": "mg", "to_unit": "g", "factor": 0.001}]})
    assert caught.value.code == "UNIT_MISMATCH"


def test_yaml_recipe_file_and_lazy_source(tmp_path):
    path = tmp_path / "recipe.yaml"
    from wrangle._protocol import dump_recipe
    path.write_text(dump_recipe({"key": "id", "checks": {"row_count": {"exact": 2}}}))
    result = wr.prepare(pl.DataFrame({"id": [1, 2]}).lazy(), path)
    assert result.data.height == 2


@pytest.mark.parametrize("recipe", [
    {"version": True}, {"checks": {"required": "id"}}, {"checks": {"unique": "id"}},
    {"checks": {"ranges": {"value": {"min": 3, "max": 1}}}},
    {"checks": {"row_count": {"exact": -1}}}, {"steps": [{"op": "cast", "columns": ["value"]}]},
])
def test_malformed_contract_has_stable_failure(recipe):
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(pl.DataFrame({"id": [1], "value": [1]}), recipe)
    assert caught.value.code == "INVALID_RECIPE"


def test_changed_result_cannot_publish_a_stale_receipt(tmp_path):
    result = wr.prepare(pl.DataFrame({"value": [1, 2]}), {})
    result.data.replace_column(0, pl.Series("value", [100, 200]))
    with pytest.raises(wr.WrangleError) as caught:
        result.write(tmp_path / "changed")
    assert caught.value.code == "RESULT_CHANGED"
    assert not (tmp_path / "changed").exists()


def test_unit_metadata_requires_declared_new_meaning():
    data = pl.DataFrame({"id": [1, 2], "mass": [2.0, 4.0]})
    recipe = {"key": "id", "units": {"mass": "mg"}, "steps": [{"op": "df_rescale_meanzero", "retain": "id"}]}
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(data, recipe)
    assert caught.value.code == "UNDECLARED_UNITS"
    recipe["steps"][0]["units"] = {"mass": "1"}
    assert wr.prepare(data, recipe).receipt["units"] == {"mass": "1"}


def test_equal_invalid_counts_do_not_hide_new_infinity():
    data = pl.DataFrame({"x": [float("inf"), 1.0], "denominator": [1.0, 0.0]})
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(data, {"steps": [{"op": "derive", "columns": {"x": {"div": [1, {"col": "denominator"}]}}, "overwrite": True}]})
    assert caught.value.code == "NONFINITE_RESULT"


def test_nan_is_not_a_valid_join_identity():
    data = pl.DataFrame({"id": [float("nan")], "value": [1]})
    metadata = pl.DataFrame({"id": [float("nan")], "group": ["control"]})
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare({"data": data, "metadata": metadata}, {"input": "data", "steps": [{"op": "join", "source": "metadata", "on": "id"}]})
    assert caught.value.code == "NULL_KEY"


def test_hidden_python_plan_callback_never_executes():
    called = []
    source = pl.DataFrame({"value": [1]}).lazy().map_batches(lambda table: called.append(True) or table, schema={"value": pl.Int64})
    with pytest.raises(wr.WrangleError) as caught:
        wr.inspect(source)
    assert caught.value.code == "UNSUPPORTED_CALLBACK"
    assert not called


def test_key_values_cannot_be_reencoded_as_observations():
    data = pl.DataFrame({"id": ["001", "002"], "value": [1, 2]})
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(data, {"key": "id", "steps": [{"op": "cast", "columns": {"id": "Int64"}}]})
    assert caught.value.code == "KEY_CHANGED"


def test_receipt_mutation_cannot_publish(tmp_path):
    result = wr.prepare(pl.DataFrame({"value": [1]}), {})
    result.receipt["output"]["rows"] = 999
    with pytest.raises(wr.WrangleError) as caught:
        result.write(tmp_path / "edited-receipt")
    assert caught.value.code == "RESULT_CHANGED"


def test_numeric_cast_preserves_precision():
    data = pl.DataFrame({"measurement": [2**53 + 1]}, schema={"measurement": pl.Int64})
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(data, {"steps": [{"op": "cast", "columns": {"measurement": "Float64"}}]})
    assert caught.value.code == "LOSSY_CAST"


def test_snapshot_hash_depends_on_data_not_memory_chunks():
    one_chunk = pl.DataFrame({"measurement": [1, 2]})
    two_chunks = pl.concat([one_chunk.head(1), one_chunk.tail(1)], rechunk=False)
    assert one_chunk.n_chunks() == 1 and two_chunks.n_chunks() == 2
    assert wr.inspect(one_chunk)["sha256"] == wr.inspect(two_chunks)["sha256"]


def test_duplicate_csv_headers_are_not_silently_renamed(tmp_path):
    source = tmp_path / "duplicate.csv"
    source.write_text("id,id\na,b\n")
    with pytest.raises(wr.WrangleError) as caught:
        wr.inspect(source)
    assert caught.value.code == "DUPLICATE_COLUMNS"


def test_nan_comparison_cannot_become_a_known_measurement():
    source = pl.DataFrame({'x': [float('nan'), 2.0]})
    result = wr.prepare(source, {'steps': [{'op': 'derive', 'columns': {'x': {'ge': [{'col': 'x'}, 0]}}, 'overwrite': True}]})
    assert result.data['x'].to_list() == [None, True]
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(source, {'steps': [{'op': 'filter', 'where': {'ge': [{'col': 'x'}, 0]}}]})
    assert caught.value.code == 'UNRESOLVED_FILTER'


@pytest.mark.parametrize('dtype,value', [(pl.UInt8, 200), (pl.UInt16, 50000)])
def test_integer_unit_conversion_widens_without_wrapping(dtype, value):
    source = pl.DataFrame({'x': pl.Series([value], dtype=dtype)})
    result = wr.prepare(source, {'steps': [{'op': 'convert_unit', 'column': 'x', 'from_unit': 'instrument', 'to_unit': 'count', 'factor': 2}]})
    assert result.data['x'].to_list() == [value * 2]
    assert result.data.schema['x'] == pl.Int64


@pytest.mark.parametrize('expression', [
    {'add': [{'col': 'x'}, 1]}, {'sub': [-2, {'col': 'x'}]},
    {'mul': [{'col': 'x'}, 2]}, {'pow': [{'col': 'x'}, 3]},
    {'abs': -(2**63)},
])
def test_integer_overflow_fails_without_output(tmp_path, expression):
    source = pl.DataFrame({'x': [2**63 - 1]})
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(source, {'steps': [{'op': 'derive', 'columns': {'x': expression}, 'overwrite': True}]}, output=tmp_path / 'overflow')
    assert caught.value.code == 'INTEGER_OVERFLOW'
    assert not (tmp_path / 'overflow').exists()


def test_integer_power_preserves_boundary_and_exact_values():
    source = pl.DataFrame({'x': [-2, 3], 'power': [63, 10]})
    result = wr.prepare(source, {'steps': [{'op': 'derive', 'columns': {'power': {'pow': [{'col': 'x'}, {'col': 'power'}]}}, 'overwrite': True}]})
    assert result.data['power'].to_list() == [-(2**63), 59049]


def test_implicit_float_arithmetic_rejects_integer_precision_loss():
    source = pl.DataFrame({'x': [2**53 + 1]})
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(source, {'steps': [{'op': 'convert_unit', 'column': 'x', 'from_unit': 'mg', 'to_unit': 'g', 'factor': 0.001}]})
    assert caught.value.code == 'LOSSY_CAST'


def test_ndjson_ingestion_rejects_implicit_precision_loss(tmp_path):
    source = tmp_path / 'measurement.ndjson'
    source.write_text('{"x":9007199254740993}\n{"x":1.5}\n')
    with pytest.raises(wr.WrangleError) as caught:
        wr.inspect(source)
    assert caught.value.code == 'LOSSY_CAST'


def test_ndjson_ingestion_preserves_late_fields(tmp_path):
    source = tmp_path / 'measurement.ndjson'
    source.write_text('{"id":"001"}\n{"id":"002","mass":1.5}\n')
    profile = wr.inspect(source)
    assert profile['examples'] == [{'id': '001', 'mass': None}, {'id': '002', 'mass': 1.5}]


def test_ndjson_duplicate_fields_require_source_correction(tmp_path):
    source = tmp_path / 'measurement.ndjson'
    source.write_text('{"mass":1,"mass":2}\n')
    with pytest.raises(wr.WrangleError) as caught:
        wr.inspect(source)
    assert caught.value.code == 'DUPLICATE_COLUMNS'


def test_semantic_overwrite_removes_stale_variable_description():
    recipe = {'descriptions': {'mass': 'Specimen mass'}, 'units': {'mass': 'mg'}, 'steps': [{'op': 'df_to_binary', 'y': 'mass', 'units': {'mass': '1'}}]}
    result = wr.prepare(pl.DataFrame({'mass': [1, 2]}), recipe)
    assert result.receipt['variables']['mass']['description'] is None
    assert result.receipt['descriptions_unresolved'] == ['mass']
    recipe['steps'][0]['descriptions'] = {'mass': 'Mass exceeds the declared threshold'}
    assert wr.prepare(pl.DataFrame({'mass': [1, 2]}), recipe).receipt['variables']['mass']['description'] == 'Mass exceeds the declared threshold'


def test_infinity_cannot_disappear_into_a_boolean_measurement():
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(pl.DataFrame({'x': [float('inf')]}), {'steps': [{'op': 'derive', 'columns': {'x': {'ge': [{'col': 'x'}, 0]}}, 'overwrite': True}]})
    assert caught.value.code == 'NONFINITE_RESULT'


@pytest.mark.parametrize('expression', [{'not': 1}, {'and': [1, 2]}, {'add': ['a', 'b']}, {'lit': 10**300}])
def test_expression_types_cannot_change_operator_meaning(expression):
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(pl.DataFrame({'x': [1]}), {'steps': [{'op': 'derive', 'columns': {'y': expression}}]})
    assert caught.value.code == 'INVALID_EXPRESSION'


def test_join_key_dtype_mismatch_has_actionable_recovery():
    sources={'observations':pl.DataFrame({'id':['001']}),'metadata':pl.DataFrame({'id':[1]})}
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(sources, {'input':'observations','steps':[{'op':'join','source':'metadata','on':'id'}]})
    assert caught.value.code == 'DTYPE_MISMATCH'
    assert caught.value.details['columns'] == ['id']


def test_network_and_source_capabilities_never_execute_as_recipe_steps(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('A data recipe must not probe the network.')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    for operation in ['network_check', 'is_connected', 'read_large_csv', 'col_rescale_max']:
        with pytest.raises(wr.WrangleError) as caught:
            wr.prepare(pl.DataFrame({'x':[1]}), {'steps':[{'op':operation}]})
        assert caught.value.code == 'NOT_A_TABLE'


def test_recipe_replay_does_not_depend_on_source_chunk_layout():
    source=pl.DataFrame({'x':[1e16,1.0,-1e16,2.0,3.0]})
    split=pl.concat([source.head(2),source.slice(2)],rechunk=False)
    recipe={'steps':[{'op':'df_rescale_meanzero'}]}
    first=wr.prepare(source,recipe)
    second=wr.prepare(split,recipe)
    assert_frame_equal(first.data,second.data)
    assert first.receipt == second.receipt


def test_aggregation_requires_an_explicit_new_observation_unit():
    source=pl.DataFrame({'id':[1,2],'group':['a','a'],'mass':[1.,2.]})
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(source, {'key':'id','steps':[{'op':'df_to_groupby','by':'group','func':'mean'}]})
    assert caught.value.code == 'OBSERVATION_UNIT_CHANGED'


@pytest.mark.parametrize('operation',['create_synth_binary_model','df_corr_randomforest'])
def test_retired_model_operations_fail_with_scope_recovery(operation):
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(pl.DataFrame({'x':[1]}), {'steps':[{'op':operation}]})
    assert caught.value.code == 'MODEL_ENGINE_REQUIRED'


def test_decimal_measurements_cannot_lose_precision_implicitly():
    source=pl.DataFrame({'x':['9007199254740993.0']}).with_columns(pl.col('x').cast(pl.Decimal(38,1)))
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(source, {'steps':[{'op':'derive','columns':{'x':{'div':[{'col':'x'},1]}},'overwrite':True}]})
    assert caught.value.code == 'LOSSY_CAST'
