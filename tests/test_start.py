"""Research answers remain explicit, recoverable, and checked by ordinary prepare."""
from copy import deepcopy
from pathlib import Path
import shlex

import polars as pl
import pytest

import wrangle as wr
from wrangle._core import WrangleError
from wrangle._protocol import dump_recipe, load_recipe
from wrangle._start import QUESTION_DEFINITIONS, draft, publish, validate_decisions


@pytest.fixture
def study(tmp_path):
    observations = tmp_path / 'instrument.csv'
    metadata = tmp_path / 'samples.csv'
    observations.write_text('sample_id,mass_mg,qc\n001,1000,pass\n002,NA,fail\n003,3000,pass\n', encoding='utf-8')
    metadata.write_text('sample_id,group\n003,treated\n001,control\n002,treated\n', encoding='utf-8')
    return {'measurements': observations, 'metadata': metadata}


@pytest.fixture
def answers():
    return {
        'input': 'measurements', 'observation': 'One sample at one visit',
        'key': ['sample_id'], 'measurements': {'mass_mg': {'dtype': 'Float64', 'unit': 'mg'}},
        'missing': {'codes': {'mass_mg': ['NA']}, 'action': 'keep'},
        'exclusions': {'action': 'keep_values', 'column': 'qc', 'values': ['pass'], 'reason': 'Predefined instrument QC acceptance', 'nulls': 'error'},
        'matching': {'action': 'attach', 'source': 'metadata', 'left_on': ['sample_id'], 'right_on': ['sample_id'], 'cardinality': '1:1', 'unmatched': 'error', 'unused': 'error', 'overlap': 'error', 'nulls': 'error'},
        'descriptions': {'sample_id': 'Sample identifier', 'mass_mg': 'Specimen mass'},
    }


def test_blank_draft_observes_without_guessing_scientific_choices(study):
    proposal = draft(study, input='measurements')
    assert proposal['ready'] is False and proposal['data_validated'] is False
    assert proposal['recipe']['research_decisions'] == {'input': 'measurements'}
    assert proposal['recipe']['steps'] == []
    assert proposal['unresolved'] == ['observation', 'key', 'measurements', 'missing', 'matching', 'exclusions']
    assert proposal['observations']['measurements']['examples'][0]['sample_id'] == '001'
    definitions = {item['id']: item for item in QUESTION_DEFINITIONS}
    for question in proposal['questions']:
        assert question['question'] == definitions[question['id']]['question']
        assert question['schema']['sample_id'] == 'String'
    matching = next(item for item in proposal['questions'] if item['id'] == 'matching')
    assert list(matching['sources']) == ['metadata']


def test_pending_decisions_block_before_any_source_access(study, monkeypatch):
    proposal = draft(study, input='measurements')
    def forbidden(*args, **kwargs):
        raise AssertionError('unresolved protocol must not read sources')
    monkeypatch.setattr('wrangle._api._source', forbidden)
    with pytest.raises(WrangleError) as caught:
        wr.prepare({'measurements': 'missing.csv'}, proposal['recipe'])
    assert caught.value.code == 'UNRESOLVED_PROTOCOL'
    assert 'key' in str(caught.value.details)


def test_removing_pending_markers_cannot_complete_missing_answers(study, monkeypatch):
    proposal = draft(study, input='measurements')
    proposal['recipe']['pending_decisions'] = []
    monkeypatch.setattr('wrangle._api._source', lambda *a, **k: pytest.fail('read incomplete protocol'))
    with pytest.raises(WrangleError) as caught:
        wr.prepare(study, proposal['recipe'])
    assert caught.value.code == 'INVALID_RECIPE'


def test_answered_draft_uses_native_checks_and_records_evidence(study, answers):
    before = {name: path.read_bytes() for name, path in study.items()}
    proposal = draft(study, answers)
    assert proposal['ready'] is True and proposal['data_validated'] is False
    assert proposal['questions'] == []
    result = wr.prepare(proposal['sources'], proposal['recipe'])
    assert result.data.to_dicts() == [
        {'sample_id': '001', 'mass_mg': 1000.0, 'qc': 'pass', 'group': 'control'},
        {'sample_id': '003', 'mass_mg': 3000.0, 'qc': 'pass', 'group': 'treated'},
    ]
    assert result.receipt['recipe']['research_decisions']['observation'] == answers['observation']
    assert result.receipt['steps'][-1]['reason'] == answers['exclusions']['reason']
    assert result.receipt['steps'][-1]['excluded_keys'] == [{'sample_id': '002'}]
    assert all(path.read_bytes() == before[name] for name, path in study.items())


def test_guided_recipe_memory_and_disk_outputs_match(study, answers, tmp_path):
    proposal = draft(study, answers)
    memory = wr.prepare(proposal['sources'], proposal['recipe'])
    disk = wr.prepare(proposal['sources'], proposal['recipe'], execution='disk', output=tmp_path/'prepared')
    assert disk.data.collect().equals(memory.data)
    assert disk.receipt['output']['sha256'] == memory.receipt['output']['sha256']
    assert disk.receipt['recipe_sha256'] == memory.receipt['recipe_sha256']


@pytest.mark.parametrize('unit', [None, 'none', 'unknown', '?'])
def test_unknown_measurement_units_remain_pending(study, answers, unit):
    answers['measurements']['mass_mg']['unit'] = unit
    proposal = draft(study, answers)
    assert proposal['unresolved'] == ['measurements']
    assert 'mass_mg' not in proposal['recipe'].get('units', {})
    proposal['recipe']['pending_decisions'] = []
    with pytest.raises(WrangleError) as caught:
        validate_decisions(proposal['recipe'])
    assert caught.value.code == 'INVALID_RECIPE'


def test_dimensionless_unit_is_explicit_text(study, answers):
    answers['measurements']['mass_mg']['unit'] = '1'
    assert draft(study, answers)['recipe']['units']['mass_mg'] == '1'
    answers['measurements']['mass_mg']['unit'] = 1
    with pytest.raises(WrangleError) as caught:
        draft(study, answers)
    assert caught.value.code == 'START_ANSWER'


def test_csv_text_cannot_be_kept_as_measurement_representation(study, answers):
    answers['measurements']['mass_mg']['dtype'] = 'keep'
    with pytest.raises(WrangleError) as caught:
        draft(study, answers)
    assert caught.value.code == 'START_ANSWER'
    assert caught.value.details['observed'] == 'String'


def test_numeric_native_measurement_can_keep_representation(tmp_path):
    path = tmp_path/'observations.parquet'
    pl.DataFrame({'id': ['001'], 'count': [7]}, schema={'id': pl.String, 'count': pl.Int32}).write_parquet(path)
    values = {'observation': 'One specimen', 'key': ['id'], 'measurements': {'count': {'dtype': 'keep', 'unit': '1'}}, 'missing': {'codes': {}, 'action': 'keep'}, 'exclusions': {'action': 'none'}}
    proposal = draft({'measurements': path}, values)
    assert proposal['answers']['matching'] == {'action': 'none'}
    assert proposal['recipe']['steps'] == []
    assert wr.prepare(proposal['sources'], proposal['recipe']).data.schema['count'] == pl.Int32


@pytest.mark.parametrize('field,value', [('extra', True), ('key', ['invented']), ('matching', {'action': 'attach'}), ('missing', {'codes': {'mass_mg': [-999]}, 'action': 'keep'}), ('exclusions', {'action': 'infer_outliers'})])
def test_unsupported_or_unobserved_answers_fail_without_guessing(study, answers, field, value):
    answers[field] = value
    with pytest.raises(WrangleError) as caught:
        draft(study, answers)
    assert caught.value.code == 'START_ANSWER'


@pytest.mark.parametrize('cardinality', [None, '1:m', 'm:m'])
def test_metadata_cardinality_is_explicit_and_cannot_expand(study, answers, cardinality):
    if cardinality is None:
        del answers['matching']['cardinality']
    else:
        answers['matching']['cardinality'] = cardinality
    with pytest.raises(WrangleError) as caught:
        draft(study, answers)
    assert caught.value.code == 'START_ANSWER'


def test_duplicate_metadata_is_discovered_by_prepare_not_called_validated_by_start(study, answers):
    study['metadata'].write_text('sample_id,group\n001,control\n001,treated\n002,treated\n003,treated\n')
    proposal = draft(study, answers)
    assert proposal['ready'] is True and proposal['data_validated'] is False
    with pytest.raises(WrangleError) as caught:
        wr.prepare(proposal['sources'], proposal['recipe'])
    assert caught.value.code == 'DUPLICATE_KEY'


def test_unmatched_policy_is_an_explicit_research_decision(study, answers):
    study['metadata'].write_text('sample_id,group\n001,control\n002,treated\n')
    proposal = draft(study, answers)
    with pytest.raises(WrangleError) as caught:
        wr.prepare(proposal['sources'], proposal['recipe'])
    assert caught.value.code == 'UNMATCHED_KEYS'
    answers['matching']['unmatched'] = 'keep'
    proposal = draft(study, answers)
    assert wr.prepare(proposal['sources'], proposal['recipe']).data['group'].to_list() == ['control', None]


def test_zero_missing_policy_checks_final_output(study, answers):
    study['metadata'].write_text('sample_id,group\n001,control\n002,treated\n')
    answers['matching']['unmatched'] = 'keep'
    answers['missing']['action'] = 'error'
    proposal = draft(study, answers)
    with pytest.raises(WrangleError) as caught:
        wr.prepare(proposal['sources'], proposal['recipe'])
    assert caught.value.code == 'MISSINGNESS_VIOLATION'
    assert caught.value.details['column'] == 'group'


@pytest.mark.parametrize('mutation', ['key', 'unit', 'cast', 'missing', 'filter', 'matching', 'reason', 'order'])
def test_edited_recipe_cannot_contradict_recorded_decisions(study, answers, mutation):
    recipe = draft(study, answers)['recipe']
    if mutation == 'key': recipe['key'] = ['qc']
    if mutation == 'unit': recipe['units']['mass_mg'] = 'g'
    if mutation == 'cast': recipe['steps'][1]['columns']['mass_mg'] = 'Int64'
    if mutation == 'missing': recipe['steps'][0]['nan'] = True
    if mutation == 'filter': recipe['steps'][-1]['where'] = {'eq': [{'col': 'qc'}, 'fail']}
    if mutation == 'matching': recipe['steps'][2]['cardinality'] = 'm:m'
    if mutation == 'reason': recipe['steps'][-1]['reason'] = 'Different protocol'
    if mutation == 'order': recipe['steps'].reverse()
    with pytest.raises(WrangleError) as caught:
        validate_decisions(recipe)
    assert caught.value.code == 'INVALID_RECIPE'


def test_saved_pending_answers_can_be_completed_and_rerun(study, answers, tmp_path):
    saved = publish(draft(study, input='measurements'), tmp_path/'protocol')
    assert set(saved['files']) == {'README.md', 'answers.yaml', 'recipe.yaml'}
    pending = load_recipe(saved['answers_path'])
    assert pending['key'] is None and pending['matching'] is None
    Path(saved['answers_path']).write_text(dump_recipe(answers), encoding='utf-8')
    resolved = draft(saved['sources'], saved['answers_path'])
    assert resolved['ready'] is True
    assert load_recipe(saved['recipe_path'])['pending_decisions']
    published = publish(resolved, tmp_path/'resolved')
    assert load_recipe(published['recipe_path']) == resolved['recipe']
    assert wr.prepare(resolved['sources'], published['recipe_path']).data.height == 2
    assert '# Preparation is blocked' in Path(published['recipe_path']).read_text()


def test_saved_protocol_preserves_declared_parsing_on_answer_rerun(study, answers, tmp_path):
    supplied = {name: path for name, path in study.items()}
    supplied['measurements'] = {'path': study['measurements'], 'format': 'csv', 'options': {'separator': ',', 'decimal_comma': False}}
    first = draft(supplied, answers)
    saved = publish(first, tmp_path/'protocol')
    second = draft(saved['sources'], saved['answers_path'])
    assert second['recipe'] == first['recipe']
    assert second['answers']['source_options']['measurements']['options']['decimal_comma'] is False
    changed = deepcopy(supplied)
    changed['measurements']['options']['decimal_comma'] = True
    with pytest.raises(WrangleError) as caught:
        draft(changed, saved['answers_path'])
    assert caught.value.code == 'START_ANSWER'


def test_publishing_never_overwrites_and_cleans_temporary_files(study, tmp_path, monkeypatch):
    proposal = draft(study, input='measurements')
    directory = tmp_path/'protocol'
    directory.mkdir(); sentinel = directory/'keep'; sentinel.write_text('existing')
    with pytest.raises(WrangleError) as caught:
        publish(proposal, directory)
    assert caught.value.code == 'OUTPUT_EXISTS' and sentinel.read_text() == 'existing'
    def fail_publication(*args):
        raise WrangleError('ATOMIC_PUBLICATION_UNSUPPORTED', 'Unsupported native exclusive publication')
    monkeypatch.setattr('wrangle._start.publish_directory', fail_publication)
    with pytest.raises(WrangleError):
        publish(proposal, tmp_path/'failed')
    assert not (tmp_path/'failed').exists()
    assert not list(tmp_path.glob('.wrangle-start-*'))
    assert not list(tmp_path.glob('*.wrangle-lock'))


def test_published_commands_quote_source_paths_as_data(tmp_path):
    source = tmp_path/'$(touch unsafe) `other` data.csv'
    source.write_text('id\n001\n')
    proposal = draft({'measurements': source})
    result = publish(proposal, tmp_path/'protocol')
    command = shlex.split(result['next_commands'][0])
    assert command[command.index('--source')+1] == 'measurements=' + str(source.resolve())
    assert not (tmp_path/'unsafe').exists()


def test_three_file_workflows_require_an_ordinary_recipe(study):
    study['third'] = study['metadata']
    with pytest.raises(WrangleError) as caught:
        draft(study)
    assert caught.value.code == 'START_SCOPE'


def test_native_temporal_and_nested_schema_are_recorded_without_inference(tmp_path):
    import datetime
    source = tmp_path/'observations.parquet'
    pl.DataFrame({'id': ['001'], 'when': [datetime.datetime(2026, 1, 1)], 'detail': [{'count': 1}], 'items': [[1, 2]]}).write_parquet(source)
    values = {'observation': 'One sample', 'key': ['id'], 'measurements': {}, 'missing': {'codes': {}, 'action': 'keep'}, 'exclusions': {'action': 'none'}}
    proposal = draft({'measurements': source}, values)
    assert proposal['recipe']['checks']['schema']['when'] == {'Datetime': {'time_unit': 'us', 'time_zone': None}}
    assert proposal['recipe']['checks']['schema']['detail'] == {'Struct': {'count': 'Int64'}}
    assert wr.prepare(proposal['sources'], proposal['recipe']).data.height == 1


@pytest.mark.parametrize('mutation', ['ready', 'status', 'validated', 'questions', 'unresolved', 'pending', 'answers', 'recorded', 'recipe'])
def test_publishing_rejects_altered_decision_status_before_creating_files(study, answers, tmp_path, mutation):
    proposal = draft(study, answers)
    if mutation == 'ready': proposal['ready'] = False
    if mutation == 'status': proposal['status'] = 'data_validated'
    if mutation == 'validated': proposal['data_validated'] = True
    if mutation == 'questions': proposal['questions'] = [{'id': 'other', 'question': 'Invented question'}]
    if mutation == 'unresolved': proposal['unresolved'] = ['other']
    if mutation == 'pending': proposal['recipe']['pending_decisions'] = [{'id': 'other', 'question': 'Invented question'}]
    if mutation == 'answers': proposal['answers']['key'] = ['qc']
    if mutation == 'recorded': proposal['recipe']['research_decisions']['observation'] = None
    if mutation == 'recipe': proposal['recipe']['steps'][-1]['reason'] = 'Different study rule'
    with pytest.raises(WrangleError) as caught:
        publish(proposal, tmp_path/'new-protocol')
    assert caught.value.code == 'INVALID_RECIPE'
    assert not (tmp_path/'new-protocol').exists()
    assert not list(tmp_path.glob('*.wrangle-lock'))


def test_publishing_pending_draft_cannot_claim_all_decisions_complete(study, tmp_path):
    proposal = draft(study, input='measurements')
    proposal.update(ready=True, status='decisions_complete', questions=[], unresolved=[])
    proposal['recipe']['pending_decisions'] = []
    with pytest.raises(WrangleError) as caught:
        publish(proposal, tmp_path/'new-protocol')
    assert caught.value.code == 'INVALID_RECIPE'


def test_further_declared_checks_and_unit_conversion_preserve_original_decisions(study, answers):
    proposal = draft(study, answers)
    recipe = proposal['recipe']
    recipe['steps'].append({'op': 'convert_unit', 'column': 'mass_mg', 'from_unit': 'mg', 'to_unit': 'g', 'factor': .001})
    recipe['checks']['ranges'] = {'mass_mg': {'min': 0, 'max': 5}}
    validate_decisions(recipe)
    result = wr.prepare(proposal['sources'], recipe)
    assert result.data['mass_mg'].to_list() == [1.0, 3.0]
    assert result.receipt['units']['mass_mg'] == 'g'
    assert result.receipt['recipe']['research_decisions']['measurements']['mass_mg']['unit'] == 'mg'


def test_metadata_qc_question_uses_approved_join_output_and_native_type(study, answers, tmp_path):
    metadata = tmp_path/'metadata.parquet'
    pl.DataFrame({'sample_id': ['003', '001', '002'], 'qc_score': [1, 1, 0]}, schema={'sample_id': pl.String, 'qc_score': pl.Int16}).write_parquet(metadata)
    study['metadata'] = metadata
    del answers['exclusions']
    proposal = draft(study, answers)
    assert proposal['unresolved'] == ['exclusions']
    question = proposal['questions'][0]
    assert question['columns'] == ['sample_id', 'mass_mg', 'qc', 'qc_score']
    assert question['schema']['qc_score'] == 'Int16'
    assert question['schema']['mass_mg'] == 'Float64'
    assert proposal['output_schema'] == question['schema']
    saved = publish(proposal, tmp_path/'pending')
    assert saved['ready'] is False
    answers['exclusions'] = {'action': 'keep_values', 'column': 'qc_score', 'values': [1], 'reason': 'Supplied metadata QC rule', 'nulls': 'error'}
    resolved = draft(study, answers)
    result = wr.prepare(resolved['sources'], resolved['recipe'])
    assert result.data['sample_id'].to_list() == ['001', '003']
    assert result.data['qc_score'].dtype == pl.Int16
    assert result.receipt['steps'][-1]['reason'] == 'Supplied metadata QC rule'


def test_metadata_qc_suffix_is_observed_from_explicit_overlap_choice(study, answers):
    study['metadata'].write_text('sample_id,qc\n001,approved\n002,rejected\n003,approved\n')
    answers['matching']['overlap'] = 'suffix'
    answers['matching']['suffix'] = '_meta'
    del answers['exclusions']
    proposal = draft(study, answers)
    assert 'qc_meta' in proposal['questions'][0]['columns']
    assert proposal['questions'][0]['schema']['qc_meta'] == 'String'
    answers['exclusions'] = {'action': 'keep_values', 'column': 'qc_meta', 'values': ['approved'], 'reason': 'Metadata QC decision', 'nulls': 'error'}
    resolved = draft(study, answers)
    assert wr.prepare(resolved['sources'], resolved['recipe']).data['sample_id'].to_list() == ['001', '003']


def test_unanswered_matching_precedes_exclusions_without_inferred_attachment(study):
    proposal = draft(study, input='measurements')
    identifiers = [item['id'] for item in proposal['questions']]
    assert identifiers.index('matching') < identifiers.index('exclusions')
    exclusion = next(item for item in proposal['questions'] if item['id'] == 'exclusions')
    assert 'group' not in exclusion['columns']
    assert proposal['recipe']['steps'] == []


def test_publishing_rejects_altered_output_schema(study, answers, tmp_path):
    proposal = draft(study, answers)
    proposal['output_schema']['mass_mg'] = 'String'
    with pytest.raises(WrangleError) as caught:
        publish(proposal, tmp_path/'protocol')
    assert caught.value.code == 'INVALID_RECIPE'
    assert not (tmp_path/'protocol').exists()
