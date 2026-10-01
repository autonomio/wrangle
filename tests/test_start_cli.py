"""A researcher and an agent author the same explicitly gated protocol."""
from io import StringIO
import json
import shlex
from pathlib import Path

import pytest

import wrangle
from wrangle._cli import _answer, _values, main
from wrangle._presentation import render_error, render_start
from wrangle._protocol import dump_recipe, load_recipe


@pytest.fixture
def own_files(tmp_path):
    measurements = tmp_path / "real samples.csv"
    measurements.write_text("sample_id,mass_mg,qc\n001,1000,pass\n002,2000,fail\n", encoding="utf-8")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("sample_id,group\n001,control\n002,treated\n", encoding="utf-8")
    return measurements, metadata


def _decisions():
    return {
        "observation": "One sample per row",
        "key": ["sample_id"],
        "measurements": {"mass_mg": {"dtype": "Float64", "unit": "mg"}},
        "missing": {"codes": {}, "action": "keep"},
        "exclusions": {"action": "none"},
    }


@pytest.mark.parametrize("prefix,suffix", [(["--json"], []), ([], ["--json"])])
def test_json_start_never_prompts_and_preserves_the_shared_draft(own_files, capsys, monkeypatch, prefix, suffix):
    from wrangle._start import draft
    source, _ = own_files
    class NoInput:
        def isatty(self):
            return True
        def readline(self):
            raise AssertionError("An agent must never be prompted")
    monkeypatch.setattr("sys.stdin", NoInput())
    assert main([*prefix, "start", str(source), *suffix]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    proposal = json.loads(captured.out)
    assert proposal == draft({"measurements": str(source)}, input="measurements")
    assert proposal["ready"] is False
    assert proposal["recipe"]["pending_decisions"]


def test_redirected_human_start_never_reads_stdin_and_explains_pending_decisions(own_files, capsys, monkeypatch):
    source, _ = own_files
    class NoInput:
        def isatty(self):
            return False
        def readline(self):
            raise AssertionError("Redirected input must not block")
    monkeypatch.setattr("sys.stdin", NoInput())
    assert main(["start", str(source)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Protocol draft: researcher decisions still needed" in captured.out
    assert "Preparation is blocked" in captured.out
    assert "No dataset has been prepared or validated" in captured.out
    assert "Rows" not in captured.out  # no claim of executed row transitions
    assert "sample_id" in captured.out
    assert set(source.parent.iterdir()) == set(own_files)


@pytest.mark.parametrize("arguments", [[], ["file.csv", "--source", "data=file.csv"], ["--metadata", "meta.csv"], ["file.csv", "--input", "other"], ["file.csv", "--interactive", "--json"]])
def test_start_rejects_ambiguous_invocations_without_reading_files(arguments, capsys):
    assert main(["start", *arguments, "--json"]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["code"] == "INVALID_INVOCATION"


def test_accepted_answers_publish_only_protocol_and_are_ready_for_same_prepare_engine(own_files, tmp_path, capsys):
    source, _ = own_files
    answers = tmp_path / "decisions.yaml"
    answers.write_text(dump_recipe(_decisions()), encoding="utf-8")
    output = tmp_path / "first protocol"
    assert main(["start", str(source), "--answers", str(answers), "--output", str(output), "--json"]) == 0
    proposal = json.loads(capsys.readouterr().out)
    assert proposal["ready"] is True
    assert set(path.name for path in output.iterdir()) == {"recipe.yaml", "answers.yaml", "README.md"}
    assert "data.parquet" not in proposal.get("files", [])
    recipe = load_recipe(output / "recipe.yaml")
    assert not recipe.get("pending_decisions")
    prepared = wrangle.prepare({"measurements": source}, recipe)
    assert prepared.data["sample_id"].to_list() == ["001", "002"]
    assert prepared.data["mass_mg"].to_list() == [1000.0, 2000.0]
    assert prepared.receipt["units"]["mass_mg"] == "mg"
    assert "One row represents: One sample per row" in prepared.summary()


def test_pending_start_can_be_saved_but_prepare_fails_closed(own_files, tmp_path, capsys):
    source, _ = own_files
    output = tmp_path / "unfinished"
    assert main(["start", str(source), "--output", str(output)]) == 0
    text = capsys.readouterr().out
    assert "Protocol draft" in text
    assert "Saved:" in text
    assert "answers.yaml" in text
    result = tmp_path / "must-not-exist"
    assert main(["prepare", str(output / "recipe.yaml"), "--source", f"measurements={source}", "--output", str(result), "--json"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["code"] == "UNRESOLVED_PROTOCOL"
    assert not result.exists()


def test_interactive_start_translates_plain_answers_to_same_schema(own_files, monkeypatch, capsys):
    source, _ = own_files
    answers = "One sample per row\nsample_id\nmass_mg\n1\nmg\nno\n1\n1\n"
    monkeypatch.setattr("sys.stdin", StringIO(answers))
    assert main(["start", str(source), "--interactive"]) == 0
    captured = capsys.readouterr()
    assert "What does one row represent" in captured.err
    assert "decimal numbers (approximate)" in captured.err
    assert "whole numbers (exact)" in captured.err
    assert "What unit" in captured.err
    assert "keep observed numeric type" not in captured.err  # CSV measurements are observed as text
    assert "Protocol ready to prepare" in captured.out
    assert "No dataset has been prepared or validated" in captured.out
    assert "pending_decisions: []" in captured.out
    assert "Float64" in captured.out


def test_interactive_eof_keeps_unresolved_decisions_and_never_prepares(own_files, monkeypatch, capsys):
    source, _ = own_files
    monkeypatch.setattr("sys.stdin", StringIO("One sample per row\n"))
    assert main(["start", str(source), "--interactive"]) == 0
    captured = capsys.readouterr()
    assert "Input ended" in captured.err
    assert "Protocol draft" in captured.out
    assert "pending_decisions" in captured.out
    assert "One sample per row" in captured.out
    assert "Protocol ready" not in captured.out


def test_interactive_uncertainty_is_never_recorded_as_a_scientific_answer(own_files, monkeypatch, capsys):
    source, _ = own_files
    monkeypatch.setattr("sys.stdin", StringIO("not sure\n\nnone\nno\n1\n1\n"))
    assert main(["start", str(source), "--interactive"]) == 0
    captured = capsys.readouterr()
    assert "Protocol draft" in captured.out
    assert "observation:" in captured.out
    assert "key:" in captured.out
    assert "not sure\n" not in captured.out


def test_keyboard_cancellation_never_publishes_partial_protocol(own_files, tmp_path, monkeypatch, capsys):
    source, _ = own_files
    class CancelledInput:
        def readline(self):
            raise KeyboardInterrupt
    monkeypatch.setattr("sys.stdin", CancelledInput())
    output = tmp_path / "cancelled"
    assert main(["start", str(source), "--interactive", "--output", str(output)]) == 130
    captured = capsys.readouterr()
    assert not captured.out
    assert "START_CANCELLED" in captured.err
    assert not output.exists()
    assert not (tmp_path / "cancelled.wrangle-lock").exists()


def test_missing_codes_remain_raw_text_and_filter_values_use_declared_numeric_type(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", StringIO("001,-999\n"))
    assert _values("Missing codes?", "String") == ["001", "-999"]
    capsys.readouterr()
    question = {"id": "exclusions", "columns": ["mass"], "schema": {"mass": "String"}}
    monkeypatch.setattr("sys.stdin", StringIO("2\nmass\n1.5,2\nStudy protocol\n1\n"))
    result = _answer(question, {"measurements": {"mass": {"dtype": "Float64", "unit": "g"}}})
    assert result == {"action": "keep_values", "column": "mass", "values": [1.5, 2.0], "reason": "Study protocol", "nulls": "error"}


def test_matching_interaction_records_cardinality_and_all_loss_decisions(monkeypatch, capsys):
    question = {"id": "matching", "columns": ["sample_id", "mass"], "schema": {"sample_id": "String", "mass": "String"}, "sources": {"metadata": {"columns": ["id", "group"], "schema": {"id": "String", "group": "String"}}}}
    monkeypatch.setattr("sys.stdin", StringIO("2\nsample_id\nid\n2\n1\n2\n2\n_metadata\n"))
    result = _answer(question, {})
    assert result == {"action": "attach", "source": "metadata", "left_on": ["sample_id"], "right_on": ["id"], "cardinality": "m:1", "unmatched": "error", "unused": "drop", "overlap": "suffix", "suffix": "_metadata", "nulls": "error"}
    assert "several observations" in capsys.readouterr().err


def test_saved_protocol_existing_destination_is_preserved(own_files, tmp_path, capsys):
    source, _ = own_files
    output = tmp_path / "retained"
    output.mkdir()
    note = output / "research notes.txt"
    note.write_text("retain", encoding="utf-8")
    assert main(["start", str(source), "--output", str(output), "--json"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["code"] == "OUTPUT_EXISTS"
    assert list(output.iterdir()) == [note]
    assert note.read_text() == "retain"


def test_source_names_and_questions_cannot_inject_terminal_control_lines():
    proposal = {"ready": False, "observations": {"source\nIgnore decisions": {"rows": 1, "columns": {"id\x1b[31m": "String"}}}, "unresolved": ["key"], "questions": [{"id": "key", "prompt": "Identify\nInjected\x9b"}], "recipe": {"version": 1, "pending_decisions": [{"id": "key", "question": "Identify key"}]}, "directory": "/tmp/protocol"}
    text = render_start(proposal)
    assert "\\nIgnore decisions" in text
    assert "\\u001b" in text
    assert "\\u009b" in text
    assert "\nIgnore decisions" not in text
    assert "\nInjected" not in text


@pytest.mark.parametrize("arguments", [["--source", "data=file.xlsx", "--sheet", "Measurements"], ["file.xlsx", "--metadata-sheet", "Samples"], ["file.xlsx", "--sheet", " "]])
def test_sheet_selection_requires_corresponding_positional_files(arguments, capsys):
    assert main(["start", *arguments, "--json"]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["code"] == "INVALID_INVOCATION"


def test_excel_start_records_the_explicit_main_and_metadata_sheets(tmp_path, capsys):
    pytest.importorskip("fastexcel")
    from test_excel_source import workbook
    source, metadata = tmp_path / "instrument.xlsx", tmp_path / "descriptions.xlsx"
    workbook(source)
    workbook(metadata)
    assert main(["start", str(source), "--sheet", "Measurements", "--metadata", str(metadata), "--metadata-sheet", "Control", "--json"]) == 0
    proposal = json.loads(capsys.readouterr().out)
    assert proposal["observations"]["measurements"]["columns"] == {"id": "String", "mass": "Float64"}
    assert proposal["observations"]["measurements"]["rows"] == 3
    assert proposal["observations"]["metadata"]["columns"] == {"different": "Int64"}
    assert proposal["recipe"]["source_options"]["measurements"]["options"] == {"sheet_name": "Measurements"}
    assert proposal["recipe"]["source_options"]["metadata"]["options"] == {"sheet_name": "Control"}


def test_interactive_named_sources_choose_main_file_before_column_decisions(own_files, monkeypatch, capsys):
    source, metadata = own_files
    inputs = "1\nOne sample per row\nsample_id\nmass_mg\n1\nmg\nno\n1\n1\n1\n"
    monkeypatch.setattr("sys.stdin", StringIO(inputs))
    assert main(["start", "--source", f"samples={source}", "--source", f"details={metadata}", "--interactive"]) == 0
    captured = capsys.readouterr()
    assert "Which file contains the observations" in captured.err
    assert "sample_id" in captured.err
    assert "Protocol ready to prepare" in captured.out
    assert "input: samples" in captured.out


def test_interactive_unresolved_main_file_does_not_ask_dependent_column_questions(own_files, monkeypatch, capsys):
    source, metadata = own_files
    monkeypatch.setattr("sys.stdin", StringIO("not sure\n"))
    assert main(["start", "--source", f"samples={source}", "--source", f"details={metadata}", "--interactive"]) == 0
    captured = capsys.readouterr()
    assert "Protocol draft" in captured.out
    assert "Which column or columns" not in captured.err


def test_interactive_literal_codes_preserve_quotes_spaces_and_reject_unclosed_quotes(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", StringIO('"unterminated\n"001"," NA","a,b",""\n'))
    assert _values("Exact missing codes?", "String") == ["001", " NA", "a,b", ""]
    assert "Enter values that match this column" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["unknown", "skip", "not sure", "?", "don't know"])
def test_exact_literal_prompts_never_infer_that_dataset_values_are_uncertain(value, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", StringIO(value + "\n"))
    assert _values("Exact missing code?", "String") == [value]
    assert "Every nonempty entry is a data value" in capsys.readouterr().err


def test_exact_literal_prompt_enter_retains_unresolved_decision(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", StringIO("\n"))
    assert _values("Exact missing code?", "String") is None
    capsys.readouterr()


def test_nested_error_details_cannot_execute_terminal_controls():
    value = "untrusted\x9b31m\x7f\u202eInjected"
    text = render_error(wrangle.WrangleError("START_ANSWER", "Review the declared code.", {"codes": [value]}))
    assert "\x9b" not in text
    assert "\x7f" not in text
    assert "\u202e" not in text
    assert "\\u009b" in text
    assert "\\u007f" in text
    assert "\\u202e" in text


def test_interactive_metadata_quality_rule_uses_the_declared_joined_column(own_files, tmp_path, monkeypatch, capsys):
    source, metadata = own_files
    metadata.write_text("sample_id,metadata_qc\n001,unknown\n002,fail\n", encoding="utf-8")
    inputs = "One sample per row\nsample_id\nmass_mg\n1\nmg\nno\n1\n2\nsample_id\nsample_id\n1\n1\n1\n1\n2\nmetadata_qc\nunknown\nPredeclared metadata QC category\n1\n"
    monkeypatch.setattr("sys.stdin", StringIO(inputs))
    protocol = tmp_path / "metadata protocol"
    assert main(["start", str(source), "--metadata", str(metadata), "--interactive", "--output", str(protocol)]) == 0
    captured = capsys.readouterr()
    assert "Protocol ready to prepare" in captured.out
    assert captured.err.index("Should information from the other file") < captured.err.index("Does the study have a rule")
    recipe = load_recipe(protocol / "recipe.yaml")
    assert recipe["research_decisions"]["exclusions"]["column"] == "metadata_qc"
    assert recipe["research_decisions"]["exclusions"]["values"] == ["unknown"]
    assert recipe["steps"][-1]["where"] == {"eq": [{"col": "metadata_qc"}, "unknown"]}
    result = wrangle.prepare({"measurements": source, "metadata": metadata}, recipe)
    assert result.data["sample_id"].to_list() == ["001"]
    assert result.data["metadata_qc"].to_list() == ["unknown"]


def test_interactive_metadata_quality_values_use_native_suffixed_numeric_type(own_files, tmp_path, monkeypatch, capsys):
    import polars as pl
    source, _ = own_files
    metadata = tmp_path / "numeric quality.parquet"
    pl.DataFrame({"sample_id": ["001", "002"], "mass_mg": pl.Series([1, 0], dtype=pl.Int8)}).write_parquet(metadata)
    inputs = "One sample per row\nsample_id\nmass_mg\n1\nmg\nno\n1\n2\nsample_id\nsample_id\n1\n1\n1\n2\n_qc\n2\nmass_mg_qc\n1\nDeclared binary metadata QC\n1\n"
    monkeypatch.setattr("sys.stdin", StringIO(inputs))
    protocol = tmp_path / "native metadata protocol"
    assert main(["start", str(source), "--metadata", str(metadata), "--interactive", "--output", str(protocol)]) == 0
    captured = capsys.readouterr()
    assert "Protocol ready to prepare" in captured.out
    assert "mass_mg_qc should be kept? Type: Int8." in captured.err
    recipe = load_recipe(protocol / "recipe.yaml")
    assert recipe["research_decisions"]["exclusions"]["values"] == [1]
    assert type(recipe["research_decisions"]["exclusions"]["values"][0]) is int
    result = wrangle.prepare({"measurements": source, "metadata": metadata}, recipe)
    assert result.data["sample_id"].to_list() == ["001"]
    assert result.data["mass_mg_qc"].dtype == pl.Int8


def test_published_human_next_command_preserves_complete_quoted_paths(own_files, tmp_path, capsys):
    source, _ = own_files
    answers = tmp_path / "researcher's decisions.yaml"
    answers.write_text(dump_recipe(_decisions()), encoding="utf-8")
    output = tmp_path / "researcher's first protocol"
    assert main(["start", str(source), "--answers", str(answers), "--output", str(output)]) == 0
    text = capsys.readouterr().out
    command = text.split("Next command:\n  ", 1)[1].splitlines()[0]
    assert shlex.split(command) == ["wrangle", "prepare", str(output / "recipe.yaml"), "--source", "measurements=" + str(source), "--output", str(output / "prepared")]
    assert command in (output / "README.md").read_text(encoding="utf-8")
    tokens = shlex.split(command)
    assert main(tokens[1:]) == 0
    assert "Preparation complete" in capsys.readouterr().out
    assert (output / "prepared" / "data.parquet").is_file()


def test_pending_human_next_command_requires_answers_and_preserves_quoted_paths(own_files, tmp_path, capsys):
    source, _ = own_files
    output = tmp_path / "pending protocol"
    assert main(["start", str(source), "--output", str(output)]) == 0
    text = capsys.readouterr().out
    command = text.split("Next command:\n  ", 1)[1].splitlines()[0]
    assert shlex.split(command) == ["wrangle", "start", "--source", "measurements=" + str(source), "--answers", str(output / "answers.yaml"), "--output", str(output.parent / "pending protocol-resolved")]
    assert "Complete answers.yaml" in text
    assert command in (output / "README.md").read_text(encoding="utf-8")


def test_next_command_terminal_controls_are_escaped_and_marked_for_display():
    proposal = {"ready": True, "observations": {}, "unresolved": [], "directory": "/tmp/protocol", "next_commands": ["wrangle prepare 'unsafe\x9b31m\x1bfile\nname\u202e' --source data=data.csv"]}
    text = render_start(proposal)
    assert "Next command (control characters escaped for display):" in text
    assert "\x9b" not in text and "\x1b" not in text and "\u202e" not in text
    assert "\nname" not in text
    assert "\\u009b" in text and "\\u001b" in text and "\\u202e" in text
    assert "\\nname" in text


def test_interactive_pending_answer_template_can_record_qc_while_measurements_remain_null(own_files, tmp_path, monkeypatch, capsys):
    source, _ = own_files
    pending = tmp_path / "pending answers"
    assert main(["start", str(source), "--output", str(pending)]) == 0
    capsys.readouterr()
    answers_path = pending / "answers.yaml"
    answers = load_recipe(answers_path)
    assert answers["measurements"] is None
    answers.update(observation="One sample per row", key=["sample_id"], missing={"codes": {}, "action": "keep"})
    answers_path.write_text(dump_recipe(answers), encoding="utf-8")
    monkeypatch.setattr("sys.stdin", StringIO("\n2\nqc\npass\nDeclared instrument QC\n1\n"))
    edited = tmp_path / "still pending measurements"
    assert main(["start", str(source), "--answers", str(answers_path), "--interactive", "--output", str(edited)]) == 0
    text = capsys.readouterr().out
    assert "Protocol draft" in text
    assert "Protocol ready" not in text
    recipe = load_recipe(edited / "recipe.yaml")
    assert [question["id"] for question in recipe["pending_decisions"]] == ["measurements"]
    assert recipe["research_decisions"]["measurements"] is None
    assert recipe["research_decisions"]["exclusions"]["values"] == ["pass"]
    assert recipe["steps"][-1]["where"] == {"eq": [{"col": "qc"}, "pass"]}


def test_partial_null_measurement_dtype_retains_observed_literal_type(monkeypatch, capsys):
    question = {"id": "exclusions", "columns": ["mass"], "schema": {"mass": "String"}}
    monkeypatch.setattr("sys.stdin", StringIO("2\nmass\n001\nDeclared raw instrument code\n1\n"))
    answer = _answer(question, {"measurements": {"mass": {"dtype": None, "unit": "mg"}}})
    assert answer["values"] == ["001"]
    capsys.readouterr()


def test_interactive_known_measurement_representation_is_retained_when_unit_is_unknown(own_files, tmp_path, monkeypatch, capsys):
    source, _ = own_files
    inputs = "One sample per row\nsample_id\nmass_mg\n1\n\nno\n1\n1\n"
    monkeypatch.setattr("sys.stdin", StringIO(inputs))
    output = tmp_path / "unknown measurement unit"
    assert main(["start", str(source), "--interactive", "--output", str(output)]) == 0
    assert "Protocol draft" in capsys.readouterr().out
    answers = load_recipe(output / "answers.yaml")
    assert answers["measurements"] == {"mass_mg": {"dtype": "Float64", "unit": None}}
    recipe = load_recipe(output / "recipe.yaml")
    assert [question["id"] for question in recipe["pending_decisions"]] == ["measurements"]
    with pytest.raises(wrangle.WrangleError) as error:
        wrangle.prepare({"measurements": source}, recipe)
    assert error.value.code == "UNRESOLVED_PROTOCOL"


def test_interactive_partial_measurement_template_preserves_unit_and_records_raw_qc(own_files, tmp_path, monkeypatch, capsys):
    source, _ = own_files
    answers = _decisions()
    answers["measurements"] = {"mass_mg": {"dtype": None, "unit": "mg"}}
    answers["exclusions"] = None
    answers_path = tmp_path / "partial measurements.yaml"
    answers_path.write_text(dump_recipe(answers), encoding="utf-8")
    monkeypatch.setattr("sys.stdin", StringIO("mass_mg\n\n2\nmass_mg\n1000\nApproved raw instrument QC code\n1\n"))
    output = tmp_path / "partial representation"
    assert main(["start", str(source), "--answers", str(answers_path), "--interactive", "--output", str(output)]) == 0
    assert "Protocol draft" in capsys.readouterr().out
    recipe = load_recipe(output / "recipe.yaml")
    assert recipe["research_decisions"]["measurements"] == {"mass_mg": {"dtype": None, "unit": "mg"}}
    assert recipe["research_decisions"]["exclusions"]["values"] == ["1000"]
    assert [question["id"] for question in recipe["pending_decisions"]] == ["measurements"]
    with pytest.raises(wrangle.WrangleError) as error:
        wrangle.prepare({"measurements": source}, recipe)
    assert error.value.code == "UNRESOLVED_PROTOCOL"


def test_interactive_eof_after_representation_retains_selected_measurement_decisions(own_files, tmp_path, monkeypatch, capsys):
    source, _ = own_files
    monkeypatch.setattr("sys.stdin", StringIO("One sample per row\nsample_id\nmass_mg\n1\n"))
    output = tmp_path / "interrupted measurement"
    assert main(["start", str(source), "--interactive", "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert "Input ended" in captured.err
    assert "Protocol draft" in captured.out
    answers = load_recipe(output / "answers.yaml")
    assert answers["measurements"] == {"mass_mg": {"dtype": "Float64", "unit": None}}


def test_unknown_unit_from_answers_is_asked_without_reasking_confirmed_representation(monkeypatch, capsys):
    question = {"id": "measurements", "columns": ["mass_mg"], "schema": {"mass_mg": "String"}}
    answers = {"measurements": {"mass_mg": {"dtype": "Float64", "unit": "unknown"}}}
    monkeypatch.setattr("sys.stdin", StringIO("mass_mg\nmg\n"))
    result = _answer(question, answers)
    assert result == {"mass_mg": {"dtype": "Float64", "unit": "mg"}}
    text = capsys.readouterr().err
    assert "Recorded number representation for mass_mg: decimal numbers (approximate)" in text
    assert "how should values be read" not in text
    assert "What unit does mass_mg use" in text
