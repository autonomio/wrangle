"""Agent navigation and recipe eligibility are derived from shipped definitions."""
import json
from pathlib import Path
import re

import wrangle
from wrangle._catalog import catalog, operation_document, recipe_eligibility, resolve

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(wrangle.__file__).parent


def test_catalog_identifies_outputs_and_pipeline_use_before_execution():
    document = catalog()
    entries = {entry["name"]: entry for entry in document["operations"]}
    assert entries["array_to_generator"]["returns"] == ["generator"]
    assert entries["create_datetime_col"]["returns"] == ["series"]
    assert entries["df_corr_pearson"]["returns"] == ["dictionary", "tuple"]
    assert set(entries["array_random_shuffle"]["returns"]) == {"series", "table", "sequence", "tuple"}
    assert entries["df_to_multiclass"]["recipe"]["eligibility"] == "conditional"
    for name in ("network_check", "is_connected", "read_large_csv", "groupby_func", "X_data", "df_to_xy", "dic_corr_perc", "create_synth_binary_model"):
        assert recipe_eligibility(resolve(name)) == "never"
        assert entries[name]["recipe"]["eligibility"] == "never"
    for name in ("df_impute_nan", "df_to_groupby", "col_resample_interval"):
        assert recipe_eligibility(resolve(name)) == "yes"
    assert all(entry["returns"] != ["none"] or entry["retired"] for entry in entries.values())


def test_catalog_points_to_the_actual_definition_and_failure_rules():
    document = catalog()
    boundary_codes = {item["code"] for item in document["boundary_validation"]}
    assert {"INVALID_INPUT", "UNKNOWN_COLUMN", "IMMUTABLE_INPUT", "UNSUPPORTED_CALLBACK"} <= boundary_codes
    for entry in document["operations"]:
        lines = (PACKAGE / entry["source"]).read_text().splitlines()
        function_name = entry["symbol"].rsplit(".", 1)[1]
        assert lines[entry["line"] - 1].startswith(f"def {function_name}(")
        assert all(item["code"] and item["precondition"] for item in entry["validation"])
    weighted = next(entry for entry in document["operations"] if entry["name"] == "array_random_weighted")
    assert "EXPLICIT_WEIGHTS_REQUIRED" in {item["code"] for item in weighted["validation"]}
    assert "rerun the same protocol" in document["recovery"]


def test_installed_manual_links_resolve_in_the_distribution_layout():
    manual = PACKAGE / "docs" / "README.md"
    text = manual.read_text()
    assert "wrangle/docs/" not in text
    assert "recipe.eligibility" in text
    assert "individual-record key" in text
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        destination = (manual.parent / target.partition("#")[0]).resolve()
        if destination == PACKAGE / "AGENTS.md":
            # The source entrypoint is force-included into the wheel at this path.
            assert (ROOT / "AGENTS.md").is_file()
        else:
            assert destination.is_file(), target
    for relative in ["docs/README.md", "docs/recipes.md", "docs/migration.md", "docs/research_batch.py"]:
        assert (PACKAGE / relative).is_file()


def test_catalog_navigation_paths_are_package_relative():
    navigation = json.loads((PACKAGE / "docs" / "operations.json").read_text())["navigation"]
    assert navigation["paths_relative_to"] == "Path(wrangle.__file__).parent"
    for key in ("recipe_grammar", "migration", "executable_workflow"):
        assert (PACKAGE / navigation[key]).is_file()


def test_each_operation_has_a_small_definition_derived_document():
    document = catalog()
    for entry in document["operations"]:
        path = PACKAGE / entry["manual"]
        actual = json.loads(path.read_text())
        assert actual == operation_document(entry["name"], document)
        assert [operation["name"] for operation in actual["operations"]] == [entry["name"]]
        assert actual["shared_validation"]["$ref"].startswith("docs/operations.json#")


def test_guided_question_catalog_uses_the_engine_definitions():
    from wrangle._start import QUESTION_DEFINITIONS
    guide = catalog()["guided_start"]
    assert guide["questions"] == list(QUESTION_DEFINITIONS)
    assert guide["symbols"]["draft"] == "wrangle._start.draft"
    assert (PACKAGE / guide["workflow"]).is_file()
    codes = {item["code"] for item in catalog()["shared_validation"]}
    assert {"UNRESOLVED_PROTOCOL", "START_ANSWER", "START_SCOPE"} <= codes
    assert all(re.fullmatch(r"[A-Z][A-Z0-9_]*", code) for code in codes)
    answer_rules = [item["precondition"] for item in catalog()["shared_validation"] if item["code"] == "START_ANSWER"]
    assert any("Choose columns present in the observed file" in rule for rule in answer_rules)
    assert "message" not in answer_rules


def test_shipped_guided_workflow_matches_checkout_and_executes():
    import runpy
    package_workflow = PACKAGE / "docs" / "start_workflow.py"
    assert package_workflow.read_bytes() == (ROOT / "examples" / "start_workflow.py").read_bytes()
    result = runpy.run_path(str(package_workflow))["run"]()
    assert result.data["sample_id"].to_list() == ["001", "003"]
