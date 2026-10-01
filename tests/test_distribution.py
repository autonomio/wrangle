"""Installed-agent documentation and native execution remain discoverable."""
import ast
import importlib.metadata
import json
from pathlib import Path
import re
import runpy

import wrangle
from wrangle._catalog import catalog, resolve

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "wrangle"


def test_catalog_matches_execution_definitions_and_points_to_existing_sources():
    documented = json.loads((PACKAGE / "docs" / "operations.json").read_text())
    assert documented == catalog()
    for entry in documented["operations"]:
        assert entry["description"]
        assert "\\" not in entry["source"]
        assert (PACKAGE / entry["source"]).is_file()
        assert callable(resolve(entry["name"]))


def test_all_legacy_preparation_entrypoints_remain_discoverable():
    for group in ("array", "col", "df", "dic"):
        tree = ast.parse((PACKAGE / group / "__init__.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert callable(getattr(wrangle, alias.asname or alias.name))


def test_no_legacy_numeric_engines_or_python_udfs_in_the_package():
    forbidden = {"numpy", "pandas", "scipy", "statsmodels", "sklearn", "tensorflow", "keras"}
    for source in PACKAGE.rglob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not ({item.name.split('.')[0] for item in node.names} & forbidden), source
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split('.')[0] not in forbidden, source
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in {"eval", "exec"}, source
                if isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in {"map_elements", "map_batches", "map_groups"}, source


def test_readme_example_is_executable(tmp_path, monkeypatch):
    from wrangle._cli import main
    study = tmp_path / "my-study"
    assert main(["example", str(study)]) == 0
    monkeypatch.chdir(study)
    blocks = re.findall(r"```python\n(.*?)\n```", (ROOT / "README.md").read_text(), re.S)
    assert blocks
    for block in blocks:
        exec(compile(block, "README.md", "exec"), {})


def test_shipped_research_workflow_checks_expected_results(tmp_path):
    namespace = runpy.run_path(str(PACKAGE / "docs" / "research_batch.py"))
    result = namespace["run"](tmp_path / "batch")
    assert result.data.height == 2
    assert (tmp_path / "batch" / "receipt.json").is_file()


def test_package_and_receipt_versions_agree():
    from wrangle._api import VERSION
    assert wrangle.__version__ == VERSION == importlib.metadata.version("wrangle")
