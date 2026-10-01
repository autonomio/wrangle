"""Synthetic first-study protocol: observed files, explicit answers, checked reuse.

These answers belong only to the supplied three-row fixture. This verifies the
software workflow; it is not a usability trial with scientific researchers.
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile

import polars as pl
import wrangle as wr
from wrangle._cli import main
from wrangle._core import WrangleError
from wrangle._protocol import dump_recipe, load_recipe
from wrangle._start import draft


def _command(arguments):
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        status = main(arguments)
    return status, stdout.getvalue(), stderr.getvalue()


def supplied_answers():
    """Research choices supplied for this synthetic example, never inferred."""
    return {
        "input": "measurements", "observation": "One specimen per row",
        "key": ["sample_id"],
        "measurements": {"mass_mg": {"dtype": "Float64", "unit": "mg"}},
        "missing": {"codes": {}, "action": "keep"},
        "exclusions": {"action": "keep_values", "column": "qc", "values": ["pass"],
                       "reason": "Predeclared instrument QC acceptance", "nulls": "error"},
        "matching": {"action": "attach", "source": "metadata", "left_on": ["sample_id"],
                     "right_on": ["sample_id"], "cardinality": "1:1", "unmatched": "error",
                     "unused": "drop", "overlap": "error", "nulls": "error"},
    }


def run(output=None):
    with tempfile.TemporaryDirectory(prefix="wrangle-guided-study-") as temporary:
        workspace = Path(temporary)
        starter = Path(wr.__file__).parent / "docs" / "starter"
        sources = {"measurements": str(starter / "samples.csv"), "metadata": str(starter / "metadata.csv")}
        bindings = ["--source", "measurements=" + sources["measurements"],
                    "--source", "metadata=" + sources["metadata"]]
        arguments = ["start", sources["measurements"], "--metadata", sources["metadata"]]
        pending_directory = workspace / "pending"
        status, text, errors = _command(arguments + ["--output", str(pending_directory), "--json"])
        assert status == 0 and not errors, errors
        pending = json.loads(text)
        assert not pending["ready"] and pending["data_validated"] is False
        assert {"observation", "key", "measurements", "missing", "exclusions", "matching"} == set(pending["unresolved"])
        assert pending["observations"]["measurements"]["examples"][0]["sample_id"] == "001"
        assert set(path.name for path in pending_directory.iterdir()) == {"recipe.yaml", "answers.yaml", "README.md"}
        failed = workspace / "unanswered-result"
        status, text, errors = _command(["prepare", pending["recipe_path"], *bindings, "--output", str(failed), "--json"])
        assert status == 1 and not text
        assert json.loads(errors)["code"] == "UNRESOLVED_PROTOCOL" and not failed.exists()

        # The researcher supplies the answers; the same command serves human and agent.
        answers_path = Path(pending["answers_path"])
        answers_path.write_text(dump_recipe(supplied_answers()), encoding="utf-8")
        resolved_directory = workspace / "resolved"
        status, human, errors = _command(arguments + ["--answers", str(answers_path), "--output", str(resolved_directory)])
        assert status == 0 and not errors, errors
        assert "Protocol ready to prepare" in human
        status, machine, errors = _command(arguments + ["--answers", str(answers_path), "--json"])
        assert status == 0 and not errors, errors
        resolved = json.loads(machine)
        assert resolved["ready"] and resolved["data_validated"] is False and not resolved["questions"]
        recipe_path = resolved_directory / "recipe.yaml"
        assert load_recipe(recipe_path) == resolved["recipe"]
        assert draft(sources, supplied_answers())["recipe"] == resolved["recipe"]
        readme = (resolved_directory / "README.md").read_text(encoding="utf-8")
        assert "wrangle prepare" in readme and "For the next batch" in readme

        expected = pl.DataFrame({"sample_id": ["001", "003"], "mass_mg": [1000.0, 3000.0],
                                 "qc": ["pass", "pass"], "group": ["control", "treated"]})
        result = wr.prepare(sources, recipe_path)
        assert result.data.equals(expected)
        assert result.receipt["units"] == {"mass_mg": "mg"}
        assert result.receipt["steps"][-1]["excluded_keys"] == [{"sample_id": "002"}]
        assert "One specimen per row" in result.summary()
        status, human, errors = _command(["prepare", str(recipe_path), *bindings])
        assert status == 0 and not errors and human.startswith(result.summary()), errors
        status, machine, errors = _command(["prepare", str(recipe_path), *bindings, "--json"])
        assert status == 0 and not errors and json.loads(machine) == result.receipt, errors

        destination = Path(output) if output is not None else workspace / "prepared"
        saved = wr.prepare(sources, recipe_path, output=destination)
        assert pl.read_parquet(destination / "data.parquet").equals(expected)
        assert (destination / "report.txt").read_text(encoding="utf-8").rstrip() == saved.summary().rstrip()
        assert wr.prepare(destination, {}).data.equals(expected)
        disk = wr.prepare(sources, recipe_path, execution="disk", output=workspace / "prepared-disk")
        assert disk.data.collect().equals(expected)

        # Reuse the same protocol with different files, without revisiting its choices.
        next_sources = {}
        for name, path in sources.items():
            next_path = workspace / ("next-" + Path(path).name)
            next_path.write_bytes(Path(path).read_bytes())
            next_sources[name] = str(next_path)
        assert wr.prepare(next_sources, recipe_path).data.equals(expected)
        incomplete = workspace / "incomplete-metadata.csv"
        incomplete.write_text("sample_id,group\n001,control\n002,control\n", encoding="utf-8")
        bad_sources = {**next_sources, "metadata": str(incomplete)}
        bad_output = workspace / "failed-matching"
        try:
            wr.prepare(bad_sources, recipe_path, output=bad_output)
        except WrangleError as error:
            assert error.code == "UNMATCHED_KEYS"
        else:
            raise AssertionError("Missing metadata must fail the supplied matching contract")
        assert not bad_output.exists()
        return result


if __name__ == "__main__":
    result = run()
    print(result.summary())
    print("Verified: explicit YAML decisions, blocked draft, human/agent parity, checked preparation and next-batch reuse.")
