"""Verified YAML walkthrough: one protocol for researchers, notebooks and agents.

The three-row fixture and its scientific decisions are supplied in docs/starter/.
This example verifies files, expected observations, evidence, presentation parity,
and useful recovery from missing metadata. No scientific decision is inferred.
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile

import polars as pl
import wrangle as wr
from wrangle._cli import main


def _command(arguments):
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        status = main(arguments)
    return status, stdout.getvalue(), stderr.getvalue()


def _prepare_arguments(directory, metadata="metadata.csv"):
    return [
        "prepare", str(directory / "recipe.yaml"),
        "--source", f"measurements={directory / 'samples.csv'}",
        "--source", f"metadata={directory / metadata}",
    ]


def run(output=None):
    with tempfile.TemporaryDirectory(prefix="wrangle-human-example-") as temporary:
        workspace = Path(temporary)
        directory = workspace / "my-study"
        status, human_example, errors = _command(["example", str(directory)])
        assert status == 0 and not errors, errors
        assert human_example.strip() and not human_example.lstrip().startswith("{")
        starter = Path(wr.__file__).parent / "docs" / "starter"
        for name in ("samples.csv", "metadata.csv", "recipe.yaml", "README.md"):
            assert (directory / name).read_bytes() == (starter / name).read_bytes()

        status, machine_example, errors = _command(["example", str(workspace / "agent-study"), "--json"])
        assert status == 0 and not errors, errors
        assert isinstance(json.loads(machine_example), dict)

        profile = wr.inspect(directory / "samples.csv")
        assert profile["rows"] == 3
        assert profile["examples"][0]["sample_id"] == "001"
        status, human_profile, errors = _command(["inspect", str(directory / "samples.csv")])
        assert status == 0 and not errors, errors
        assert human_profile.rstrip() == profile.summary().rstrip()
        status, machine_profile, errors = _command(["inspect", str(directory / "samples.csv"), "--json"])
        assert status == 0 and not errors, errors
        assert json.loads(machine_profile) == profile

        sources = {"measurements": directory / "samples.csv", "metadata": directory / "metadata.csv"}
        result = wr.prepare(sources, directory / "recipe.yaml")
        expected = pl.DataFrame({
            "sample_id": ["001", "003"], "mass_g": [1.0, 3.0],
            "qc": ["pass", "pass"], "group": ["control", "treated"],
        })
        assert result.data.equals(expected)
        assert result.receipt["steps"][2]["excluded_keys"] == [{"sample_id": "002"}]
        assert result.receipt["units"] == {"mass_g": "g"}
        assert result.receipt["units_unresolved"] == []
        assert result.receipt["descriptions_unresolved"] == []
        assert result.summary().strip()

        variant = workspace / "same-protocol.yml"
        text = (directory / "recipe.yaml").read_text(encoding="utf-8")
        variant.write_text("# Same scientific protocol; different presentation.\n\n" + text.replace("key: sample_id", 'key: "sample_id"'), encoding="utf-8")
        replay = wr.prepare(sources, variant)
        assert replay.receipt["recipe_sha256"] == result.receipt["recipe_sha256"]
        assert replay.receipt["output"]["sha256"] == result.receipt["output"]["sha256"]

        status, human_preparation, errors = _command(_prepare_arguments(directory))
        assert status == 0 and not errors, errors
        assert human_preparation.startswith(result.summary() + "\n")
        assert "Not saved. Add --output NEW_DIRECTORY" in human_preparation
        status, machine_preparation, errors = _command(_prepare_arguments(directory) + ["--json"])
        assert status == 0 and not errors, errors
        assert json.loads(machine_preparation) == result.receipt

        destination = Path(output) if output is not None else workspace / "prepared"
        saved = wr.prepare(sources, directory / "recipe.yaml", output=destination)
        assert pl.read_parquet(destination / "data.parquet").equals(expected)
        assert (destination / "report.txt").read_text(encoding="utf-8").rstrip() == saved.summary().rstrip()
        assert (destination / "recipe.yaml").is_file()
        assert not (destination / "recipe.json").exists()
        assert json.loads((destination / "receipt.json").read_text(encoding="utf-8")) == saved.receipt
        assert wr.prepare(destination, {}).data.equals(expected)
        saved_profile = wr.inspect(destination)
        assert saved_profile["rows"] == 2 and saved_profile["columns"]["mass_g"] == "Float64"
        status, saved_view, errors = _command(["inspect", str(destination)])
        assert status == 0 and not errors, errors
        assert saved_view.rstrip() == saved_profile.summary().rstrip()
        status, saved_machine, errors = _command(["inspect", str(destination), "--json"])
        assert status == 0 and not errors, errors
        assert json.loads(saved_machine) == saved_profile

        # Missing metadata is a data/protocol decision, never permission to drop a sample.
        (directory / "incomplete-metadata.csv").write_text("sample_id,group\n001,control\n002,control\n", encoding="utf-8")
        failed_output = workspace / "failed-preparation"
        invalid = _prepare_arguments(directory, "incomplete-metadata.csv") + ["--output", str(failed_output)]
        status, stdout, human_error = _command(invalid)
        assert status == 1 and not stdout
        assert "UNMATCHED_KEYS" in human_error and not human_error.lstrip().startswith("{")
        status, stdout, machine_error = _command(invalid + ["--json"])
        assert status == 1 and not stdout
        assert json.loads(machine_error)["code"] == "UNMATCHED_KEYS"
        assert not failed_output.exists()
        assert wr.inspect(directory / "samples.csv")["rows"] == 3
        return result


if __name__ == "__main__":
    result = run()
    print(result.summary())
    print("Verified: YAML protocol, 2 samples, 1 declared exclusion, shared human/agent evidence and recovery.")
