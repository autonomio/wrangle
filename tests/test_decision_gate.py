"""Unanswered guided protocols cannot read data, execute, or publish evidence."""
import pytest
import wrangle


@pytest.mark.parametrize("execution", ["memory", "disk"])
def test_pending_research_decisions_block_before_source_access(tmp_path, monkeypatch, execution):
    import wrangle._api as api
    accessed = []
    def forbidden(*args, **kwargs):
        accessed.append(True)
        raise AssertionError("unresolved recipe read a source")
    monkeypatch.setattr(api, "_source", forbidden)
    pending = [{"id": "observation", "question": "What does one row represent?"}, {"id": "key", "question": "Which fields identify one observation?"}]
    output = tmp_path / "failed"
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(tmp_path / "absent.csv", {"version": 1, "pending_decisions": pending}, execution=execution, output=output)
    assert caught.value.code == "UNRESOLVED_PROTOCOL"
    assert caught.value.details["decisions"] == pending
    assert not accessed and not output.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("pending", [None, False, {}, "unanswered", [None], [{"id": "key"}], [{"id": "", "question": "Choose a key"}], [{"id": "key", "question": 3}], [{"id": "key", "question": "Choose a key"}, {"id": "key", "question": "Other choice"}]])
def test_malformed_pending_questions_are_rejected_as_recipe_errors(pending):
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare("absent.csv", {"pending_decisions": pending})
    assert caught.value.code == "INVALID_RECIPE"


def test_empty_pending_list_keeps_ordinary_manual_recipe_available():
    import polars as pl
    result = wrangle.prepare(pl.DataFrame({"id": ["001"]}), {"key": ["id"], "pending_decisions": []})
    assert result.data["id"].to_list() == ["001"]
