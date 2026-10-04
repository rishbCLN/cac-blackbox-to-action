"""Tests for ``src/utils`` --- the three helpers every other module leans on.

Nothing here is an algorithm, but two of these functions decide what a *reader*
sees.  :func:`~src.utils.io.save_latex_table` writes the numbers the paper is
judged on, so a missing quantity has to print as an absence rather than as the
word ``NaN``; :class:`~src.utils.io.NumpyJSONEncoder` writes the metadata later
runs load, and a numpy scalar it refuses would abort a finished experiment at the
final line.  :mod:`src.utils.seeding` is global rule #2 itself: if
:func:`~src.utils.seeding.get_rng` ever handed back the global generator, two
callers asking for the same seed would silently diverge.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG
from src.utils.io import (
    NumpyJSONEncoder,
    existing,
    load_json,
    save_dataframe,
    save_json,
    save_latex_table,
    update_json,
)
from src.utils.logging import get_logger, setup_logging
from src.utils.seeding import get_rng, set_global_seed


@pytest.fixture()
def tables_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the table writer at a scratch directory.

    ``CONFIG`` is frozen, so the override is a copy installed over the writer's own
    reference to it.  Without this the test would overwrite a real table in
    ``paper/tables/`` with its own fixture data, and the damage would only show up
    in the built PDF.
    """
    out = tmp_path / "tables"
    monkeypatch.setattr("src.utils.io.CONFIG",
                        CONFIG.model_copy(update={"TABLES_DIR": out}))
    return out


def _table(tables_dir: Path, frame: pd.DataFrame, **kwargs: object) -> str:
    path = save_latex_table(frame, "fixture", caption="A caption.", **kwargs)
    assert path == tables_dir / "fixture.tex"
    return path.read_text(encoding="utf-8")


def test_a_quantity_that_does_not_exist_is_not_reported_as_a_crash(
        tables_dir: Path) -> None:
    """A missing cell prints as an en dash, not as ``NaN``.

    The mean effort of a method's usable plans is undefined when it has none, and
    several rows of the robustness tables are exactly that case.  ``NaN`` in a
    published table reads as a bug in the pipeline rather than as the honest
    answer, which is that there was no plan to price.
    """
    frame = pd.DataFrame({"Method": ["cac", "wachter"],
                          "Usable": [1.0, 0.0], "Cost": [0.25, np.nan]})
    body = _table(tables_dir, frame)
    assert "NaN" not in body
    assert "wachter & 0.000 & -- \\\\" in body
    assert "cac & 1.000 & 0.250 \\\\" in body


def test_every_number_is_printed_to_the_same_precision(tables_dir: Path) -> None:
    """Three decimals throughout, so a column cannot be read as ranked by width."""
    frame = pd.DataFrame({"Rate": [1.0, 0.5, 0.125], "N": [10, 200, 3]})
    body = _table(tables_dir, frame)
    for cell in ("1.000", "0.500", "0.125"):
        assert cell in body
    assert "& 10 \\\\" in body  # integers stay integers


def test_a_table_carrying_latex_can_opt_out_of_escaping(tables_dir: Path) -> None:
    """``escape=False`` is what lets a caption or a cell hold a macro.

    The robustness and case-study tables put ``\\checkmark`` and ``$\\pm$`` in
    cells, which escaping would print literally, backslash and all.
    """
    frame = pd.DataFrame({"Causal": ["$1.00 \\pm 0.00$"]})
    assert "$1.00 \\pm 0.00$" in _table(tables_dir, frame, escape=False)
    assert "textbackslash" in _table(tables_dir, frame, escape=True)


def test_the_table_is_booktabs_and_carries_its_own_reference(
        tables_dir: Path) -> None:
    body = _table(tables_dir, pd.DataFrame({"A": [1.0]}), label="tab:fixture")
    for rule in ("\\toprule", "\\midrule", "\\bottomrule"):
        assert rule in body
    assert "\\label{tab:fixture}" in body
    assert "\\caption{A caption.}" in body
    assert body.endswith("\\end{table}\n")


def test_the_metadata_encoder_takes_what_numpy_hands_it() -> None:
    """Every scalar type the pipeline actually stores, round-tripped.

    These land in ``metadata.json`` at the end of a training run: a type the
    encoder refuses would throw away the model that had just been fitted.
    """
    payload = {"seed": np.int64(42), "auc": np.float32(0.9), "ok": np.bool_(True),
               "bounds": np.array([0.0, 1.0]), "path": Path("a/b.csv"),
               "cols": {"age", "hours"}, "shape": (3, 4),
               "when": pd.Timestamp("2026-01-01")}
    out = json.loads(json.dumps(payload, cls=NumpyJSONEncoder))
    assert out["seed"] == 42 and isinstance(out["seed"], int)
    assert out["auc"] == pytest.approx(0.9, abs=1e-6)
    assert out["ok"] is True
    assert out["bounds"] == [0.0, 1.0]
    assert out["path"] == str(Path("a/b.csv"))  # separator is the platform's
    assert sorted(out["cols"]) == ["age", "hours"]
    assert out["shape"] == [3, 4]
    assert out["when"].startswith("2026-01-01")


def test_an_unknown_object_still_raises() -> None:
    """The encoder widens what JSON accepts; it does not silence a real mistake."""
    with pytest.raises(TypeError):
        json.dumps({"model": object()}, cls=NumpyJSONEncoder)


def test_metadata_survives_a_second_writer(tmp_path: Path) -> None:
    """``update_json`` merges, because two scripts write the same file.

    Training writes the encoded bounds and the black-box scores; a later stage
    adds its own keys and must not drop them.
    """
    path = tmp_path / "meta.json"
    save_json({"dataset": "adult", "auc": 0.918}, path)
    merged = update_json(path, {"auc": 0.920, "scm_edges": 25})
    assert merged == {"dataset": "adult", "auc": 0.920, "scm_edges": 25}
    assert load_json(path) == merged
    assert not list(tmp_path.glob("*.tmp"))  # written atomically, nothing left over


def test_a_dataframe_is_written_without_its_index(tmp_path: Path) -> None:
    """Results are read back by column name, and a stray index column shifts them."""
    frame = pd.DataFrame({"record": [7, 9], "cost": [0.1, 0.2]}).set_index("record")
    path = save_dataframe(frame.reset_index(), tmp_path / "r" / "out.csv")
    assert list(pd.read_csv(path).columns) == ["record", "cost"]


def test_existing_keeps_only_what_is_there(tmp_path: Path) -> None:
    there = tmp_path / "there.csv"
    there.write_text("a\n", encoding="utf-8")
    assert existing([there, tmp_path / "gone.csv"]) == [there]


def test_two_callers_asking_for_one_seed_get_the_same_draws() -> None:
    """Rule #2: a generator per caller, not the global one.

    Sharing the global generator would make a method's output depend on how many
    numbers some other method happened to draw first, which is exactly the
    irreproducibility the rule exists to prevent.
    """
    assert get_rng(7).integers(0, 1000, 5).tolist() == \
           get_rng(7).integers(0, 1000, 5).tolist()
    assert get_rng().integers(0, 1000, 5).tolist() == \
           get_rng(CONFIG.RANDOM_STATE).integers(0, 1000, 5).tolist()
    rng = get_rng(7)
    first = rng.integers(0, 1000, 5).tolist()
    assert rng.integers(0, 1000, 5).tolist() != first  # a stream, not a constant


def test_seeding_the_world_reports_the_seed_it_used() -> None:
    assert set_global_seed() == CONFIG.RANDOM_STATE
    assert set_global_seed(3) == 3
    drawn = np.random.rand(3).tolist()
    set_global_seed(3)
    assert np.random.rand(3).tolist() == drawn
    set_global_seed()  # leave the interpreter as the rest of the suite expects


def test_logging_writes_a_file_named_for_the_script(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rule #4: stdout *and* a timestamped file, one per script."""
    monkeypatch.setattr("src.utils.logging.CONFIG",
                        CONFIG.model_copy(update={"LOG_DIR": tmp_path / "logs"}))
    monkeypatch.setattr("src.utils.logging._CONFIGURED", {})
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    try:
        path = setup_logging("99_fixture")
        get_logger(__name__).info("a message with a dash \u2014 and a caption")
        assert path.parent == tmp_path / "logs"
        assert path.name.startswith("99_fixture_") and path.suffix == ".log"
        # UTF-8 on disk whatever the console encoding is, or a logged em dash would
        # abort the run that logged it.
        assert "dash \u2014 and" in path.read_text(encoding="utf-8")
    finally:
        # ``setup_logging`` replaces the root handlers, so leaving them attached
        # would keep this test's log file open for the rest of the session --- on
        # Windows that also blocks the temporary directory from being cleaned up.
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in before:
            root.addHandler(handler)
        root.setLevel(level)
