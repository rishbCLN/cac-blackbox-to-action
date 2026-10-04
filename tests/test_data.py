"""Tests for the data layer (§4, global rule #6).

This layer owns the only translation in the project: *semantic* records a human
reads, *encoded* records the SCM and the cost model reason about, and *model*
records the black box scores.  Everything downstream trusts that the three views
describe the same person, so most of what follows is round-trips — encode/decode,
model/encoded, spec/JSON — plus the two claims §4 makes that no single row can
witness: that the scaler learned from the training split only, and that
protected attributes are never rescaled.

The acquisition chain is exercised with local files and fakes.  Nothing here
touches the network: a suite that quietly depends on a Kaggle mirror being up is
a suite that fails for the next person trying to reproduce the paper.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.config import CONFIG, DATASETS
from src.data import load_diabetes, load_heloc, sources
from src.data.load_adult import EDUCATION_LEVELS
from src.data.preprocess import (
    DatasetBundle,
    LevelEncoder,
    Transformer,
    build_dataset,
    dataset_spec,
    get_dataset,
    load_dataset,
    processed_paths,
)
from src.data.schema import DatasetSpec


@pytest.fixture(scope="module", params=list(DATASETS))
def bundle(request: pytest.FixtureRequest) -> DatasetBundle:
    """One prepared dataset, read from disk — never rebuilt, never downloaded."""
    return load_dataset(request.param)


@pytest.fixture(scope="module")
def adult() -> DatasetBundle:
    return load_dataset("adult")


@pytest.fixture(scope="module")
def heloc() -> DatasetBundle:
    return load_dataset("heloc")


@pytest.fixture(scope="module")
def diabetes() -> DatasetBundle:
    return load_dataset("diabetes")
# ------------------------------------------------------------------------ schema
def test_every_feature_declares_a_space_to_live_in(bundle: DatasetBundle) -> None:
    """A feature with neither levels nor bounds has no search space.

    ``encoded_bounds`` is the box every method optimises inside, so a column that
    cannot answer that question would be silently unsearchable rather than
    loudly broken.
    """
    spec = bundle.spec
    assert spec.feature_names == list(dict.fromkeys(spec.feature_names)), "duplicate column"
    assert spec.target not in spec.feature_names, "the target is not a feature"
    for f in spec.features:
        assert f.kind in ("numeric", "ordinal", "categorical", "binary"), f.name
        assert f.monotone in ("any", "increase", "decrease"), f.name
        lo, hi = f.encoded_bounds()
        assert lo < hi, f"{f.name}: empty box [{lo}, {hi}]"
        if f.kind == "numeric":
            assert f.levels is None and f.bounds is not None, f.name
        else:
            assert f.levels and len(set(f.levels)) == len(f.levels), f.name
            assert (lo, hi) == (0.0, float(f.n_levels - 1)), f.name
            assert f.is_discrete, f.name
        if f.kind == "binary":
            assert f.n_levels == 2, f.name


def test_the_schema_survives_the_round_trip_through_json(bundle: DatasetBundle) -> None:
    """``metadata.json`` is how the SCM and cost layers learn the schema.

    They read it back through :meth:`DatasetSpec.from_dict`, so a field that does
    not survive ``json.dumps`` is a field those layers would never see.
    """
    spec = bundle.spec
    assert DatasetSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_the_flags_and_the_views_tell_the_same_story(bundle: DatasetBundle) -> None:
    spec = bundle.spec
    names = set(spec.feature_names)
    assert set(spec.numeric_features) | set(spec.categorical_features) == names
    assert not set(spec.numeric_features) & set(spec.categorical_features)
    assert set(spec.manipulable_features) | set(spec.non_manipulable_features) == names
    assert not set(spec.manipulable_features) & set(spec.non_manipulable_features)
    assert all(spec.get(n).protected for n in spec.protected_features)
    for j, name in enumerate(spec.feature_names):
        assert spec.index(name) == j and spec.get(name).name == name
    assert spec.subset(spec.feature_names) == spec.features
    with pytest.raises(KeyError, match="not a feature"):
        spec.get("no_such_column")
# ---------------------------------------------------------------- level encoding
def test_the_level_encoder_keeps_the_order_the_spec_declares() -> None:
    """A documented deviation from §4.3, and the reason for it.

    §4.3 offers ``LabelEncoder`` or one-hot; ordinal columns get neither.
    ``LabelEncoder`` sorts alphabetically, which on Adult's education ladder puts
    ``Bachelors`` below ``Doctorate`` below ``HS-grad`` — so "education went up by
    one" would mean nothing, and §7's ladder cost would price a step by whatever
    the spelling happened to be.
    """
    enc = LevelEncoder(EDUCATION_LEVELS)
    codes = enc.transform(EDUCATION_LEVELS)
    assert list(codes) == list(range(len(EDUCATION_LEVELS)))
    assert len(enc) == len(EDUCATION_LEVELS)
    assert list(enc.inverse_transform(codes)) == list(EDUCATION_LEVELS)

    ladder = ["HS-grad", "Bachelors", "Masters", "Doctorate"]
    got = list(enc.transform(ladder))
    assert got == sorted(got), "the declared ladder must encode monotonically"
    alphabetical = list(LevelEncoder(sorted(EDUCATION_LEVELS)).transform(ladder))
    assert alphabetical != sorted(alphabetical), (
        "vacuous test: alphabetical order happens to agree with the ladder")


def test_an_unknown_level_is_an_error_rather_than_a_silent_code() -> None:
    enc = LevelEncoder(("Female", "Male"))
    with pytest.raises(ValueError, match="unknown level"):
        enc.transform(["Female", "Nonbinary"])


def test_inverse_transform_snaps_a_code_that_wandered_out_of_range() -> None:
    """Search methods propose real numbers; decoding must still name a level.

    The rounding is ``np.rint``, i.e. half-to-even, which is also what ``project``
    uses — so a code sitting exactly between two levels resolves the same way
    everywhere in the pipeline rather than one way in the search and another way
    in the explanation a human reads.
    """
    enc = LevelEncoder(("a", "b", "c"))
    assert list(enc.inverse_transform([-4.0, 0.49, 1.4, 99.0])) == ["a", "a", "b", "c"]
    assert list(enc.inverse_transform([0.5, 1.5])) == ["a", "c"]  # half to even
# --------------------------------------------------------------- the three spaces
def test_encode_then_decode_is_the_identity_on_real_records(
        bundle: DatasetBundle) -> None:
    """Whatever a human is shown has been through this round trip.

    Every explanation in the paper, the demo app and the user study is a decoded
    counterfactual, so a lossy ``decode`` would misreport advice that was computed
    correctly — the hardest kind of error to notice.
    """
    frame = bundle.frame_test()
    back = bundle.transformer.decode(bundle.transformer.encode(frame))
    assert list(back.columns) == bundle.feature_names
    for f in bundle.spec.features:
        if f.kind == "numeric":
            assert np.allclose(back[f.name].to_numpy(float),
                               frame[f.name].to_numpy(float), atol=1e-9), f.name
        else:
            assert list(back[f.name]) == list(frame[f.name].astype(str)), f.name


def test_model_space_is_reversible(bundle: DatasetBundle) -> None:
    t = bundle.transformer
    X = bundle.X_test
    assert np.allclose(t.from_model_space(t.to_model_space(X)), X, atol=1e-6)
    assert t.as_frame(X).shape == X.shape
    params = t.scaler_params()
    assert len(params["mean"]) == len(params["scale"]) == len(params["columns"])


def test_protected_attributes_are_never_rescaled(bundle: DatasetBundle) -> None:
    """§4.2, stated as an equality between two spaces.

    ``age`` has to read as years wherever a human or the SCM looks at it, which
    means the scaler must leave it alone even though it is numeric.
    """
    t, spec = bundle.transformer, bundle.spec
    X, M = bundle.X_test, bundle.transformer.to_model_space(bundle.X_test)
    for name in spec.protected_features:
        assert np.allclose(M[:, spec.index(name)], X[:, spec.index(name)],
                           atol=1e-9), f"{name} was rescaled"
    scaled = [n for n in spec.numeric_features if n not in spec.protected_features]
    assert t.scaled_columns == scaled, "the scaler's column list is not the rule §4.2 states"
    assert scaled, f"{bundle.name}: nothing is scaled, so this comparison proves nothing"
    assert any(not np.allclose(M[:, spec.index(n)], X[:, spec.index(n)], atol=1e-9)
               for n in scaled), "the scaler is a no-op everywhere; the test is vacuous"
def test_the_scaler_learned_from_the_training_split_only(bundle: DatasetBundle) -> None:
    """Leakage would be invisible in every number the pipeline prints.

    ``build_dataset`` fits the transformer on ``frame.iloc[train_idx]``.  Refitting
    on the training rows must reproduce it exactly — and refitting on *all* rows
    must not, or the check would pass on a leaking implementation too.
    """
    t = bundle.transformer
    if not t.scaled_columns:
        pytest.skip(f"{bundle.name} scales nothing")
    train_fit = Transformer(bundle.spec).fit(bundle.frame_train())
    assert np.allclose(train_fit.scaler.mean_, t.scaler.mean_, atol=1e-9)
    assert np.allclose(train_fit.scaler.scale_, t.scaler.scale_, atol=1e-9)
    assert np.allclose(train_fit.bounds, t.bounds, atol=1e-9), (
        "the encoded box was tightened on rows the model never trained on")
    full_fit = Transformer(bundle.spec).fit(bundle.frame)
    assert not np.allclose(full_fit.scaler.mean_, t.scaler.mean_, atol=1e-9), (
        "train-only and all-rows fits agree, so this test cannot detect leakage")


def test_the_transformers_box_is_the_training_support_not_the_search_space(
        bundle: DatasetBundle) -> None:
    """Two boxes exist in this project, and they are not the same box.

    :meth:`Transformer.fit` tightens each numeric range to the support observed
    **on the training split**, which is what ``metadata.json`` publishes as
    ``encoded_bounds``.  The box every search actually optimises inside is the
    *declared* one from :meth:`FeatureSpec.encoded_bounds` (``cost_model.bounds``,
    ``FCM.bounds``), which is wider.

    So training rows are fixed points of ``Transformer.project`` by construction,
    while a held-out row may legitimately sit outside it — an applicant with more
    open accounts than anyone in the training split is not an infeasible person.
    That is why no method calls this method: clipping a real record to the
    training support would silently rewrite the factual profile being explained.
    """
    t, spec = bundle.transformer, bundle.spec
    declared = np.array([f.encoded_bounds() for f in spec.features], dtype=float)
    assert np.all(t.bounds[:, 0] >= declared[:, 0] - 1e-9)
    assert np.all(t.bounds[:, 1] <= declared[:, 1] + 1e-9)
    assert np.allclose(t.project(bundle.X_train), bundle.X_train, atol=1e-9), (
        "the box was fitted on these very rows, so they cannot be outside it")

    once = t.project(bundle.X_test + 3.7)
    assert np.allclose(t.project(once), once, atol=1e-12), "project is not idempotent"
    assert np.all(once >= t.bounds[:, 0] - 1e-9)
    assert np.all(once <= t.bounds[:, 1] + 1e-9)
    disc = np.flatnonzero(t.is_integer)
    assert np.allclose(once[:, disc], np.rint(once[:, disc]), atol=1e-9)


def test_an_unfitted_transformer_refuses_to_map_to_model_space(
        bundle: DatasetBundle) -> None:
    """Encoding needs no fit; scaling does.  The failure has to be loud."""
    raw = Transformer(bundle.spec)
    row = bundle.X_test[:1]
    with pytest.raises(RuntimeError, match="fitted"):
        raw.to_model_space(row)
    with pytest.raises(RuntimeError, match="fitted"):
        raw.from_model_space(row)
    assert np.allclose(raw.encode(bundle.frame_test().head(3)), bundle.X_test[:3])
# -------------------------------------------------------------------- the split
def test_the_split_is_the_one_the_config_asks_for(bundle: DatasetBundle) -> None:
    """80/20, stratified, disjoint — and reproducible from the seed alone (rule #2).

    Re-deriving the split rather than checking its shape is what keeps the
    published tables reproducible: a change in how rows are assigned surfaces
    here instead of as a quietly different number in §9.
    """
    n = len(bundle.frame)
    train, test = bundle.train_idx, bundle.test_idx
    assert len(train) + len(test) == n
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(n))
    assert len(test) == pytest.approx(CONFIG.TEST_SIZE * n, abs=1)
    expect_train, expect_test = train_test_split(
        np.arange(n), test_size=CONFIG.TEST_SIZE,
        random_state=CONFIG.RANDOM_STATE, stratify=bundle.y,
    )
    assert np.array_equal(train, np.sort(expect_train))
    assert np.array_equal(test, np.sort(expect_test))
    assert bundle.y_train.mean() == pytest.approx(bundle.y_test.mean(), abs=0.02)
    d = len(bundle.feature_names)
    assert bundle.X_train.shape == (len(train), d)
    assert bundle.X_test.shape == (len(test), d)
    assert len(bundle.frame_train()) == len(train)
    assert len(bundle.frame_test()) == len(test)


# ------------------------------------------------------------ what "clean" means
def test_a_prepared_dataset_has_nothing_left_to_clean(bundle: DatasetBundle) -> None:
    """The claims §4 makes about ``data/processed/<dataset>.csv``."""
    frame, spec = bundle.frame, bundle.spec
    assert not frame.isna().any().any(), "missing values survived cleaning"
    assert set(np.unique(bundle.y)) == {0, 1}, "the target is not binary 0/1"
    assert 0.05 < float(bundle.y.mean()) < 0.95, "one class is nearly absent"
    assert np.isfinite(bundle.X).all()
    for f in spec.features:
        col = frame[f.name]
        if f.kind == "numeric":
            lo, hi = f.bounds
            assert col.min() >= lo - 1e-9, f"{f.name} below its declared bound"
            assert col.max() <= hi + 1e-9, f"{f.name} above its declared bound"
            if f.integer:
                values = col.to_numpy(dtype=float)
                assert np.allclose(values, np.rint(values)), f"{f.name} is not integral"
        else:
            assert set(col.astype(str)) <= set(f.levels or ()), f"{f.name} has stray levels"
def test_the_ficos_missing_value_codes_are_gone(heloc: DatasetBundle) -> None:
    """-7/-8/-9 are FICO's "no record" sentinels, not utilisation percentages.

    Left in, they would be perfectly ordinary numbers to the black box, the SCM
    and the cost model alike, and recourse would be priced against them.
    """
    numeric = heloc.frame[heloc.spec.feature_names]
    assert not numeric.isin(load_heloc.SPECIAL_VALUES).any().any()


def test_adult_keeps_no_duplicate_profiles(adult: DatasetBundle) -> None:
    """Identical rows would let one person be sampled into both splits."""
    assert not adult.frame.duplicated().any()


# -------------------------------------------------------------------- persistence
def test_the_metadata_describes_the_bundle_it_was_written_with(
        bundle: DatasetBundle) -> None:
    """``metadata.json`` is §4's deliverable, and the SCM/cost layers' input."""
    path = Path(str(processed_paths(bundle.name)["metadata"]))
    meta = json.loads(path.read_text(encoding="utf-8"))
    spec, t = bundle.spec, bundle.transformer
    assert meta["dataset"] == bundle.name
    assert meta["n_rows"] == len(bundle.frame)
    assert meta["n_features"] == len(spec.features)
    assert meta["feature_names"] == spec.feature_names
    assert meta["feature_types"] == {f.name: f.kind for f in spec.features}
    assert meta["target"] == spec.target
    assert meta["protected_attributes"] == spec.protected_features
    assert meta["non_manipulable"] == spec.non_manipulable_features
    assert meta["positive_rate"] == pytest.approx(float(bundle.y.mean()))
    assert meta["train_idx"] == bundle.train_idx.tolist()
    assert meta["test_idx"] == bundle.test_idx.tolist()
    assert np.allclose(meta["encoded_bounds"], t.bounds)
    assert meta["scaler"]["columns"] == t.scaled_columns
    assert meta["level_maps"] == {n: list(e.classes_) for n, e in t.encoders.items()}
    assert meta["random_state"] == CONFIG.RANDOM_STATE == 42
    assert bool(meta["is_synthetic"]) == bundle.is_synthetic


def test_reading_the_cache_reproduces_the_prepared_dataset(bundle: DatasetBundle) -> None:
    """A second run must be offline and identical to the first.

    ``build_dataset`` returns the cache when the CSV and the transformer are both
    on disk; if it rebuilt instead, it would re-download and could re-split.
    """
    again = build_dataset(bundle.name)
    assert np.array_equal(again.X, bundle.X)
    assert np.array_equal(again.y, bundle.y)
    assert np.array_equal(again.train_idx, bundle.train_idx)
    assert again.spec.features == bundle.spec.features
    assert np.array_equal(get_dataset(bundle.name).X, bundle.X)
    assert dataset_spec(bundle.name).features == bundle.spec.features
    assert dataset_spec(bundle.name).target == bundle.spec.target
def test_an_unknown_dataset_is_refused_and_a_missing_one_says_what_to_run() -> None:
    with pytest.raises(KeyError, match="unknown dataset"):
        dataset_spec("titanic")
    with pytest.raises(KeyError, match="unknown dataset"):
        build_dataset("titanic")
    with pytest.raises(FileNotFoundError, match="01_train_blackbox"):
        load_dataset("titanic")


# --------------------------------------------------------- acquisition, offline
def test_local_csv_skips_what_it_cannot_read(tmp_path: Path) -> None:
    """One unreadable file in ``data/raw/`` must not sink the whole chain."""
    assert sources.local_csv(tmp_path / "does_not_exist") is None
    (tmp_path / "a_broken.csv").write_text("", encoding="utf-8")
    (tmp_path / "b_good.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    got = sources.local_csv(tmp_path)
    assert got is not None and list(got.columns) == ["x", "y"]


def test_has_columns_matches_names_exactly() -> None:
    """A prefix match would accept ``age_band`` as ``age``."""
    frame = pd.DataFrame({"age": [1], "income": [0]})
    assert sources.has_columns("age", "income")(frame)
    assert not sources.has_columns("ag")(frame)
    assert not sources.has_columns("age", "gender")(frame)


def test_cache_raw_never_overwrites_an_existing_file(tmp_path: Path) -> None:
    """The cache is what makes run two offline; it must not be clobbered."""
    path = tmp_path / "raw.csv"
    path.write_text("keep,me\n1,2\n", encoding="utf-8")
    sources.cache_raw(pd.DataFrame({"other": [9]}), path)
    assert list(pd.read_csv(path).columns) == ["keep", "me"]


def test_acquire_says_where_to_put_a_csv_when_every_source_fails(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"data/raw/toy"):
        sources.acquire(
            "toy",
            steps=(("none", lambda: None),),
            validate=sources.has_columns("age"),
            cache_to=tmp_path / "unused.csv",
        )
    assert not (tmp_path / "unused.csv").exists()
def test_acquire_walks_the_chain_and_caches_what_it_used(tmp_path: Path) -> None:
    """A *reduced* public copy has to be rejected, not silently used.

    Several mirrors of these datasets ship fewer columns — Kaggle's
    ``uciml/german-credit`` has no risk label at all — which is why ``acquire``
    validates every candidate instead of taking the first one that parses.
    """
    calls: list[str] = []

    def step(label: str, value: object):
        def _run() -> pd.DataFrame | None:
            calls.append(label)
            if isinstance(value, Exception):
                raise value
            return value  # type: ignore[return-value]
        return _run

    cache = tmp_path / "cached.csv"
    frame, label = sources.acquire(
        "toy",
        steps=(
            ("boom", step("boom", RuntimeError("offline"))),
            ("none", step("none", None)),
            ("empty", step("empty", pd.DataFrame())),
            ("reduced", step("reduced", pd.DataFrame({"age": [1]}))),
            ("mirror", step("mirror", pd.DataFrame({" age ": [30, 40], "income": [0, 1]}))),
        ),
        validate=sources.has_columns("age", "income"),
        cache_to=cache,
    )
    assert label == "mirror"
    assert calls == ["boom", "none", "empty", "reduced", "mirror"], "the chain skipped a step"
    assert list(frame.columns) == ["age", "income"], "column names are stripped"
    assert cache.exists() and list(pd.read_csv(cache).columns) == ["age", "income"]


# ------------------------------------------------------------ the HELOC surrogate
def test_the_heloc_surrogate_is_deterministic() -> None:
    """Rule #2 reaches the fallback too, or the offline pipeline is not reproducible."""
    first = load_heloc._synthetic(n=200)
    second = load_heloc._synthetic(n=200)
    pd.testing.assert_frame_equal(first, second)
    assert set(load_heloc.SPEC.feature_names) <= set(first.columns)
def test_the_surrogate_declares_itself_so_no_number_is_mistaken_for_real(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """``build_dataset`` keys ``is_synthetic`` off the word SYNTHETIC in the notes.

    If the fallback ever fires quietly, the paper would report illustrative
    numbers as FICO's.  That one string is the only thing standing between the two,
    so the exact condition ``build_dataset`` tests is asserted here — along with
    the surrogate satisfying the same cleaning invariants as the real thing.
    """
    monkeypatch.setattr(
        load_heloc, "sources",
        types.SimpleNamespace(acquire=lambda *a, **k: (load_heloc._synthetic(n=400),
                                                       "synthetic")),
    )
    frame, spec = load_heloc.load_heloc()
    assert any("SYNTHETIC" in note for note in spec.notes)
    assert "SYNTHETIC" in spec.source
    assert spec.features == load_heloc.SPEC.features
    assert spec.target == load_heloc.SPEC.target
    assert not frame.isna().any().any()
    assert set(np.unique(frame[spec.target])) == {0, 1}
    for f in spec.features:
        if f.bounds is not None:
            assert frame[f.name].min() >= f.bounds[0] - 1e-9, f.name
            assert frame[f.name].max() <= f.bounds[1] + 1e-9, f.name


# ------------------------------------------------------------------ Pima diabetes
def test_the_diabetes_label_points_the_same_way_as_every_other_dataset(
        diabetes: DatasetBundle) -> None:
    """Pima's ``Outcome`` is 1 for the bad news; every other target here is not.

    The whole evaluation layer asks for ``y_target=1`` and calls it the
    favourable flip.  Storing Pima's sign unchanged would silently make one of
    four datasets measure recourse *towards* diabetes, and every aggregate would
    average the two together.
    """
    assert diabetes.spec.target == "no_diabetes"
    # Pima is 500 negative / 268 positive, so 500/768 avoid onset.
    assert diabetes.frame[diabetes.spec.target].sum() == 500
    assert set(np.unique(diabetes.y)) == {0, 1}
    assert "no_diabetes = 1" in diabetes.spec.positive_meaning


def test_the_two_readings_cannot_be_prescribed(diabetes: DatasetBundle) -> None:
    """``SkinThickness`` and ``Insulin`` are measurements, not levers.

    This is the claim the dataset was added to test.  If either became
    manipulable, CAC would be free to "prescribe" a serum insulin level and the
    comparison against the point-based baselines would lose its sharpest case.
    """
    fixed = set(diabetes.spec.non_manipulable_features)
    assert {"SkinThickness", "Insulin"} <= fixed
    assert {"Age", "Pregnancies", "DiabetesPedigreeFunction"} <= fixed
    assert set(diabetes.spec.manipulable_features) == {"BMI", "Glucose", "BloodPressure"}


def test_the_redistributors_class_conditional_fill_is_gone(
        diabetes: DatasetBundle) -> None:
    """Nothing the leak detector recognises may survive into the prepared frame.

    The copy of Pima this project was given had ``Insulin`` filled with 102.5 on
    236 negative rows and 169.5 on 138 positive ones, and ``SkinThickness`` with
    27 and 32 the same way: 655 of 768 rows carried their own answer in a
    column, and an XGBoost read that as AUC 0.96 against a literature norm near
    0.83.  Stated as idempotence — run the detector over its own output and it
    must find nothing — so that a differently-imputed copy of the file fails here
    too, not just this one's five values.

    Note that purity *alone* would be the wrong assertion: Glucose is 99 on 17
    rows and every one of them avoids onset, which is not a leak, it is what a
    normal blood-sugar reading means.  The detector requires a frequency spike as
    well, and so does this test.
    """
    frame = diabetes.frame
    y = frame[diabetes.spec.target]
    for name in load_diabetes.ZERO_IS_MISSING:
        # The detector measures purity against Pima's own sign, as the loader does.
        assert load_diabetes._leaked_values(frame[name], 1 - y) == [], name
    assert not frame["Insulin"].isin([102.5, 169.5]).any(), "the shipped fill survived"
    assert not frame["BloodPressure"].isin([74.5]).any()


def test_the_leak_detector_leaves_honest_measurements_alone() -> None:
    """A column with no fill value in it must come back untouched.

    The detector uses the labels, which is the one place in this project where a
    label touches a feature before the split.  That is only defensible while it
    cannot fire on real data, so the guard is tested in both directions: a clean
    column keeps every value, and a planted class-conditional fill is caught.
    """
    rng = np.random.default_rng(CONFIG.RANDOM_STATE)
    n = 600
    y = pd.Series(rng.integers(0, 2, n))
    clean = pd.Series(rng.integers(60, 200, n), name="Glucose")
    assert load_diabetes._leaked_values(clean, y) == []

    planted = clean.copy()
    planted[y == 0] = 102.5  # exactly what the shipped file does to Insulin
    assert 102.5 in load_diabetes._leaked_values(planted, y)


def test_the_diabetes_surrogate_is_deterministic() -> None:
    """Rule #2 reaches the fallback too, or the offline pipeline is not reproducible."""
    first = load_diabetes._synthetic(n=200)
    second = load_diabetes._synthetic(n=200)
    pd.testing.assert_frame_equal(first, second)
    assert set(load_diabetes.SPEC.feature_names) <= set(first.columns)
    assert load_diabetes.SOURCE_TARGET in first.columns, "the surrogate ships Pima's sign"


def test_the_diabetes_surrogate_declares_itself(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Same contract as the HELOC surrogate: the notes say SYNTHETIC or nothing does."""
    monkeypatch.setattr(
        load_diabetes, "sources",
        types.SimpleNamespace(acquire=lambda *a, **k: (load_diabetes._synthetic(n=400),
                                                      "synthetic")),
    )
    frame, spec = load_diabetes.load_diabetes()
    assert any("SYNTHETIC" in note for note in spec.notes)
    assert "SYNTHETIC" in spec.source
    assert spec.features == load_diabetes.SPEC.features
    assert spec.target == load_diabetes.SPEC.target
    assert not frame.isna().any().any()
    assert set(np.unique(frame[spec.target])) == {0, 1}
    for f in spec.features:
        if f.bounds is not None:
            assert frame[f.name].min() >= f.bounds[0] - 1e-9, f.name
            assert frame[f.name].max() <= f.bounds[1] + 1e-9, f.name
