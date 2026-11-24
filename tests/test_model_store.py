import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from nba_predictor.model_store import (
    MANIFEST_FILE,
    MODEL_FILE,
    ModelIntegrityError,
    load_model,
    save_model,
)

X = np.array([[0.0], [1.0]] * 20)
Y = np.array([0, 1] * 20)


@pytest.fixture
def saved(tmp_path):
    digest = save_model(LogisticRegression().fit(X, Y), tmp_path, {"name": "toy"})
    return tmp_path, digest


def test_round_trip(saved):
    folder, digest = saved
    loaded = load_model(folder, digest)
    assert loaded.model.predict([[0.0], [1.0]]).tolist() == [0, 1]
    assert loaded.manifest["name"] == "toy"


def test_swapped_model_file_is_refused(saved):
    folder, _ = saved
    save_model(LogisticRegression().fit(X, 1 - Y), folder / "other")
    (folder / MODEL_FILE).write_bytes((folder / "other" / MODEL_FILE).read_bytes())
    with pytest.raises(ModelIntegrityError, match="hash"):
        load_model(folder)


def test_pinned_hash_beats_an_edited_manifest(saved):
    folder, digest = saved
    manifest = json.loads((folder / MANIFEST_FILE).read_text())
    manifest["sha256"] = "0" * 64
    (folder / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(ModelIntegrityError):
        load_model(folder)
    assert load_model(folder, digest.upper()).model is not None


@pytest.mark.parametrize("damage", ["no manifest", "bad json", "no model", "no hash"])
def test_damaged_folders_are_refused(saved, damage):
    folder, _ = saved
    if damage == "no manifest":
        (folder / MANIFEST_FILE).unlink()
    elif damage == "bad json":
        (folder / MANIFEST_FILE).write_text("{not json")
    elif damage == "no model":
        (folder / MODEL_FILE).unlink()
    else:
        (folder / MANIFEST_FILE).write_text(json.dumps({"sklearn_version": "x"}))
    with pytest.raises(ModelIntegrityError):
        load_model(folder)


def test_model_from_another_sklearn_version_is_refused(saved):
    folder, digest = saved
    manifest = json.loads((folder / MANIFEST_FILE).read_text())
    manifest["sklearn_version"] = "0.1"
    (folder / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(ModelIntegrityError, match="retrain"):
        load_model(folder, digest)


def test_a_model_that_cannot_be_unpickled_is_refused(saved):
    import hashlib

    folder, _ = saved
    data = b"cbuiltins\nno_such_function\n."
    (folder / MODEL_FILE).write_bytes(data)
    manifest = json.loads((folder / MANIFEST_FILE).read_text())
    manifest["sha256"] = hashlib.sha256(data).hexdigest()
    (folder / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(ModelIntegrityError, match="can't be loaded"):
        load_model(folder)
