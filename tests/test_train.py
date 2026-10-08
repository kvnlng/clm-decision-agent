"""Tests for train.py, with tiny stand-in heads and random vectors: no model is loaded."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name,unused-argument
import json

import mlx.core as mx
import pytest
from mlx import nn

from train import HeadsModel, accuracy, fit, probe_accuracy, save_heads

DIM, CLASSES = 12, 3


@pytest.fixture
def data():
    mx.random.seed(0)
    centres = mx.random.normal((CLASSES, DIM)) * 3

    def make(n):
        labels = mx.array([i % CLASSES for i in range(n)])
        return centres[labels] + mx.random.normal((n, DIM)), labels
    actions = mx.random.normal((CLASSES, DIM))
    return make(90), make(45), actions


def tiny_model():
    mx.random.seed(1)
    return HeadsModel(nn.Linear(DIM, 8), nn.Linear(DIM, 8), 20.0)


def test_training_improves_accuracy_on_examples_it_has_not_seen(data):
    (train_x, train_y), (test_x, test_y), actions = data
    model = tiny_model()
    before = accuracy(model, test_x, test_y, actions)
    info = fit(model, train_x, train_y, actions, epochs=40, learning_rate=1e-2)
    after = accuracy(model, test_x, test_y, actions)
    assert after > before and after >= 0.9
    assert info["epochs"] >= 1
    assert 0 <= info["validation_accuracy"] <= 1


def test_fit_is_repeatable(data):
    (train_x, train_y), (test_x, test_y), actions = data
    results = []
    for _ in range(2):
        model = tiny_model()
        fit(model, train_x, train_y, actions, epochs=5, learning_rate=1e-2)
        results.append(accuracy(model, test_x, test_y, actions))
    assert results[0] == results[1]


def test_probe_baseline_learns_separable_data(data):
    (train_x, train_y), (test_x, test_y), _ = data
    assert probe_accuracy(train_x, train_y, test_x, test_y, CLASSES) >= 0.9


def test_saved_heads_use_the_released_layout_and_record_the_training(data, tmp_path):
    source = tmp_path / "released"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"cfg": {"width": 8}}))
    model = tiny_model()
    out = tmp_path / "trained"
    save_heads(model, source, out, {"routing_sha": "abc", "train": 90})

    weights = mx.load(str(out / "CLM_v0.1-8B.safetensors"))
    assert set(weights) == {"state_head.weight", "state_head.bias", "action_head.weight", "action_head.bias",
                            "logit_scale"}
    assert mx.allclose(weights["state_head.weight"], model.state_head.weight).item()
    assert abs(mx.exp(weights["logit_scale"]).item() - 20.0) < 1e-3
    assert json.loads((out / "config.json").read_text()) == {"cfg": {"width": 8}}
    assert json.loads((out / "training.json").read_text()) == {"routing_sha": "abc", "train": 90}


def test_pinned_rows_are_never_held_out_for_validation(data):
    (train_x, train_y), _, actions = data
    seen = []

    class Spy(HeadsModel):
        def logits(self, states, actions):
            seen.append(states.shape[0])
            return super().logits(states, actions)

    mx.random.seed(1)
    model = Spy(nn.Linear(DIM, 8), nn.Linear(DIM, 8), 20.0)
    info = fit(model, train_x, train_y, actions, epochs=1, learning_rate=1e-2, pinned=40)
    assert info["validation_examples"] == 10        # a fifth of the 50 rows that are not pinned


def test_training_that_only_hurts_keeps_the_starting_weights(data):
    (train_x, train_y), _, actions = data
    model = tiny_model()
    fit(model, train_x, train_y, actions, epochs=40, learning_rate=1e-2)
    good = accuracy(model, train_x, train_y, actions)
    # Ten correctly labelled rows (two of them become the validation rows), then eighty mislabelled, pinned ones
    labels = mx.concatenate([train_y[:10], (train_y[10:] + 1) % CLASSES])
    info = fit(model, train_x, labels, actions, epochs=3, learning_rate=1e-2, pinned=80)
    assert info["best_epoch"] == 0
    assert accuracy(model, train_x, train_y, actions) == good


def test_saving_removes_the_old_description_before_writing_new_weights(data, tmp_path, monkeypatch):
    source = tmp_path / "released"
    source.mkdir()
    (source / "config.json").write_text("{}")
    out = tmp_path / "trained"
    save_heads(tiny_model(), source, out, {"routing_sha": "old"})

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(mx, "save_safetensors", interrupted)
    with pytest.raises(KeyboardInterrupt):
        save_heads(tiny_model(), source, out, {"routing_sha": "new"})
    assert not (out / "training.json").exists()
