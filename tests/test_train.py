"""Tests for train.py, with tiny stand-in heads and random vectors: no model is loaded."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name,unused-argument
import functools
import json
import math

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file
from torch import nn

import train
from clm_torch import TorchHeads
from train import HeadsModel, accuracy, fit, probe_accuracy, save_heads

DIM, CLASSES = 12, 3


@pytest.fixture
def data():
    torch.manual_seed(0)
    centres = torch.randn(CLASSES, DIM) * 3

    def make(n):
        labels = torch.tensor([i % CLASSES for i in range(n)])
        return centres[labels] + torch.randn(n, DIM), labels
    actions = torch.randn(CLASSES, DIM)
    return make(90), make(45), actions


def tiny_model():
    torch.manual_seed(1)
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


def test_saved_heads_use_the_released_layout_and_record_the_training(tmp_path):
    model = tiny_model()
    out = tmp_path / "trained"
    save_heads(model.weights(), {"cfg": {"width": 8}}, out, {"routing_sha": "abc", "train": 90})

    weights = load_file(str(out / "CLM_v0.1-8B.safetensors"))
    assert set(weights) == {"state_head.weight", "state_head.bias", "action_head.weight", "action_head.bias",
                            "logit_scale"}
    np.testing.assert_allclose(weights["state_head.weight"], model.state_head.weight.detach().numpy())
    assert abs(math.exp(weights["logit_scale"][0]) - 20.0) < 1e-3
    assert json.loads((out / "config.json").read_text()) == {"cfg": {"width": 8}}
    assert json.loads((out / "training.json").read_text()) == {"routing_sha": "abc", "train": 90}


def test_pinned_rows_are_never_held_out_for_validation(data):
    (train_x, train_y), _, actions = data
    seen = []

    class Spy(HeadsModel):  # pylint: disable=abstract-method
        def logits(self, states, actions):
            seen.append(states.shape[0])
            return super().logits(states, actions)

    torch.manual_seed(1)
    model = Spy(nn.Linear(DIM, 8), nn.Linear(DIM, 8), 20.0)
    info = fit(model, train_x, train_y, actions, epochs=1, learning_rate=1e-2, pinned=40)
    assert info["validation_examples"] == 10        # a fifth of the 50 rows that are not pinned


def test_training_that_only_hurts_keeps_the_starting_weights(data):
    (train_x, train_y), _, actions = data
    model = tiny_model()
    fit(model, train_x, train_y, actions, epochs=40, learning_rate=1e-2)
    good = accuracy(model, train_x, train_y, actions)
    # Ten correctly labelled rows (two of them become the validation rows), then eighty mislabelled, pinned ones
    labels = torch.cat([train_y[:10], (train_y[10:] + 1) % CLASSES])
    info = fit(model, train_x, labels, actions, epochs=3, learning_rate=1e-2, pinned=80)
    assert info["best_epoch"] == 0
    assert accuracy(model, train_x, train_y, actions) == good


def test_saving_removes_the_old_description_before_writing_new_weights(tmp_path, monkeypatch):
    out = tmp_path / "trained"
    save_heads(tiny_model().weights(), {}, out, {"routing_sha": "old"})

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(train, "save_file", interrupted)
    with pytest.raises(KeyboardInterrupt):
        save_heads(tiny_model().weights(), {}, out, {"routing_sha": "new"})
    assert not (out / "training.json").exists()


class FakeEngine:
    """CLM's engine interface over tiny real heads and a made-up encoder that separates the routes."""

    def __init__(self, names):
        from test_clm_torch import CONFIG, tiny_weights  # pylint: disable=import-outside-toplevel
        self.heads, self.names = TorchHeads(tiny_weights(), CONFIG, "cpu"), names
        self.head_config = CONFIG
        rng = np.random.default_rng(0)
        self.centres = {name: rng.normal(size=8) * 3 for name in names}
        self.rng = np.random.default_rng(1)

    def embed(self, texts):
        rows = []
        for text in texts:
            name = next((name for name in self.names if name in text), None)
            centre = self.centres[name] if name else self.rng.normal(size=8)
            rows.append(centre + self.rng.normal(size=8))
        return np.array(rows, dtype=np.float32)

    def project(self, vectors, which):
        return self.heads.project(vectors, which)

    def head_weights(self):
        return self.heads.weights()

    def set_head_weights(self, weights):
        self.heads.load(weights)

    def answer(self, state, questions):
        return {"answers": {name: {"type": "noul", "noul": 0.5} for name in questions}}


def test_train_pushes_trained_weights_into_the_engine(scenario):
    names = list(scenario.routes)
    engine = FakeEngine([route.description for route in scenario.routes.values()])
    rows = [(f"{scenario.routes[name].description} case {i}", name) for i, name in
            enumerate(names * 40)]
    facts = train.train(scenario, engine, rows[:60], [], rows[60:], write=lambda line: None)
    assert facts["clm_backend"] == "mlx"
    assert facts["accuracy_after"] >= facts["accuracy_before"]
    model = HeadsModel.from_weights(engine.head_weights(), engine.head_config)
    for name, value in model.weights().items():
        np.testing.assert_allclose(value, engine.head_weights()[name])


def test_saved_heads_record_the_backend_and_load_in_torch(tmp_path):
    from test_clm_torch import CONFIG, tiny_weights  # pylint: disable=import-outside-toplevel
    save_heads(tiny_weights(), CONFIG, tmp_path / "heads", {"clm_backend": "torch"})
    assert json.loads((tmp_path / "heads" / "training.json").read_text())["clm_backend"] == "torch"
    vectors = np.random.default_rng(0).normal(size=(2, 8)).astype(np.float32)
    np.testing.assert_allclose(TorchHeads.trained(tmp_path / "heads", "cpu").project(vectors, "state"),
                               TorchHeads(tiny_weights(), CONFIG, "cpu").project(vectors, "state"), atol=1e-6)


def test_from_weights_round_trips():
    from test_clm_torch import CONFIG, tiny_weights  # pylint: disable=import-outside-toplevel
    model = HeadsModel.from_weights(tiny_weights(), CONFIG)
    for name, value in tiny_weights().items():
        np.testing.assert_allclose(model.weights()[name], value, atol=1e-6)


def test_main_trains_with_the_torch_backend(write_scenario, monkeypatch, tmp_path):
    from conftest import BASE_TOML  # pylint: disable=import-outside-toplevel
    path = write_scenario(BASE_TOML.replace('model_id = "test/clm"',
                                            'backend = "torch"\nmodel_id = "test/clm"\nheads = "my.heads"'))
    examples = [{"text": f"{d} case {i}", "route": r} for i, (r, d) in
                enumerate([("Billing", "Charges and refunds"), ("Tech", "Bugs and outages")] * 40)]
    path.with_name("scenario.examples.jsonl").write_text("\n".join(json.dumps(e) for e in examples))
    seen = []

    def fake_load_engine(*args):
        seen.append(args)
        return FakeEngine(["Charges and refunds", "Bugs and outages"])
    monkeypatch.setattr(train, "load_engine", fake_load_engine)
    # Tiny made-up heads need a faster rate than the real ones to move in a few passes
    monkeypatch.setattr(train, "fit", functools.partial(train.fit, learning_rate=1e-2, epochs=40))
    code = train.main([str(path)])
    assert seen == [("torch", "test/clm", None, None)]
    assert code == 0
    assert json.loads((tmp_path / "my.heads" / "training.json").read_text())["clm_backend"] == "torch"


def test_the_optimizer_steps_as_mlx_adamw_does():
    # MLX's AdamW, which the heads were first trained with, decays first and skips Adam's bias correction
    param = nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = train.MlxAdamW([param], learning_rate=0.1, weight_decay=0.01)
    grads = [torch.tensor([0.5, -1.0]), torch.tensor([0.2, 0.3])]
    expected, m, v = np.array([1.0, -2.0]), np.zeros(2), np.zeros(2)
    for grad in grads:
        param.grad = grad.clone()
        optimizer.step()
        g = grad.numpy().astype(np.float64)
        expected = expected * (1 - 0.1 * 0.01)
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
        expected = expected - 0.1 * m / (np.sqrt(v) + 1e-8)
    np.testing.assert_allclose(param.detach().numpy(), expected, rtol=1e-5)
