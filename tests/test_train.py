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
from scenario import load_scenario
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
        # Each cue in the text adds its own direction, so a route's and a question's cues can sit together
        rows = []
        for text in texts:
            found = [self.centres[name] for name in self.names if name in text]
            rows.append((sum(found) if found else self.rng.normal(size=8)) + self.rng.normal(size=8))
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
    rows = [{"text": f"{scenario.routes[name].description} case {i}", "route": name, "answers": {}, "facts": [],
             "options": None, "option": None} for i, name in enumerate(names * 40)]
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
    monkeypatch.setattr(train, "fit_tasks", functools.partial(train.fit_tasks, learning_rate=1e-2, epochs=40))
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
    np.testing.assert_allclose(param.data.numpy(), expected, rtol=1e-5)


def test_a_bad_label_stops_training_naming_the_line(write_scenario, capsys):
    from conftest import BASE_TOML  # pylint: disable=import-outside-toplevel
    path = write_scenario(BASE_TOML.replace('model_id = "test/clm"', 'model_id = "test/clm"\nheads = "my.heads"'))
    lines = [{"text": "charged twice", "route": "Billing"}, {"text": "app down", "route": "Tech",
                                                           "answers": {"mood": True}}]
    path.with_name("scenario.examples.jsonl").write_text("\n".join(json.dumps(line) for line in lines))
    assert train.main([str(path)]) == 2
    assert capsys.readouterr().err.strip() == "error: scenario.examples.jsonl: line 2: unknown question 'mood'"


def row(text, route, **more):
    return {"text": text, "route": route, "answers": {}, "facts": [], "options": None, "option": None, **more}


def question_rows(scenario, count=120, urgent_share=2):
    """Rows whose urgency shows as the word URGENT; one in `urgent_share` is urgent (noul true is index 1)."""
    names = list(scenario.routes)
    made = []
    for i in range(count):
        name, urgent = names[i % 2], i % urgent_share == 0
        text = f"{scenario.routes[name].description} case {i}" + (" URGENT" if urgent else "")
        made.append(row(text, name, answers={"urgent": int(urgent)}))
    return made


def quiet_train(scenario, engine, rows, **kwargs):
    lines = []
    facts = train.train(scenario, engine, rows[:80], [], rows[80:], write=lines.append, **kwargs)
    return facts, "\n".join(lines)


def fast(monkeypatch, **more):
    # Tiny made-up heads need a faster rate than the real ones to move in a few passes
    monkeypatch.setattr(train, "fit_tasks", functools.partial(train.fit_tasks, **{"learning_rate": 1e-2,
                                                                                  "epochs": 40, **more}))


def test_a_labelled_question_is_trained_with_the_route(scenario, monkeypatch):
    fast(monkeypatch)
    engine = FakeEngine([route.description for route in scenario.routes.values()] + ["URGENT"])
    facts, _ = quiet_train(scenario, engine, question_rows(scenario))
    assert facts["trained"] == ["route", "urgent"]
    before, after = facts["per_class"]["urgent"]["before"], facts["per_class"]["urgent"]["after"]
    assert sum(c for c, _ in after.values()) > sum(c for c, _ in before.values())
    assert after["true"][0] / after["true"][1] >= 0.8 and after["false"][0] / after["false"][1] >= 0.8


def test_a_rare_class_is_weighted_up():
    weights = train.class_weights(torch.tensor([0] * 90 + [1] * 10), 3)
    assert weights.tolist() == pytest.approx([100 / (3 * 90), 100 / (3 * 10), 0.0])


def test_a_question_with_one_class_trains_and_reports_the_missing_class(scenario, monkeypatch):
    fast(monkeypatch)
    engine = FakeEngine([route.description for route in scenario.routes.values()] + ["URGENT"])
    rows = question_rows(scenario)
    for one in rows[:80]:
        one["answers"] = {"urgent": 0}                  # training never sees an urgent one
    facts, report = quiet_train(scenario, engine, rows)
    assert facts["per_class"]["urgent"]["after"]["true"][0] == 0
    assert "true" in report and "0 of" in report


def test_ranking_rows_train_against_their_own_options(write_scenario, monkeypatch):
    from conftest import BASE_TOML  # pylint: disable=import-outside-toplevel
    # Ranking among options made of random directions takes the tiny heads longer to learn
    fast(monkeypatch, learning_rate=3e-2, epochs=200, patience=50)
    scenario = load_scenario(write_scenario(BASE_TOML.replace(
        'inform = ["lookup"]\n', 'inform = ["lookup", "alternatives"]\n\n[routes.Billing.options]\n'
        'tool = "alternatives"\ninstructions = "Which refund suits them?"\n')))
    engine = FakeEngine(["Charges and refunds", "Bugs and outages", "ALPHA", "BETA", "GAMMA"])
    rows = []
    for i in range(120):
        cues = ["ALPHA", "BETA", "GAMMA"][i % 3:] + ["ALPHA", "BETA", "GAMMA"][:i % 3]
        wanted = cues[i % 2]
        options = [{"id": cue.lower(), "text": f"{cue} refund"} for cue in cues]
        rows.append(row(f"Charges and refunds case {i} {wanted}", "Billing", options=options,
                        option=cues.index(wanted)))
    facts, report = quiet_train(scenario, engine, rows)
    assert facts["options_trained"] == ["Billing"]
    right, total = facts["per_class"]["options of Billing"]["after"]["right option"]
    assert right / total >= 0.8
    assert "options of Billing" in report


def test_reports_are_per_class(scenario, monkeypatch):
    fast(monkeypatch)
    engine = FakeEngine([route.description for route in scenario.routes.values()] + ["URGENT"])
    _, report = quiet_train(scenario, engine, question_rows(scenario))
    for name in ("Billing", "Tech", "true", "false"):
        assert any(line.split()[:1] == [name] and " of " in line for line in report.splitlines())


def test_training_records_each_trained_questions_wording(scenario, monkeypatch):
    fast(monkeypatch)
    engine = FakeEngine([route.description for route in scenario.routes.values()] + ["URGENT"])
    facts, _ = quiet_train(scenario, engine, question_rows(scenario))
    assert facts["questions_sha"] == {"urgent": scenario.question_sha("urgent")}


def test_a_question_labelled_only_by_corrections_trains(scenario, monkeypatch):
    fast(monkeypatch)
    engine = FakeEngine([route.description for route in scenario.routes.values()] + ["URGENT"])
    rows = question_rows(scenario)
    plain = [dict(one, answers={}) for one in rows]
    lines = []
    facts = train.train(scenario, engine, plain[:60], rows[:20], plain[80:], write=lines.append)
    assert facts["trained"] == ["route", "urgent"]
    assert any("urgent" in line and "no held-back examples" in line for line in lines)
    assert facts["per_class"]["urgent"] == {"before": {}, "after": {}}


def test_a_ranking_with_no_held_back_rows_trains(write_scenario, monkeypatch):
    from conftest import BASE_TOML  # pylint: disable=import-outside-toplevel
    fast(monkeypatch)
    scenario = load_scenario(write_scenario(BASE_TOML.replace(
        'inform = ["lookup"]\n', 'inform = ["lookup", "alternatives"]\n\n[routes.Billing.options]\n'
        'tool = "alternatives"\ninstructions = "Which refund suits them?"\n')))
    engine = FakeEngine(["Charges and refunds", "Bugs and outages", "ALPHA", "BETA"])
    options = [{"id": "a", "text": "ALPHA refund"}, {"id": "b", "text": "BETA refund"}]
    ranked = [row(f"Charges and refunds case {i} ALPHA", "Billing", options=options, option=0) for i in range(20)]
    plain = [row(f"{d} case {i}", r) for i, (r, d) in
             enumerate([("Billing", "Charges and refunds"), ("Tech", "Bugs and outages")] * 40)]
    facts = train.train(scenario, engine, ranked + plain[:40], [], plain[40:], write=lambda line: None)
    assert "options of Billing" in facts["trained"]
