"""Tests for clm_torch.py, with tiny made-up heads and no real model."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name,unused-argument
import json
import math
import sys

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors.numpy import save_file

import clm_schema
import clm_torch
from clm_router import HEADS_FILE, ClmError
from clm_torch import TorchEncoder, TorchEngine, TorchHeads, pick_device

HIDDEN, WIDTH, PROJ = 8, 6, 4
CONFIG = {"cfg": {"model": "test/encoder", "hidden_size": HIDDEN, "projection_dim": PROJ, "width": WIDTH,
                  "depth": 3, "activation": "gelu", "layernorm": True, "residual": False},
          "hidden_size": HIDDEN, "projection_dim": PROJ}


def tiny_weights(seed=0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = {"logit_scale": np.array([math.log(20.0)], dtype=np.float32)}
    for head in ("state_head", "action_head"):
        for name, shape in (("inp.weight", (WIDTH, HIDDEN)), ("inp.bias", (WIDTH,)),
                            ("hidden.0.weight", (WIDTH, WIDTH)), ("hidden.0.bias", (WIDTH,)),
                            ("norms.0.weight", (WIDTH,)), ("norms.0.bias", (WIDTH,)),
                            ("out.weight", (PROJ, WIDTH)), ("out.bias", (PROJ,))):
            weights[f"{head}.{name}"] = rng.normal(size=shape).astype(np.float32)
    return weights


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def reference_project(weights, vectors, head):
    """The authors' head in plain numpy: Linear, GELU, [Linear, LayerNorm, GELU], Linear, then unit length."""
    w = {k[len(head) + 1:]: v.astype(np.float64) for k, v in weights.items() if k.startswith(head + ".")}
    gelu = np.vectorize(lambda x: 0.5 * x * (1 + math.erf(x / math.sqrt(2))))
    x = gelu(unit(vectors.astype(np.float64)) @ w["inp.weight"].T + w["inp.bias"])
    h = x @ w["hidden.0.weight"].T + w["hidden.0.bias"]
    h = (h - h.mean(-1, keepdims=True)) / np.sqrt(h.var(-1, keepdims=True) + 1e-5) * w["norms.0.weight"] \
        + w["norms.0.bias"]
    return unit(gelu(h) @ w["out.weight"].T + w["out.bias"])


class FakeEncoder:
    """Fixed unit vectors per text, noting every text it was asked to embed."""

    def __init__(self):
        self.asked = []

    def embed(self, texts):
        self.asked.extend(texts)
        rows = [np.random.default_rng(abs(hash(t)) % 2**32).normal(size=HIDDEN) for t in texts]
        return unit(np.array(rows, dtype=np.float32))


def test_head_matches_a_numpy_reference():
    weights = tiny_weights()
    heads = TorchHeads(weights, CONFIG, "cpu")
    vectors = np.random.default_rng(1).normal(size=(3, HIDDEN)).astype(np.float32)
    for which, head in (("state", "state_head"), ("action", "action_head")):
        np.testing.assert_allclose(heads.project(vectors, which), reference_project(weights, vectors, head),
                                   atol=1e-5)
    assert heads.scale == pytest.approx(20.0)


def test_scale_is_capped_at_100():
    weights = tiny_weights() | {"logit_scale": np.array([math.log(500.0)], dtype=np.float32)}
    assert TorchHeads(weights, CONFIG, "cpu").scale == 100.0


def test_heads_round_trip_through_safetensors(tmp_path):
    heads = TorchHeads(tiny_weights(), CONFIG, "cpu")
    save_file(heads.weights(), str(tmp_path / HEADS_FILE))
    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    again = TorchHeads.trained(tmp_path, "cpu")
    vectors = np.random.default_rng(2).normal(size=(2, HIDDEN)).astype(np.float32)
    np.testing.assert_array_equal(again.project(vectors, "state"), heads.project(vectors, "state"))
    assert again.config == CONFIG


def test_answer_matches_upstream_maths_with_a_fake_encoder():
    encoder, heads = FakeEncoder(), TorchHeads(tiny_weights(), CONFIG, "cpu")
    engine = TorchEngine(encoder, heads)
    questions = {"team": {"type": "choice", "instructions": "Which team?",
                          "criteria": {"billing": "Charges", "tech": "Bugs"}}}
    reply = engine.answer("my invoice is wrong", questions)
    state, keys, texts = clm_schema.build_pairs("my invoice is wrong", questions)["team"]
    assert encoder.asked == [state, *texts]
    zs = heads.project(encoder.embed([state]), "state")[0]
    za = heads.project(encoder.embed(texts), "action")
    logits = heads.scale * (za @ zs)
    expected = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    got = reply["answers"]["team"]["probabilities"]
    assert [got[k] for k in keys] == pytest.approx(expected.tolist(), abs=1e-5)


def test_projections_are_cached_per_text():
    encoder = FakeEncoder()
    engine = TorchEngine(encoder, TorchHeads(tiny_weights(), CONFIG, "cpu"))
    question = {"q": {"type": "noul", "instructions": "Urgent?"}}
    engine.answer("a", question)
    asked = len(encoder.asked)
    engine.answer("a", question)
    assert len(encoder.asked) == asked


def test_set_head_weights_clears_the_cache():
    encoder = FakeEncoder()
    engine = TorchEngine(encoder, TorchHeads(tiny_weights(), CONFIG, "cpu"))
    question = {"q": {"type": "noul", "instructions": "Urgent?"}}
    before = engine.answer("a", question)["answers"]["q"]["noul"]
    engine.set_head_weights(tiny_weights(seed=5))
    after = engine.answer("a", question)["answers"]["q"]["noul"]
    assert after != before
    np.testing.assert_array_equal(engine.head_weights()["state_head.inp.weight"],
                                  tiny_weights(seed=5)["state_head.inp.weight"])
    assert engine.head_config == CONFIG


def test_pick_device_falls_back_and_explains(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device(None) == "mps"
    assert pick_device("cpu") == "cpu"
    with pytest.raises(ClmError, match=r"CLM was set to run on cuda, which this machine does not have "
                                       r"\(available: mps, cpu\)"):
        pick_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert pick_device(None) == "cuda"


class StubTokenizer:
    def __call__(self, text):
        return {"input_ids": [ord(c) for c in text]}


def test_empty_text_is_embedded_as_a_space():
    encoder = TorchEncoder("test/encoder", None, "cpu", loaded=(None, StubTokenizer()))
    assert encoder.ids("") == encoder.ids(" ") == [32]


def test_long_text_keeps_its_last_tokens():
    encoder = TorchEncoder("test/encoder", None, "cpu", max_tokens=3, loaded=(None, StubTokenizer()))
    assert encoder.ids("abcdef") == [ord("d"), ord("e"), ord("f")]


class TinyModel(torch.nn.Module):
    """Stands in for the encoder: each position's hidden state depends on the tokens up to it."""

    def forward(self, input_ids, attention_mask):  # pylint: disable=arguments-differ
        x = torch.nn.functional.one_hot(  # pylint: disable=not-callable
            input_ids % HIDDEN, HIDDEN).float().cumsum(dim=1)
        return type("Out", (), {"last_hidden_state": x.to(torch.bfloat16)})()


def test_padding_does_not_change_a_text_vector():
    encoder = TorchEncoder("test/encoder", None, "cpu", loaded=(TinyModel(), StubTokenizer()))
    alone = encoder.embed(["ab"])
    together = encoder.embed(["ab", "a much longer text"])
    np.testing.assert_allclose(together[0], alone[0], atol=1e-6)
    assert together.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(together, axis=1), 1.0, atol=1e-5)


def test_released_pt_is_loaded_with_weights_only(monkeypatch, tmp_path):
    seen = {}
    weights = tiny_weights()

    def fake_load(path, **kwargs):
        seen.update(kwargs)
        return {"state_head": {k[11:]: torch.from_numpy(v) for k, v in weights.items() if k.startswith("state_head.")},
                "action_head": {k[12:]: torch.from_numpy(v) for k, v in weights.items()
                                if k.startswith("action_head.")},
                "logit_scale": torch.tensor(float(weights["logit_scale"][0])), "cfg": CONFIG["cfg"],
                "hidden_size": HIDDEN, "projection_dim": PROJ}
    monkeypatch.setattr(clm_torch, "snapshot_download", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(torch, "load", fake_load)
    heads = TorchHeads.released("test/heads", "cpu")
    assert seen["weights_only"] is True
    vectors = np.random.default_rng(3).normal(size=(2, HIDDEN)).astype(np.float32)
    np.testing.assert_allclose(heads.project(vectors, "action"), reference_project(weights, vectors, "action_head"),
                               atol=1e-5)
    assert heads.config["cfg"] == CONFIG["cfg"]


def test_a_mlx_trained_heads_file_loads_in_torch(tmp_path):
    mx = pytest.importorskip("mlx.core")
    try:
        folder = snapshot_download(repo_id="mlx-community/CLM-v0.1-8B-MLX-8bit", local_files_only=True,
                                   allow_patterns=["clm_mlx/*"])
    except (OSError, ValueError):
        pytest.skip("the MLX build is not downloaded")
    sys.path.insert(0, folder)
    from clm_mlx.heads import HeadPair  # pylint: disable=import-outside-toplevel,import-error
    weights = tiny_weights()
    save_file(weights, str(tmp_path / HEADS_FILE))
    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    vectors = np.random.default_rng(4).normal(size=(2, HIDDEN)).astype(np.float32)
    in_mlx = np.array(HeadPair(str(tmp_path)).project(mx.array(vectors), "state"))
    np.testing.assert_allclose(TorchHeads.trained(tmp_path, "cpu").project(vectors, "state"), in_mlx, atol=1e-3)
