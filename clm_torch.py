"""CLM with PyTorch: the authors' Qwen3-8B encoder and released heads, on an NVIDIA GPU, a Mac's GPU or a CPU.

It reproduces the authors' scoring, as the MLX build does: the encoder's hidden state after its final
norm at the last token, unit length; each head's projection of that, unit length; and a softmax over
`scale * cos / temperature`. The state and option texts come from the authors' schema (clm_schema.py).
"""
from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file

import clm_schema
from clm_common import HEADS_FILE, ClmError

RELEASED_HEADS_FILE = "CLM_v0.1-8B.pt"
# The encoder revisions the authors' heads were released for; the MLX build was made from the same one
ENCODER_REVISIONS = {"Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218"}
ACTIVATIONS = {"gelu": torch.nn.GELU, "relu": torch.nn.ReLU, "silu": torch.nn.SiLU}


def pick_device(wanted: str | None) -> str:
    """The device CLM runs on: the one asked for, or else the first of cuda, mps and cpu this machine has."""
    available = [name for name, present in (("cuda", torch.cuda.is_available()),
                                             ("mps", torch.backends.mps.is_available())) if present] + ["cpu"]
    if wanted is None:
        return available[0]
    if wanted not in available:
        raise ClmError(f"CLM was set to run on {wanted}, which this machine does not have "
                       f"(available: {', '.join(available)})")
    return wanted


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


class Head(torch.nn.Module):
    """One projection head, laid out as the authors' make_head so its weights load by name."""

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def __init__(self, hidden: int, width: int, depth: int, proj: int, activation: str = "gelu",
                 layernorm: bool = False, residual: bool = False):
        super().__init__()
        self.inp = torch.nn.Linear(hidden, width)
        self.hidden = torch.nn.ModuleList(torch.nn.Linear(width, width) for _ in range(depth - 2))
        self.norms = torch.nn.ModuleList(torch.nn.LayerNorm(width, eps=1e-5) for _ in range(depth - 2)) \
            if layernorm else torch.nn.ModuleList()
        self.out = torch.nn.Linear(width, proj)
        self.act = ACTIVATIONS[activation]()
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        x = self.act(self.inp(x))
        for i, linear in enumerate(self.hidden):
            h = linear(x)
            if len(self.norms):
                h = self.norms[i](h)
            h = self.act(h)
            x = x + h if self.residual else h
        return self.out(x)

    @staticmethod
    def from_config(config: dict) -> "Head":
        """A head shaped as a heads config.json (or a released checkpoint's cfg) describes."""
        cfg = config["cfg"]
        return Head(hidden=config.get("hidden_size", cfg.get("hidden_size", 4096)), width=cfg["width"],
                    depth=cfg["depth"], proj=config.get("projection_dim", cfg.get("projection_dim", 512)),
                    activation=cfg.get("activation", "gelu"), layernorm=cfg.get("layernorm", False),
                    residual=cfg.get("residual", False))


class TorchHeads:
    """The state head, the action head and the score scale, held in float32."""

    def __init__(self, weights: dict[str, np.ndarray], config: dict, device: str = "cpu"):
        self.config, self.device = config, device
        self.state, self.action = Head.from_config(config).to(device), Head.from_config(config).to(device)
        self.state.eval()
        self.action.eval()
        self.scale = 0.0
        self.load(weights)

    def load(self, weights: dict[str, np.ndarray]) -> None:
        """Replace the weights: `state_head.*`, `action_head.*` and `logit_scale`."""
        for prefix, head in (("state_head", self.state), ("action_head", self.action)):
            head.load_state_dict({name[len(prefix) + 1:]: torch.as_tensor(np.asarray(value, dtype=np.float32))
                                  for name, value in weights.items() if name.startswith(prefix + ".")}, strict=True)
        logit_scale = float(np.asarray(weights["logit_scale"]).reshape(-1)[0])
        self.scale = min(math.exp(logit_scale), 100.0)        # the authors clamp the scale at 100

    def weights(self) -> dict[str, np.ndarray]:
        """The weights in the layout of a heads file."""
        weights = {"logit_scale": np.array([math.log(self.scale)], dtype=np.float32)}
        for prefix, head in (("state_head", self.state), ("action_head", self.action)):
            weights.update({f"{prefix}.{name}": value.detach().cpu().numpy().copy()
                            for name, value in head.state_dict().items()})
        return weights

    def project(self, vectors: np.ndarray, which: str) -> np.ndarray:
        """[n, hidden] encoder vectors -> [n, projection] unit-length projections, for "state" or "action"."""
        head = self.state if which == "state" else self.action
        with torch.inference_mode():
            x = torch.as_tensor(np.asarray(vectors, dtype=np.float32), device=self.device)
            return _unit(head(_unit(x))).cpu().numpy()

    @classmethod
    def released(cls, model_id: str, device: str = "cpu") -> "TorchHeads":
        """The authors' released heads from the Hugging Face repo `model_id`."""
        folder = snapshot_download(repo_id=model_id, allow_patterns=[RELEASED_HEADS_FILE])
        checkpoint = torch.load(os.path.join(folder, RELEASED_HEADS_FILE), map_location="cpu", weights_only=True)
        weights = {"logit_scale": np.array([float(checkpoint["logit_scale"])], dtype=np.float32)}
        for prefix in ("state_head", "action_head"):
            weights.update({f"{prefix}.{name}": value.float().numpy() for name, value in checkpoint[prefix].items()})
        config = {"cfg": checkpoint["cfg"], "hidden_size": checkpoint.get("hidden_size", 4096),
                  "projection_dim": checkpoint.get("projection_dim", 512)}
        return cls(weights, config, device)

    @classmethod
    def trained(cls, folder, device: str = "cpu") -> "TorchHeads":
        """Heads written by train.py: the weights file and config.json in `folder`."""
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        return cls(load_file(str(folder / HEADS_FILE)), config, device)


class TorchEncoder:
    """The frozen encoder: text -> the last token's hidden state after the final norm, unit length."""

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def __init__(self, repo: str, revision: str | None, device: str, max_tokens: int = 2048,
                 batch_tokens: int = 4096, loaded: tuple | None = None):
        self.device, self.max_tokens, self.batch_tokens = device, max_tokens, batch_tokens
        if loaded is None:
            # pylint: disable-next=import-outside-toplevel
            from transformers import AutoModel, AutoTokenizer, logging as transformers_logging
            transformers_logging.set_verbosity_error()
            model = AutoModel.from_pretrained(repo, revision=revision, dtype=torch.bfloat16).to(device)
            model.eval()
            loaded = (model, AutoTokenizer.from_pretrained(repo, revision=revision))
        self.model, self.tokenizer = loaded

    def ids(self, text: str) -> list[int]:
        """Token ids, keeping the last `max_tokens`: the authors trained the heads on the end of long states."""
        ids = list(self.tokenizer(text)["input_ids"]) or list(self.tokenizer(" ")["input_ids"])
        return ids[-self.max_tokens:]

    def _batches(self, id_lists: list[list[int]]):
        """Groups of indexes into `id_lists`, shortest first, each within `batch_tokens` once padded."""
        order = sorted(range(len(id_lists)), key=lambda i: len(id_lists[i]))
        start = 0
        while start < len(order):
            end = start + 1            # in ascending length, the newest row is the longest in the batch
            while end < len(order) and (end - start + 1) * len(id_lists[order[end]]) <= self.batch_tokens:
                end += 1
            yield order[start:end]
            start = end

    def embed(self, texts: list[str]) -> np.ndarray:
        """[n, hidden] float32 unit vectors, in the order of `texts`."""
        id_lists = [self.ids(text) for text in texts]
        out: list = [None] * len(id_lists)
        for batch in self._batches(id_lists):
            rows = [id_lists[k] for k in batch]
            longest = len(rows[-1])
            # Padding goes after each text: attention is causal, so it cannot change a text's last token
            ids = torch.tensor([row + [0] * (longest - len(row)) for row in rows], device=self.device)
            mask = torch.tensor([[1] * len(row) + [0] * (longest - len(row)) for row in rows], device=self.device)
            last = torch.tensor([len(row) - 1 for row in rows], device=self.device)
            with torch.inference_mode():
                hidden = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
                vectors = _unit(hidden[torch.arange(len(rows), device=self.device), last].float()).cpu().numpy()
            for k, vector in zip(batch, vectors):
                out[k] = vector
        return np.stack(out) if out else np.zeros((0, 0), dtype=np.float32)


class TorchEngine:
    """CLM's answers to typed questions about a state, as the authors' Engine.answer gives them."""

    def __init__(self, encoder, heads: TorchHeads, cache_size: int = 50_000):
        self.encoder, self.heads, self.cache_size = encoder, heads, cache_size
        self.cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()

    @property
    def head_config(self) -> dict:
        """The heads' config.json contents."""
        return self.heads.config

    def embed(self, texts: list[str]) -> np.ndarray:
        """[n, hidden] encoder vectors, unit length."""
        return self.encoder.embed(texts)

    def project(self, vectors: np.ndarray, which: str) -> np.ndarray:
        """Encoder vectors through the "state" or "action" head."""
        return self.heads.project(vectors, which)

    def head_weights(self) -> dict[str, np.ndarray]:
        """The current heads' weights, in the layout of a heads file."""
        return self.heads.weights()

    def set_head_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Use these head weights from now on."""
        self.heads.load(weights)
        self.cache.clear()

    def _vectors(self, texts: list[str], which: str) -> np.ndarray:
        missing = list(dict.fromkeys(text for text in texts if (which, text) not in self.cache))
        if missing:
            for text, vector in zip(missing, self.project(self.embed(missing), which)):
                self.cache[(which, text)] = vector
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        for text in texts:
            self.cache.move_to_end((which, text))
        return np.stack([self.cache[(which, text)] for text in texts])

    def answer(self, state: Any, questions: dict[str, dict], temperature: float = 1.0) -> dict:
        """{"answers": {question id: answer}} in the authors' format."""
        if not questions:
            raise ValueError("questions must not be empty")
        pairs = clm_schema.build_pairs(state, questions)
        states = self._vectors([pair[0] for pair in pairs.values()], "state")
        actions = self._vectors([text for pair in pairs.values() for text in pair[2]], "action")
        answers, k = {}, 0
        for i, (qid, (_, keys, texts)) in enumerate(pairs.items()):
            cos = actions[k:k + len(texts)] @ states[i]
            k += len(texts)
            answers[qid] = clm_schema.answer_from_logits(questions[qid], keys,
                                                         (self.heads.scale * cos / temperature).tolist())
        return {"model": "clm-torch", "answers": answers}


def load_torch_engine(model_id: str, heads_dir=None, device: str | None = None) -> TorchEngine:
    """The authors' encoder and heads from Hugging Face, with heads from train.py in place of the released
    ones if `heads_dir` is given."""
    device = pick_device(device)
    released = TorchHeads.released(model_id, device)
    heads = TorchHeads.trained(heads_dir, device) if heads_dir else released
    repo = released.config["cfg"]["model"]
    return TorchEngine(TorchEncoder(repo, ENCODER_REVISIONS.get(repo), device), heads)
