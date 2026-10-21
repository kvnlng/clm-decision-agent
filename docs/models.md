# Choosing and pointing at models

A scenario names two models: CLM, which the decision server uses for every decision, and a
language model, which only `synthesize.py` uses to write training examples. Both can run in
the same process or on a server.

## CLM

CLM is a frozen encoder (Qwen3-8B) with two small "heads" on top. The encoder turns text into
a vector; one head projects the situation, the other projects each option, and the answer is
whichever option lands closest.

CLM always runs inside the server's process. `backend` under `[clm]` chooses how, to suit the
machine. The example scenario has both blocks, with one commented out.

### On a Mac with Apple Silicon: `backend = "mlx"` (the default)

```toml
[clm]
backend = "mlx"
model_id = "mlx-community/CLM-v0.1-8B-MLX-8bit"
heads = "scenario.heads"     # optional: heads trained for this scenario
```

This uses the community MLX builds of CLM, converted from the authors' release and checked against it.

| Build | Download | Agreement with the original | Use it when |
|---|---|---|---|
| `...-MLX-8bit` | 8 GB | 99.0% of top answers | You want CLM as its authors released it |
| `...-MLX-4bit` | smaller | 91.4% | Memory is tight and approximate answers are acceptable |

- "8-bit" and "4-bit" describe how much the encoder's numbers were compressed. The 8-bit build
  keeps the released heads unchanged. The 4-bit build's heads were re-fitted to a more
  compressed encoder, so its answers differ more.
- The agreement figures are the build publishers' own.
- A Mac needs about 9 GB of free memory for the 8-bit build; 16 GB is the practical minimum.

### On Linux with an NVIDIA GPU: `backend = "torch"`

```toml
[clm]
backend = "torch"
model_id = "Contrastive-LM/CLM-v0.1-8B"
device = "cuda"              # optional: cuda, mps or cpu; the first one available otherwise
heads = "scenario.heads"     # optional: heads trained for this scenario
```

This runs the CLM authors' own release with PyTorch: their heads (`Contrastive-LM/CLM-v0.1-8B`)
and the encoder they were made for (`Qwen/Qwen3-8B`, at the revision the authors used), at full
precision. The first start downloads about 16 GB.

- **A GPU with 24 GB of memory or more.** The encoder takes about 17 GB. An A100 is plenty.
- **It also runs on a Mac** (`device = "mps"`) **and on a plain CPU**, slowly. That is how it
  was checked, as described below.
- `uv sync` installs PyTorch on every machine. On Linux the PyPI build includes CUDA 13, which
  needs NVIDIA driver 580 or later; `nvidia-smi` shows the driver version.

### What has been checked

| Check | Result |
|---|---|
| The torch backend on a Mac's GPU against the MLX 8-bit build, released heads, 81 requests from the example | Same route for 79; the largest difference in a route's probability was 0.06 |
| Training the example with the torch backend on a Mac | 55% → 98% on held-back examples; 15 of 17 hand-written messages |
| The torch backend on Linux (arm64), in Docker, on the CPU: the slow test through the server, released heads | Passes: 11 of 17 hand-written messages, in about 5 minutes on 2 CPUs. On the Mac's GPU the same test got 12; the difference is a near tie |
| An Intel Linux image (amd64) | Installs, with PyTorch's CUDA 13.0 build. CLM was not run in it |
| An NVIDIA GPU | **Not yet** |

### Heads belong to one backend

Heads trained with `train.py` record the backend and the CLM build they were trained with.
The MLX build's encoder is compressed and the torch backend's is not, so heads trained on one
do not fit the other. The server checks this, says so at start-up, and uses the released heads
until you run `train.py` on the machine you serve from. There is no conversion; training takes
under a minute.

The heads file itself has the same layout for both backends.

### Earlier settings

Earlier versions could ask the CLM authors' server (`clm-serve`) with `backend = "server"`,
`url` and `api_key_env`. The torch backend replaces it, and a scenario that still uses those
settings stops at start-up with a line pointing here.

## The language model

It writes example messages for training and is not used while the server runs. Any model that
follows instructions will do. The `mlx` backend needs a Mac; on Linux, use `ollama` or
`openai`.

```toml
[generator]
backend = "mlx"                                       # in this process
model = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
```

```toml
backend = "ollama"                                    # a local Ollama server
model = "gemma4:26b"
```

```toml
backend = "openai"                                    # any server with the OpenAI chat interface
base_url = "http://localhost:11434/v1"                # Ollama, vLLM, LM Studio, llama.cpp ...
model = "gemma4:26b"
api_key_env = "MODEL_API_KEY"                         # optional
```

All three have been used to write the example's training messages. A larger model writes more
varied examples but takes longer; the small one is the default because it runs in this process.

`base_url` can point at a machine other than yours; see [data-handling.md](data-handling.md).
