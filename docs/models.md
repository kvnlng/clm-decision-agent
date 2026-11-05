# Choosing and pointing at models

A scenario names two models: CLM, which the decision server uses for every decision, and a language model, which only
`clm-synthesize` uses to write training examples.

## CLM

CLM is a frozen encoder (Qwen3-8B) with two small "heads": the encoder turns text into a vector, one head projects the
situation, the other each option, and the closest option wins. It always runs inside the server's process; `backend`
under `[clm]` chooses how. The example scenario has both blocks, one commented out.

| Key | Backend | Default | Meaning |
|---|---|---|---|
| `backend` | | `"mlx"` | `"mlx"` on a Mac with Apple Silicon, `"torch"` on Linux with an NVIDIA GPU |
| `model_id` | both | `mlx-community/CLM-v0.1-8B-MLX-8bit` (mlx), `Contrastive-LM/CLM-v0.1-8B` (torch) | The build to download |
| `heads` | both | none: the released heads | A folder of heads trained for this scenario, relative to the scenario file's folder |
| `device` | torch only | the first of `cuda`, `mps`, `cpu` available | Where the encoder runs |

Earlier settings (`backend = "server"`, `url`, `api_key_env`) asked the CLM authors' server (`clm-serve`). The torch
backend replaces them; a scenario that still uses them stops at start-up with a line pointing here.

### On a Mac with Apple Silicon: `backend = "mlx"`

```toml
[clm]
backend = "mlx"
model_id = "mlx-community/CLM-v0.1-8B-MLX-8bit"
heads = "scenario.heads"
```

Community MLX builds of CLM, converted from the authors' release and checked against it. "8-bit" and "4-bit" say how
far the encoder's numbers were compressed; the agreement figures are the build publishers' own.

| Build | Download | Agreement with the original | Use it when |
|---|---|---|---|
| `...-MLX-8bit` | 8 GB | 99.0% of top answers | You want CLM as its authors released it; the released heads are unchanged |
| `...-MLX-4bit` | smaller | 91.4% | Memory is tight and approximate answers are acceptable; the heads were re-fitted to the more compressed encoder, so answers differ more |

A Mac needs about 9 GB of free memory for the 8-bit build; 16 GB is the practical minimum. An MLX build runs the
Python code (`clm_mlx`) in its model repo, inside the server's process; the two builds above are downloaded at a
fixed commit, so that code does not change. Any other `model_id` is taken as its repo has it now, and the server
warns so.

### On Linux with an NVIDIA GPU: `backend = "torch"`

```toml
[clm]
backend = "torch"
model_id = "Contrastive-LM/CLM-v0.1-8B"
device = "cuda"
heads = "scenario.heads"
```

The CLM authors' own release with PyTorch: their heads (`Contrastive-LM/CLM-v0.1-8B`) and the encoder they were made
for (`Qwen/Qwen3-8B`, at the authors' revision), at full precision. The first start downloads about 16 GB into
Hugging Face's cache (`HF_HOME`, by default `~/.cache/huggingface`).

- **A GPU with 24 GB of memory or more:** on an A100, CLM used at most 14.4 GB while deciding and 15.2 GB while
  training. On a cloud VM, take a whole GPU; some sizes give a share of one (Azure's NV6ads_A10_v5 has 4 GB of GPU
  memory, however much ordinary memory the VM lists). It also runs on a Mac (`device = "mps"`) and on a plain CPU,
  slowly, with about 24 GB of memory; that is how it was checked, below.
- `uv sync` installs PyTorch on every machine. On Linux the PyPI build includes CUDA 13, which needs NVIDIA driver 580
  or later (`nvidia-smi` shows the version) and no separate toolkit; without a GPU, CLM runs on the CPU. Linux also
  needs a C compiler and Python's headers (`sudo apt install build-essential python3.12-dev` on Ubuntu): on a GPU,
  PyTorch compiles a helper with Triton on CLM's first run, and stops with "Failed to find C compiler" or "Python.h:
  No such file" without them. Installing without a checkout: [serving.md](serving.md#on-linux).

### What has been checked

| Check | Result |
|---|---|
| Torch on a Mac's GPU against the MLX 8-bit build, released heads, 81 requests from the example | Same route for 79; largest difference in a route's probability 0.06 |
| Training the example with torch on a Mac | 55% → 98% on held-back examples; 15 of 17 hand-written messages |
| Torch on Linux (arm64), in Docker, on the CPU: the slow test through the server, released heads | Passes: 11 of 17 hand-written messages, in about 5 minutes on 2 CPUs. The Mac's GPU got 12; the difference is a near tie |
| An Intel Linux image (amd64) | Installs, with PyTorch's CUDA 13.0 build. CLM was not run in it |
| Torch on an NVIDIA A100 80GB (Azure NC24ads_A100_v4, Ubuntu, driver 580): the slow test through the server, released heads | Passes: 12 of 17, as on the Mac's GPU. About 40 ms a decision; the model loads in 7 seconds once downloaded. The fast tests pass (241, with 2 MLX-only ones skipped) |
| Torch in Docker on the A100 (an image since removed), with the GPU: the same slow test | Passes: 12 of 17 |
| Training the example on the A100 | 55% → 98% on held-back examples, kept after the same pass as on the Mac; 15 of 17 hand-written messages; 20 seconds, loading the model included |
| Torch on an NVIDIA H100 80GB SXM (DigitalOcean, Ubuntu 22.04, driver 580, Python 3.13): the CrowdStrike example, trained and replayed for the whole day | Training 2 min 40 s with the download; 76% on held-back examples as on the A100; 99,806 decisions at 51.1 a second |

### Heads belong to one backend

Heads trained with `clm-train` record the backend and CLM build they were trained with. The MLX build's encoder is
compressed and the torch backend's is not, so heads trained on one do not fit the other. The server checks this, says
so at start-up and uses the released heads until you run `clm-train` on the machine you serve from. There is no
conversion; training takes under a minute. The heads file has the same layout for both backends.

## The language model

It writes example messages for training and is not used while the server runs. Any model that follows instructions
will do. The `mlx` backend runs in this process and needs a Mac; on Linux, use `ollama` or `openai`. All three have
written the example's training messages; a larger model writes more varied examples but takes longer. The small one
is the default because it runs in this process.

```toml
[generator]
backend = "mlx"
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
api_key_env = "MODEL_API_KEY"                         # optional; if set, the variable must be too
```

`base_url` can point at a machine other than yours; see [data-handling.md](data-handling.md).
