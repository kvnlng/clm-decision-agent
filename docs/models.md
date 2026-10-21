# Choosing and pointing at models

A scenario names two models: CLM, which the decision server uses for every decision, and a
language model, which only `synthesize.py` uses to write training examples. Both can run in
the same process or on a server.

## CLM

CLM is a frozen encoder (Qwen3-8B) with two small "heads" on top. The encoder turns text into
a vector; one head projects the situation, the other projects each option, and the answer is
whichever option lands closest.

### In this process (the default)

```toml
[clm]
model_id = "mlx-community/CLM-v0.1-8B-MLX-8bit"
heads = "scenario.heads"     # optional: heads trained for this scenario
```

This uses the community MLX builds, which run on Apple Silicon.

| Build | Download | Agreement with the original | Use it when |
|---|---|---|---|
| `...-MLX-8bit` | 8 GB | 99.0% of top answers | You want CLM as its authors released it |
| `...-MLX-4bit` | smaller | 91.4% | Memory is tight and approximate answers are acceptable |

- "8-bit" and "4-bit" describe how much the encoder's numbers were compressed. The 8-bit build
  keeps the released heads unchanged. The 4-bit build's heads were re-fitted to a more
  compressed encoder, so its answers differ more.
- The agreement figures are the build publishers' own.
- **Heads only work with the encoder build they were made for.** Heads trained with the 8-bit
  build must be used with the 8-bit build; the server checks this.

### On the CLM authors' server

```toml
[clm]
backend = "server"
url = "http://localhost:8700"
api_key_env = "CLM_API_KEY"      # optional: the variable holding the server's key
model = "clm-latest"
```

The authors' `clm-serve` needs a GPU machine. This backend follows their server code but
**has not been run against a real server**. Two things differ from the local backend:

- A corrected route is reused only when the request text is exactly the same, because
  the server returns answers and not the vectors that "near-identical" matching needs.
- `train.py` does not apply. Train and load heads on the server with the authors' tools.

## The language model

It writes example messages for training and is not used while the server runs. Any model that
follows instructions will do.

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
