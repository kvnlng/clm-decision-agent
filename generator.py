"""The language model synthesize.py asks for example messages: text in, text out.

Three backends: MLX in this process (an Apple Silicon Mac), a local Ollama server, or any server with
the OpenAI chat interface (vLLM, LM Studio, llama.cpp, Ollama ...), which suits a Linux machine.
"""
from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage

from scenario import Scenario


class GeneratorError(RuntimeError):
    """The generator backend could not produce a reply."""


class MLXGenerator:
    """Runs in the same process. `loaded` lets tests pass a (model, tokenizer) pair."""

    def __init__(self, model_id: str, max_tokens: int, loaded: tuple | None = None):
        if loaded is None:
            from mlx_lm import load  # pylint: disable=import-outside-toplevel,import-error
            loaded = load(model_id)
        self.model, self.tokenizer = loaded[0], loaded[1]
        self.max_tokens = max_tokens

    def build_prompt(self, system: str, user: str) -> str:
        """The full prompt text, in the model's chat template."""
        chat = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

    def generate(self, system: str, user: str) -> str:
        """The model's reply."""
        from mlx_lm import generate  # pylint: disable=import-outside-toplevel,import-error
        return generate(self.model, self.tokenizer, prompt=self.build_prompt(system, user),
                        max_tokens=self.max_tokens, verbose=False).strip()


class OllamaGenerator:
    """Talks to a local Ollama server through langchain-ollama."""

    def __init__(self, model: str, max_tokens: int, llm=None):
        if llm is None:
            from langchain_ollama import ChatOllama  # pylint: disable=import-outside-toplevel
            # reasoning=False: a thinking model would otherwise spend the token budget before replying
            llm = ChatOllama(model=model, num_predict=max_tokens, reasoning=False)
        self.llm, self.model = llm, model

    def generate(self, system: str, user: str) -> str:
        """The model's reply."""
        import httpx  # pylint: disable=import-outside-toplevel
        import ollama  # pylint: disable=import-outside-toplevel
        try:
            reply = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except (httpx.ConnectError, ConnectionError) as error:
            raise GeneratorError("Cannot reach Ollama. Start it with `ollama serve` and try again.") from error
        except ollama.ResponseError as error:
            raise GeneratorError(f"Ollama could not run '{self.model}': {error}. "
                                 f"If the model is missing, run `ollama pull {self.model}`.") from error
        return str(reply.content).strip()


class OpenAIGenerator:
    """Talks to any server that offers the OpenAI chat interface (Ollama, vLLM, LM Studio, llama.cpp, ...)."""

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def __init__(self, model: str, max_tokens: int, base_url: str, api_key: str | None = None, llm=None):
        if llm is None:
            from langchain_openai import ChatOpenAI  # pylint: disable=import-outside-toplevel
            # Local servers ignore the key, but the client insists on one
            llm = ChatOpenAI(model=model, max_tokens=max_tokens, base_url=base_url, api_key=api_key or "not-needed",
                             timeout=120, max_retries=1)
        self.llm, self.model, self.base_url = llm, model, base_url

    def generate(self, system: str, user: str) -> str:
        """The model's reply."""
        import openai  # pylint: disable=import-outside-toplevel
        try:
            reply = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except openai.APIConnectionError as error:
            raise GeneratorError(f"Cannot reach the model server at {self.base_url}.") from error
        except openai.APIStatusError as error:
            # The server's message is not repeated: it may echo the request, key included
            raise GeneratorError(f"The model server at {self.base_url} refused the request for '{self.model}' "
                                 f"(HTTP {error.status_code}). Check the model name and api_key_env.") from error
        except (ValueError, IndexError, KeyError, TypeError) as error:
            raise GeneratorError(f"The model server at {self.base_url} sent a reply that could not be read "
                                 f"({type(error).__name__}).") from error
        return str(reply.content).strip()


def make_generator(scenario: Scenario):
    """The generator backend named in the scenario."""
    if scenario.generator_backend == "openai":
        key = os.environ.get(scenario.generator_key_env or "", "") or None
        return OpenAIGenerator(scenario.generator_model, scenario.max_tokens, scenario.generator_base_url, key)
    if scenario.generator_backend == "ollama":
        return OllamaGenerator(scenario.generator_model, scenario.max_tokens)
    return MLXGenerator(scenario.generator_model, scenario.max_tokens)
