"""Tests for generator.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
import httpx
import ollama
import openai
import pytest
from conftest import BASE_TOML
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import generator
from generator import GeneratorError, MLXGenerator, OllamaGenerator, OpenAIGenerator, make_generator
from scenario import load_scenario


class FakeTokenizer:
    def apply_chat_template(self, chat, **kwargs):
        self.chat, self.kwargs = chat, kwargs
        return "PROMPT"


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error

    def invoke(self, messages):
        self.seen = messages
        if self.error:
            raise self.error
        return self.reply


def test_generators_offer_text_in_and_text_out_only():
    for made in (MLXGenerator, OllamaGenerator, OpenAIGenerator):
        assert not hasattr(made, "bind_tools")
    assert not hasattr(generator, "finish_reply") and not hasattr(generator, "NO_TOOL_REPLY")


def test_mlx_prompt_is_the_system_and_user_text():
    tokenizer = FakeTokenizer()
    gen = MLXGenerator("unused", 50, loaded=(None, tokenizer))
    assert gen.build_prompt("Be brief.", "hi") == "PROMPT"
    assert tokenizer.chat == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
    ]
    assert tokenizer.kwargs["add_generation_prompt"] is True
    assert tokenizer.kwargs["tokenize"] is False
    assert "tools" not in tokenizer.kwargs


def test_ollama_generate_returns_stripped_text():
    llm = FakeLLM(reply=AIMessage(content="  hello  "))
    assert OllamaGenerator("m", 50, llm=llm).generate("Be brief.", "hi") == "hello"
    assert llm.seen == [SystemMessage(content="Be brief."), HumanMessage(content="hi")]


def test_ollama_unreachable_gives_actionable_error():
    llm = FakeLLM(error=httpx.ConnectError("refused"))
    with pytest.raises(GeneratorError, match="ollama serve"):
        OllamaGenerator("m", 50, llm=llm).generate("s", "hi")


def test_ollama_model_problem_gives_actionable_error():
    llm = FakeLLM(error=ollama.ResponseError("model 'm' not found", 404))
    with pytest.raises(GeneratorError, match="ollama pull m"):
        OllamaGenerator("m", 50, llm=llm).generate("s", "hi")


def test_make_generator_picks_backend(scenario, monkeypatch):
    made = []
    monkeypatch.setattr(generator, "MLXGenerator", lambda model, max_tokens: made.append(("mlx", model, max_tokens)))
    monkeypatch.setattr(generator, "OllamaGenerator",
                        lambda model, max_tokens: made.append(("ollama", model, max_tokens)))
    make_generator(scenario)
    assert made == [("mlx", "test/gen", 50)]


def test_openai_backend_is_made_with_the_address_and_key(write_scenario, monkeypatch):
    text = BASE_TOML.replace('backend = "mlx"', 'backend = "openai"\nbase_url = "http://localhost:9/v1"\n'
                                                'api_key_env = "DEMO_GEN_KEY"')
    monkeypatch.setenv("DEMO_GEN_KEY", "sk-demo")
    made = make_generator(load_scenario(write_scenario(text)))
    assert isinstance(made, OpenAIGenerator)
    assert str(made.llm.openai_api_base) == "http://localhost:9/v1"
    assert made.llm.openai_api_key.get_secret_value() == "sk-demo"
    assert (made.llm.model_name, made.llm.max_tokens) == ("test/gen", 50)


def test_openai_backend_returns_stripped_text():
    llm = FakeLLM(reply=AIMessage(content=" a list \n"))
    assert OpenAIGenerator("m", 50, "http://x/v1", llm=llm).generate("s", "hi") == "a list"


def test_openai_backend_unreachable_names_the_address():
    llm = FakeLLM(error=openai.APIConnectionError(request=httpx.Request("POST", "http://x/v1")))
    with pytest.raises(GeneratorError, match="http://x/v1"):
        OpenAIGenerator("m", 50, "http://x/v1", llm=llm).generate("s", "hi")


@pytest.mark.parametrize("error", [ValueError("not JSON"), IndexError("no choices")])
def test_openai_backend_reports_a_reply_it_cannot_read(error):
    with pytest.raises(GeneratorError, match="could not be read"):
        OpenAIGenerator("m", 50, "http://x/v1", llm=FakeLLM(error=error)).generate("s", "hi")
