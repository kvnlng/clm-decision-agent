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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import generator
from generator import (NO_TOOL_REPLY, GeneratorError, MLXGenerator, OllamaGenerator, OpenAIGenerator, finish_reply,
                       make_generator, parse_reply, to_chat_dicts)
from scenario import load_scenario


@tool
def lookup_invoice(invoice_number: str) -> dict:
    """Look up an invoice and its charges by invoice number."""
    return {"invoice_number": invoice_number}


class FakeTokenizer:
    def apply_chat_template(self, chat, **kwargs):
        self.chat, self.kwargs = chat, kwargs
        return "PROMPT"


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error
        self.bound = None

    def bind_tools(self, tools):
        self.bound = tools
        return self

    def invoke(self, messages):
        self.seen = messages
        if self.error:
            raise self.error
        return self.reply


def test_to_chat_dicts_maps_roles():
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    assert to_chat_dicts(messages) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_to_chat_dicts_rejects_unknown_message_types():
    with pytest.raises(TypeError, match="SystemMessage"):
        to_chat_dicts([SystemMessage(content="x")])


def test_mlx_prompt_starts_with_system_message():
    tokenizer = FakeTokenizer()
    gen = MLXGenerator("unused", 50, loaded=(None, tokenizer))
    assert gen.build_prompt("Be brief.", [HumanMessage(content="hi")]) == "PROMPT"
    assert tokenizer.chat == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
    ]
    assert tokenizer.kwargs["add_generation_prompt"] is True
    assert tokenizer.kwargs["tokenize"] is False


def test_ollama_generate_returns_stripped_text():
    llm = FakeLLM(reply=AIMessage(content="  hello  "))
    reply = OllamaGenerator("m", 50, llm=llm).generate("Be brief.", [HumanMessage(content="hi")])
    assert reply.content == "hello"
    assert isinstance(llm.seen[0], SystemMessage) and llm.seen[0].content == "Be brief."


def test_ollama_unreachable_gives_actionable_error():
    llm = FakeLLM(error=httpx.ConnectError("refused"))
    with pytest.raises(GeneratorError, match="ollama serve"):
        OllamaGenerator("m", 50, llm=llm).generate("s", [HumanMessage(content="hi")])


def test_make_generator_picks_backend(scenario, monkeypatch):
    made = []
    monkeypatch.setattr(generator, "MLXGenerator", lambda model, max_tokens: made.append(("mlx", model, max_tokens)))
    monkeypatch.setattr(generator, "OllamaGenerator",
                        lambda model, max_tokens: made.append(("ollama", model, max_tokens)))
    make_generator(scenario)
    assert made == [("mlx", "test/gen", 50)]


def test_parse_reply_with_closing_tag():
    reply = parse_reply(
        '<tool_call>\n{"name": "lookup_invoice", "arguments": {"invoice_number": "INV-1"}}\n</tool_call>')
    assert reply.content == ""
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert (call["name"], call["args"]) == ("lookup_invoice", {"invoice_number": "INV-1"})
    assert call["id"].startswith("call_")


def test_parse_reply_without_closing_tag():
    reply = parse_reply('<tool_call>\n{"name": "lookup_invoice", "arguments": {"invoice_number": "INV-1"}}')
    assert reply.tool_calls[0]["args"] == {"invoice_number": "INV-1"}


def test_parse_reply_keeps_text_before_the_call_and_reads_two_calls():
    text = ('Let me check. <tool_call>{"name": "a", "arguments": {}}</tool_call>\n'
            '<tool_call>{"name": "b", "arguments": {"k": "}"}}</tool_call>')
    reply = parse_reply(text)
    assert reply.content == "Let me check."
    assert [c["name"] for c in reply.tool_calls] == ["a", "b"]
    assert reply.tool_calls[0]["id"] != reply.tool_calls[1]["id"]


@pytest.mark.parametrize("text", [
    "Just a normal answer.",
    '<tool_call>\n{"name": "lookup_invoice", "arguments": {invoice_number: INV-1}}\n</tool_call>',
    '<tool_call>{"arguments": {}}</tool_call>',
    '<tool_call>{"name": "a", "arguments": "INV-1"}</tool_call>',
    '<tool_call>\n{"name": "lookup_invoice", "argu',
])
def test_parse_reply_treats_anything_else_as_plain_text(text):
    reply = parse_reply(text)
    assert reply.tool_calls == []
    assert reply.content == text


def test_to_chat_dicts_converts_tool_traffic():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "lookup_invoice", "args": {"invoice_number": "INV-1"}, "id": "c1"}]),
        ToolMessage(content='{"charges": []}', tool_call_id="c1", name="lookup_invoice"),
    ]
    assert to_chat_dicts(messages) == [
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "lookup_invoice", "arguments": {"invoice_number": "INV-1"}}}]},
        {"role": "tool", "name": "lookup_invoice", "content": '{"charges": []}'},
    ]


def test_mlx_prompt_passes_tool_schemas_only_when_tools_are_given():
    tokenizer = FakeTokenizer()
    gen = MLXGenerator("unused", 50, loaded=(None, tokenizer))
    gen.build_prompt("s", [HumanMessage(content="hi")])
    assert tokenizer.kwargs["tools"] is None
    gen.build_prompt("s", [HumanMessage(content="hi")], tools=[lookup_invoice])
    schema = tokenizer.kwargs["tools"][0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "lookup_invoice"
    assert schema["function"]["parameters"]["required"] == ["invoice_number"]


def test_ollama_binds_tools_and_fills_missing_call_ids():
    call = {"name": "lookup_invoice", "args": {"invoice_number": "INV-1"}, "id": None}
    llm = FakeLLM(reply=AIMessage(content="", tool_calls=[call]))
    reply = OllamaGenerator("m", 50, llm=llm).generate("s", [HumanMessage(content="hi")], tools=[lookup_invoice])
    assert llm.bound == [lookup_invoice]
    assert reply.tool_calls[0]["name"] == "lookup_invoice"
    assert reply.tool_calls[0]["id"]


def test_ollama_without_tools_does_not_bind():
    llm = FakeLLM(reply=AIMessage(content="hello"))
    OllamaGenerator("m", 50, llm=llm).generate("s", [HumanMessage(content="hi")])
    assert llm.bound is None


def test_ollama_model_problem_gives_actionable_error():
    llm = FakeLLM(error=ollama.ResponseError("model 'm' not found", 404))
    with pytest.raises(GeneratorError, match="ollama pull m"):
        OllamaGenerator("m", 50, llm=llm).generate("s", [HumanMessage(content="hi")])


def test_ollama_empty_reply_without_tools_becomes_the_fallback():
    llm = FakeLLM(reply=AIMessage(content="", tool_calls=[{"name": "a", "args": {}, "id": "c"}]))
    reply = OllamaGenerator("m", 50, llm=llm).generate("s", [HumanMessage(content="hi")])
    assert (reply.content, reply.tool_calls) == (NO_TOOL_REPLY, [])


def test_parse_reply_accepts_a_bare_json_tool_call():
    reply = parse_reply('{"name": "lookup_invoice", "arguments": {"invoice_number": "INV-7"}}')
    assert reply.tool_calls[0]["args"] == {"invoice_number": "INV-7"}


@pytest.mark.parametrize("text", [
    '<tool_call>\n{"name": "lookup_invoice", "arguments": {"invoice_number": "INV-7"}}\n</tool_call>',
    '{"name": "lookup_invoice", "arguments": {"invoice_number": "INV-7"}}',
])
def test_a_tool_call_when_no_tools_were_offered_is_never_shown_as_the_reply(text):
    reply = finish_reply(text, offered=False)
    assert (reply.content, reply.tool_calls) == (NO_TOOL_REPLY, [])


def test_finish_reply_keeps_text_and_offered_tool_calls():
    assert finish_reply("  Hello.  ", offered=False).content == "Hello."
    offered = finish_reply('<tool_call>{"name": "a", "arguments": {}}</tool_call>', offered=True)
    assert offered.tool_calls[0]["name"] == "a"


def test_openai_backend_is_made_with_the_address_and_key(write_scenario, monkeypatch):
    text = BASE_TOML.replace('backend = "mlx"', 'backend = "openai"\nbase_url = "http://localhost:9/v1"\n'
                                                'api_key_env = "DEMO_GEN_KEY"')
    monkeypatch.setenv("DEMO_GEN_KEY", "sk-demo")
    made = make_generator(load_scenario(write_scenario(text)))
    assert isinstance(made, OpenAIGenerator)
    assert str(made.llm.openai_api_base) == "http://localhost:9/v1"
    assert made.llm.openai_api_key.get_secret_value() == "sk-demo"
    assert (made.llm.model_name, made.llm.max_tokens) == ("test/gen", 50)


def test_openai_backend_replies_and_binds_tools():
    call = {"name": "lookup_invoice", "args": {"invoice_number": "INV-1"}, "id": None}
    llm = FakeLLM(reply=AIMessage(content="", tool_calls=[call]))
    reply = OpenAIGenerator("m", 50, "http://x/v1", llm=llm).generate("s", [HumanMessage(content="hi")],
                                                                    tools=[lookup_invoice])
    assert llm.bound == [lookup_invoice] and reply.tool_calls[0]["id"]


def test_openai_backend_unreachable_names_the_address():
    llm = FakeLLM(error=openai.APIConnectionError(request=httpx.Request("POST", "http://x/v1")))
    with pytest.raises(GeneratorError, match="http://x/v1"):
        OpenAIGenerator("m", 50, "http://x/v1", llm=llm).generate("s", [HumanMessage(content="hi")])


@pytest.mark.parametrize("error", [ValueError("not JSON"), IndexError("no choices")])
def test_openai_backend_reports_a_reply_it_cannot_read(error):
    with pytest.raises(GeneratorError, match="could not be read"):
        OpenAIGenerator("m", 50, "http://x/v1", llm=FakeLLM(error=error)).generate("s", [HumanMessage(content="hi")])
