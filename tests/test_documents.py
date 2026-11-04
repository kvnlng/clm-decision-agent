"""Tests for documents.py: collections declared in an environment, and where their store is."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import pytest
from conftest import ENV_TOML
from fake_documents import FakeEmbedder, FakeStore

from clm_decision.documents import DocumentSettings, document_settings, documents_password
from clm_decision.environment import load_environment
from clm_decision.scenario import ScenarioError

DOCUMENTS = '''
[documents]
backend = "postgres"
password_env = "TOY_DOCUMENTS_PASSWORD"

[collections.cases]
description = "Court opinions on non-competes"
embedding_model = "toy-embed"
filters = { state = "string", year = "integer", decided = "date" }
'''


def load(write_environment, toml=ENV_TOML + DOCUMENTS, **options):
    options.setdefault("store", FakeStore())
    options.setdefault("embedder", FakeEmbedder())
    return load_environment(write_environment(toml), **options)


def test_a_collection_is_read(write_environment):
    store = FakeStore()
    env = load(write_environment, store=store)
    cases = env.collections["cases"]
    assert cases.description == "Court opinions on non-competes" and cases.embedding_model == "toy-embed"
    assert cases.filters == {"state": "string", "year": "integer", "decided": "date"}
    assert env.store is store


def test_an_environment_without_documents_has_no_store(write_environment):
    env = load_environment(write_environment())
    assert not env.collections and env.store is None


def test_collections_need_a_documents_table(write_environment):
    toml = ENV_TOML + DOCUMENTS.replace('[documents]\nbackend = "postgres"\npassword_env = "TOY_DOCUMENTS_PASSWORD"\n',
                                        "")
    with pytest.raises(ScenarioError, match=r"^collections: need a \[documents\] table"):
        load(write_environment, toml)


def test_an_unknown_filter_type_stops_loading(write_environment):
    toml = ENV_TOML + DOCUMENTS.replace('year = "integer"', 'year = "whole"')
    with pytest.raises(ScenarioError, match=r"^collections\.cases\.filters\.year: type must be string, integer"):
        load(write_environment, toml)


def test_a_collection_needs_an_embedding_model(write_environment):
    toml = ENV_TOML + DOCUMENTS.replace('embedding_model = "toy-embed"\n', "")
    with pytest.raises(ScenarioError, match=r"^collections\.cases\.embedding_model: must be text"):
        load(write_environment, toml)


def test_only_postgres_is_a_backend(write_environment):
    toml = ENV_TOML + DOCUMENTS.replace('backend = "postgres"', 'backend = "sqlite"')
    with pytest.raises(ScenarioError, match=r"^documents\.backend: must be postgres"):
        load(write_environment, toml)


def test_settings_have_defaults():
    settings = document_settings({"documents": {}})
    assert settings == DocumentSettings("127.0.0.1", 5432, "clm_documents", "clm", "CLM_DOCUMENTS_PASSWORD",
                                        "http://127.0.0.1:11434")


def test_password_comes_from_the_named_variable(monkeypatch):
    settings = document_settings({"documents": {"password_env": "TOY_DOCUMENTS_PASSWORD"}})
    monkeypatch.delenv("TOY_DOCUMENTS_PASSWORD", raising=False)
    with pytest.raises(ScenarioError, match="^documents.password_env: set TOY_DOCUMENTS_PASSWORD"):
        documents_password(settings)
    monkeypatch.setenv("TOY_DOCUMENTS_PASSWORD", "secret")
    assert documents_password(settings) == "secret"
