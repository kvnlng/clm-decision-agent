"""Tests for registry.py: the server.json that lists one deployment of the server in an MCP registry."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import json
from pathlib import Path

import pytest

import registry

TARIFF = Path(__file__).parent.parent / "examples" / "tariff" / "environment.toml"
URL = "https://decisions.example.org/mcp"


def entry(**overrides):
    settings = {"name": "io.github.kvnlng/tariff", "url": URL, "version": "0.1.0", "auth": "key"}
    settings.update(overrides)
    return registry.server_json(str(TARIFF), **settings)


def test_a_key_server_needs_the_authorization_header():
    server = entry()
    assert server["$schema"] == registry.SCHEMA
    assert (server["name"], server["version"]) == ("io.github.kvnlng/tariff", "0.1.0")
    assert server["remotes"] == [{"type": "streamable-http", "url": URL, "headers": [
        {"name": "Authorization", "description": "Bearer <the server's key>", "isRequired": True,
         "isSecret": True}]}]
    assert "packages" not in server


def test_an_oauth_server_names_no_header():
    # A client finds the issuer through the server's protected resource metadata, not the registry
    assert entry(auth="oauth")["remotes"] == [{"type": "streamable-http", "url": URL}]


def test_the_description_defaults_to_the_environments_name():
    assert entry()["description"] == "Component sourcing"
    assert entry(description="Sourcing decisions for Product Line X")["description"] == \
        "Sourcing decisions for Product Line X"


def test_the_repository_is_named_when_given():
    assert entry(repository="https://github.com/kvnlng/clm-decision-agent")["repository"] == \
        {"url": "https://github.com/kvnlng/clm-decision-agent", "source": "github"}


def test_a_clm_scenario_needs_a_description(write_scenario):
    with pytest.raises(ValueError, match="^--description: a CLM scenario has none of its own"):
        registry.server_json(str(write_scenario()), "io.github.kvnlng/desk", URL, "0.1.0", "key")
    server = registry.server_json(str(write_scenario()), "io.github.kvnlng/desk", URL, "0.1.0", "key",
                                  description="Routes service desk requests")
    assert server["description"] == "Routes service desk requests"


@pytest.mark.parametrize("overrides, message", [
    ({"name": "tariff"}, "^name: must be namespace/name"),
    ({"name": "io.github.kvnlng/"}, "^name: must be namespace/name"),
    ({"description": "x" * 101}, "^--description: at most 100 characters"),
    ({"version": "latest"}, "^--version: a version such as 1.0.0"),
    ({"url": "http://decisions.example.org/mcp"}, "^--url: must be https"),
    ({"url": "https://decisions.example.org/"}, "^--url: the server answers at /mcp"),
    ({"auth": "none"}, "^--auth: key or oauth"),
])
def test_bad_settings_are_refused(overrides, message):
    with pytest.raises(ValueError, match=message):
        entry(**overrides)


@pytest.mark.parametrize("url", ["https://localhost:8443/mcp", "https://127.0.0.1/mcp", "https://[::1]/mcp",
                                 "https://decisions.localhost/mcp"])
def test_local_addresses_are_refused(url):
    with pytest.raises(ValueError, match="^--url: name the deployment's host, not localhost or an IP address"):
        entry(url=url)


def test_an_internal_hostname_is_accepted():
    assert entry(url="https://decisions.corp.internal/mcp")["remotes"][0]["url"] == \
        "https://decisions.corp.internal/mcp"


def test_main_writes_the_file(tmp_path, capsys):
    out = tmp_path / "server.json"
    assert registry.main([str(TARIFF), "--name", "io.github.kvnlng/tariff", "--url", URL, "--version", "0.1.0",
                          "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["name"] == "io.github.kvnlng/tariff"
    assert str(out) in capsys.readouterr().err


def test_main_reports_a_bad_setting_in_one_line(capsys):
    assert registry.main([str(TARIFF), "--name", "tariff", "--url", URL, "--version", "0.1.0"]) == 2
    assert capsys.readouterr().err.startswith("error: name: must be namespace/name")
