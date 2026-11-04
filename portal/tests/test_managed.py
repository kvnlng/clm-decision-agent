"""Tests for the portal's own server and its jobs: the command and environment it builds, never a real process."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.server import MCPServer

from clm_portal import managed


class FakeProcess:
    """A started process that keeps running, or has already exited with `code` after writing `stderr`."""

    def __init__(self, argv, env, stderr_file, code=None, stderr=""):
        self.args, self.env, self.code = argv, env, code
        if stderr:
            stderr_file.write(stderr.encode())
            stderr_file.flush()
        self.stopped = False

    def poll(self):
        return self.code

    def terminate(self):
        self.stopped = True
        self.code = -15

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):  # pylint: disable=unused-argument
        return self.code


class FakePopen:
    """Records every start; each process exits at once with `code` when given one."""

    def __init__(self, code=None, stderr=""):
        self.started: list[FakeProcess] = []
        self.code, self.stderr = code, stderr

    def __call__(self, argv, *, env, stderr, **_):
        process = FakeProcess(argv, env, stderr, self.code, self.stderr)
        self.started.append(process)
        return process


def stand_in():
    server = MCPServer("stand-in")

    @server.tool()
    def describe() -> dict:
        """What the environment offers."""
        return {"name": "stand-in"}
    return server


def managed_server(popen, tmp_path):
    environment = tmp_path / "environment.toml"
    environment.write_text("[environment]\nname = \"x\"\n")
    return managed.ManagedServer(environment, popen=popen, connect=lambda url, key: stand_in(), wait=0.5)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_the_server_runs_on_loopback_on_a_free_port_with_its_own_key(tmp_path):
    popen = FakePopen()
    server = managed_server(popen, tmp_path)
    await server.start()
    argv = popen.started[0].args
    assert Path(argv[0]).name == "clm-server" and argv[1] == str(tmp_path / "environment.toml")
    assert argv[2:6] == ["--http", "--host", "127.0.0.1", "--port"]
    assert int(argv[6]) not in (0, 8000) and argv[7:] == ["--key-env", "CLM_PORTAL_SERVER_KEY"]
    assert server.error is None and [tool.name for tool in await server.tools()] == ["describe"]
    first = popen.started[0].env["CLM_PORTAL_SERVER_KEY"]
    await server.restart()
    assert popen.started[0].stopped and popen.started[1].env["CLM_PORTAL_SERVER_KEY"] != first


@pytest.mark.anyio
async def test_the_key_is_only_in_the_childs_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CLM_PORTAL_KEY", "the-portal-key-value")
    popen = FakePopen()
    await managed_server(popen, tmp_path).start()
    process = popen.started[0]
    key = process.env["CLM_PORTAL_SERVER_KEY"]
    assert len(key) >= 32 and key not in " ".join(process.args)
    assert "CLM_PORTAL_SERVER_KEY" not in os.environ and key not in os.environ.values()
    assert "CLM_PORTAL_KEY" not in process.env


@pytest.mark.anyio
async def test_testing_adds_admit_derived_as_the_contract_writes_it(tmp_path):
    popen = FakePopen()
    server = managed_server(popen, tmp_path)
    await server.start(testing=True)
    assert popen.started[0].args[-2:] == ["--admit-derived", "proposed,checked,approved"] and server.testing
    await server.restart(testing=False)
    assert "--admit-derived" not in popen.started[1].args and not server.testing


@pytest.mark.anyio
async def test_a_server_that_exits_at_start_reports_its_error(tmp_path):
    popen = FakePopen(code=2, stderr="error: environment.functions: cannot import 'x'\n")
    server = managed_server(popen, tmp_path)
    await server.start()
    assert "cannot import 'x'" in server.error and not server.running


def recorder(release=None, code=0, output="done\n"):
    """A runner that records each command, waiting for `release` when given one."""
    seen = []

    def run(argv, **kwargs):
        seen.append((argv, kwargs))
        if release is not None:
            release.wait(5)
        return SimpleNamespace(returncode=code, stdout=output, stderr="")
    run.seen = seen
    return run


@pytest.mark.anyio
async def test_a_second_job_waits_for_the_first():
    release = threading.Event()
    jobs = managed.Jobs(runner=recorder(release), script=lambda name: Path("/bin") / name)
    first = await jobs.start("load", ["clm-load-documents", "a"])
    assert first.running and jobs.current() is first
    with pytest.raises(managed.Busy, match="wait"):
        await jobs.start("derive", ["clm-derive", "run"])
    release.set()
    await jobs.wait()
    assert not first.running and first.returncode == 0 and "done" in first.output and jobs.current() is None
    assert jobs.last() is first and len(jobs.runner.seen) == 1


@pytest.mark.anyio
async def test_a_successful_job_restarts_the_server(tmp_path):
    popen = FakePopen()
    server = managed_server(popen, tmp_path)
    await server.start()
    jobs = managed.Jobs(runner=recorder(), script=lambda name: Path("/bin") / name)
    await jobs.start("load", ["clm-load-documents", "a"], on_success=server.restart)
    await jobs.wait()
    assert len(popen.started) == 2
    failing = managed.Jobs(runner=recorder(code=1), script=lambda name: Path("/bin") / name)
    await failing.start("load", ["clm-load-documents", "a"], on_success=server.restart)
    await failing.wait()
    assert len(popen.started) == 2 and failing.last().returncode == 1


@pytest.mark.anyio
async def test_a_missing_clm_derive_is_reported_not_raised():
    jobs = managed.Jobs(runner=recorder(), script=lambda name: None)
    job = await jobs.start("derive", ["clm-derive", "run", "env.toml", "handbook"])
    assert not job.running and "clm-derive is not installed" in job.output and jobs.current() is None
    assert not jobs.runner.seen


@pytest.mark.anyio
async def test_a_jobs_output_never_shows_a_secret(monkeypatch):
    monkeypatch.setenv("CLM_DOCUMENTS_PASSWORD", "a-long-database-password")
    jobs = managed.Jobs(runner=recorder(output="connecting with a-long-database-password\n"),
                        script=lambda name: Path("/bin") / name)
    job = await jobs.start("load", ["clm-load-documents", "a"])
    await jobs.wait()
    assert "a-long-database-password" not in job.output and "CLM_DOCUMENTS_PASSWORD" in jobs.runner.seen[0][1]["env"]


def test_script_finds_a_console_script_beside_the_python():
    assert managed.script("clm-server") is not None and managed.script("no-such-script") is None


@pytest.mark.anyio
async def test_a_missing_clm_server_is_reported_and_nothing_else_is_started(tmp_path):
    popen = FakePopen()
    server = managed.ManagedServer(tmp_path / "environment.toml", popen=popen, connect=lambda url, key: stand_in(),
                                   find=lambda _: None)
    await server.start()
    assert not popen.started and "clm-server is not installed" in server.error and not server.running


@pytest.mark.anyio
async def test_a_job_is_running_until_the_server_has_restarted():
    seen = []
    jobs = managed.Jobs(runner=recorder(), script=lambda name: Path("/bin") / name)

    async def restart():
        seen.append(jobs.current() is not None and jobs.current().running)
    await jobs.start("load", ["clm-load-documents", "a"], on_success=restart)
    await jobs.wait()
    assert seen == [True] and jobs.current() is None
