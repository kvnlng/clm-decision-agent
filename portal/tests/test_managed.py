"""Tests for the portal's own server and its jobs: the command and environment it builds, never a real process."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import asyncio
import contextlib
import os
import threading
import time
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
    server = MCPServer("stand-in", instructions="Call describe first.")

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


@pytest.mark.anyio
async def test_a_failed_testing_start_leaves_the_next_restart_out_of_testing(tmp_path):
    popen = FakePopen(code=2, stderr="error: unrecognized arguments: --admit-derived\n")
    server = managed_server(popen, tmp_path)
    await server.start(testing=True)
    popen.code = None
    await server.restart()
    assert "--admit-derived" not in popen.started[1].args and server.running and not server.testing


@pytest.mark.anyio
async def test_a_server_that_dies_after_starting_says_why(tmp_path):
    popen = FakePopen()
    server = managed_server(popen, tmp_path)
    await server.start()
    server._stderr.write(b"error: the database connection was lost\n")  # pylint: disable=protected-access
    popen.started[0].code = 1
    assert not server.running and "database connection was lost" in server.error
    await server.restart()
    assert server.running and server.error is None


class SlowProcess(FakeProcess):
    """A process that takes a while to stop once told to."""

    def wait(self, timeout=None):
        time.sleep(0.5)
        return super().wait(timeout)


@pytest.mark.anyio
async def test_a_restart_waits_for_the_old_server_without_blocking_other_requests(tmp_path):
    server = managed_server(lambda argv, *, env, stderr, **_: SlowProcess(argv, env, stderr), tmp_path)
    await server.start()
    ticks = []

    async def tick():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.05)
    task = asyncio.create_task(tick())
    await server.restart()
    task.cancel()
    assert len(ticks) >= 5 and server.running


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
async def test_a_second_job_is_refused_while_the_first_runs():
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
async def test_a_job_can_start_once_the_last_one_finished():
    jobs = managed.Jobs(runner=recorder(), script=lambda name: Path("/bin") / name)
    first = await jobs.start("load", ["clm-load-documents", "a"])
    await jobs.wait()
    second = await jobs.start("derive", ["clm-derive", "run"])
    await jobs.wait()
    assert not first.running and not second.running and second.returncode == 0 and jobs.last() is second
    assert [argv[1:] for argv, _ in jobs.runner.seen] == [["a"], ["run"]] and not jobs.lock.locked()


@pytest.mark.anyio
async def test_prepare_runs_only_once_the_job_holds_the_lock():
    release = threading.Event()
    jobs = managed.Jobs(runner=recorder(release), script=lambda name: Path("/bin") / name)
    seen = []
    await jobs.start("load", ["clm-load-documents", "a"], prepare=lambda: seen.append(jobs.lock.locked()))
    with pytest.raises(managed.Busy):
        await jobs.start("load", ["clm-load-documents", "b"], prepare=lambda: seen.append("second"))
    release.set()
    await jobs.wait()
    missing = managed.Jobs(runner=recorder(), script=lambda name: None)
    await missing.start("load", ["clm-load-documents", "c"], prepare=lambda: seen.append("missing"))
    assert seen == [True]


@pytest.mark.anyio
async def test_a_failed_prepare_runs_no_job_and_frees_the_lock():
    jobs = managed.Jobs(runner=recorder(), script=lambda name: Path("/bin") / name)

    def fail():
        raise OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        await jobs.start("load", ["clm-load-documents", "a"], prepare=fail)
    assert not jobs.lock.locked() and jobs.current() is None and not jobs.runner.seen


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


NEW_SITE = Path(__file__).parents[2] / "examples" / "new_site" / "environment.toml"


@pytest.mark.anyio
@pytest.mark.skipif(managed.script("clm-server") is None or not NEW_SITE.is_file(),
                    reason="needs clm-server and examples/new_site")
async def test_the_portal_and_the_chat_reach_a_real_server_through_the_sdks_own_http_client(monkeypatch):
    from mcp import Client  # pylint: disable=import-outside-toplevel
    from clm_portal import chat  # pylint: disable=import-outside-toplevel
    server = managed.ManagedServer(NEW_SITE, wait=60)
    await server.start()
    try:
        assert server.running, server.error
        assert "describe" in [tool.name for tool in await server.tools()]
        monkeypatch.setenv(chat.KEY_ENV, server._key)  # pylint: disable=protected-access
        async with Client(chat._connect(server.url, [])) as client:  # pylint: disable=protected-access
            assert (await client.list_tools()).tools
        refused: list[int] = []
        monkeypatch.setenv(chat.KEY_ENV, "not-the-key")
        with pytest.raises(Exception):
            async with Client(chat._connect(server.url, refused)) as client:  # pylint: disable=protected-access
                await client.list_tools()
        assert refused and refused[0] == 401
    finally:
        server.stop()


@pytest.mark.anyio
async def test_the_portal_and_the_chat_give_the_sdk_its_own_http_client(monkeypatch):
    import httpx2  # pylint: disable=import-outside-toplevel
    from clm_decision import connection  # pylint: disable=import-outside-toplevel
    from clm_portal import chat  # pylint: disable=import-outside-toplevel
    seen = []

    @contextlib.asynccontextmanager
    async def streams(url, http_client):
        seen.append((url, http_client))
        yield "streams"
    monkeypatch.setattr(connection, "streamable_http_client", streams)
    async with managed.connect("http://127.0.0.1:1/mcp", "k") as found:
        assert found == "streams"
    async with chat._connect("http://127.0.0.1:1/mcp", []) as found:  # pylint: disable=protected-access
        assert found == "streams"
    assert len(seen) == 2 and all(isinstance(http, httpx2.AsyncClient) for _, http in seen)
    assert seen[0][1].timeout.read == 600 and seen[0][1].headers["Authorization"] == "Bearer k"
    assert not seen[1][1].follow_redirects
