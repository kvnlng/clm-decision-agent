"""Tests for the portal's jobs: one at a time, under the lock layer writes share, with what a child may see."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import contextlib
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from clm_decision.portal import managed
from .conftest import FakeServer


@pytest.fixture
def anyio_backend():
    return "asyncio"


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
async def test_a_job_is_running_until_the_server_has_reloaded():
    seen = []
    jobs = managed.Jobs(runner=recorder(), script=lambda name: Path("/bin") / name)

    async def reload():
        seen.append(jobs.current() is not None and jobs.current().running)
    await jobs.start("load", ["clm-load-documents", "a"], on_success=reload)
    await jobs.wait()
    assert seen == [True] and jobs.current() is None


@pytest.mark.anyio
async def test_a_successful_job_reloads_the_server():
    server = FakeServer()
    jobs = managed.Jobs(runner=recorder(), script=lambda name: Path("/bin") / name)
    await jobs.start("load", ["clm-load-documents", "a"], on_success=server.reload)
    await jobs.wait()
    assert server.reloads == [False]
    failing = managed.Jobs(runner=recorder(code=1), script=lambda name: Path("/bin") / name)
    await failing.start("load", ["clm-load-documents", "a"], on_success=server.reload)
    await failing.wait()
    assert server.reloads == [False] and failing.last().returncode == 1


@pytest.mark.anyio
async def test_jobs_lose_server_keys(monkeypatch):
    monkeypatch.setenv("CLM_PORTAL_KEY", "the-portal-key-value")
    monkeypatch.setenv("AGENTS_KEY", "the-agents-key-value")
    monkeypatch.setenv("CLM_DOCUMENTS_PASSWORD", "a-long-database-password")
    jobs = managed.Jobs(runner=recorder(), script=lambda name: Path("/bin") / name, withheld=("AGENTS_KEY",))
    await jobs.start("derive", ["clm-derive", "run"])
    await jobs.wait()
    env = jobs.runner.seen[0][1]["env"]
    assert "CLM_PORTAL_KEY" not in env and "AGENTS_KEY" not in env
    assert env["CLM_DOCUMENTS_PASSWORD"] == "a-long-database-password"     # a job needs the database's


@pytest.mark.anyio
async def test_the_chat_gives_the_sdk_its_own_http_client(monkeypatch):
    import httpx2  # pylint: disable=import-outside-toplevel
    from clm_decision import connection  # pylint: disable=import-outside-toplevel
    from clm_decision.portal import chat  # pylint: disable=import-outside-toplevel
    seen = []

    @contextlib.asynccontextmanager
    async def streams(url, http_client):
        seen.append((url, http_client))
        yield "streams"
    monkeypatch.setattr(connection, "streamable_http_client", streams)
    async with chat._connect("http://127.0.0.1:1/mcp", []) as found:  # pylint: disable=protected-access
        assert found == "streams"
    assert len(seen) == 1 and isinstance(seen[0][1], httpx2.AsyncClient)
    assert seen[0][1].timeout.read == chat.READ_TIMEOUT == 600 and not seen[0][1].follow_redirects
