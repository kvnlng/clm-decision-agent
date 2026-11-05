"""Tests for Documents: loading a collection from a file in the folder or an upload, and deriving from it."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import os
import time

import anyio.to_thread
import pytest
from starlette.datastructures import UploadFile

from conftest import post, sign_in


def finished(client) -> dict:
    """The last job, once it has finished."""
    for _ in range(200):
        job = client.get("/jobs/current").json()
        if not job["running"]:
            return job
        time.sleep(0.01)
    raise AssertionError("the job did not finish")


def test_load_runs_clm_load_documents_with_the_contract_arguments(client, environment, runner, server):
    (environment.parent / "handbook.jsonl").write_text('{"id": "hb-1"}\n')
    token = sign_in(client)
    page = client.get("/documents").text
    assert "handbook.jsonl" in page and "2026-11-01T09:30:00+00:00" in page
    assert post(client, "/documents/load", token, {"collection": "handbook", "file": "handbook.jsonl"}).status_code \
        == 200
    assert finished(client)["returncode"] == 0
    assert runner.seen == [["/stand-in/bin/clm-load-documents", str(environment), "handbook",
                            str(environment.parent / "handbook.jsonl")]]
    assert len(server.starts) == 2
    assert post(client, "/documents/load", token, {"collection": "handbook", "file": "../x.jsonl"}).status_code == 400
    assert post(client, "/documents/load", token, {"collection": "nope", "file": "handbook.jsonl"}).status_code == 400


def upload(client, token, name, content, **fields):
    return post(client, "/documents/load", token, {"collection": "handbook", **fields},
                files={"upload": (name, content, "application/jsonl")})


def partials(folder) -> list[str]:
    """Upload files left half saved in the folder."""
    return sorted(path.name for path in folder.iterdir() if path.name.endswith(".partial"))


def test_an_upload_is_saved_in_the_folder_under_a_safe_name(client, environment, runner, tmp_path):
    token = sign_in(client)
    assert upload(client, token, "../../my handbook!.jsonl", b'{"id": "hb-1"}\n').status_code == 200
    saved = environment.parent / "my_handbook_.jsonl"
    assert saved.read_bytes() == b'{"id": "hb-1"}\n' and not (tmp_path / "my handbook!.jsonl").exists()
    finished(client)
    assert runner.seen[0][-1] == str(saved)
    assert upload(client, token, "notes.txt", b"x").status_code == 400
    assert not (environment.parent / "notes.txt").exists()


def test_an_upload_does_not_overwrite_without_the_tick(client, environment, runner):
    existing = environment.parent / "handbook.jsonl"
    existing.write_text("old\n")
    token = sign_in(client)
    response = upload(client, token, "handbook.jsonl", b"new\n")
    assert response.status_code == 409 and "overwrite" in response.text
    assert existing.read_text() == "old\n" and not runner.seen
    assert upload(client, token, "handbook.jsonl", b"new\n", overwrite="on").status_code == 200
    assert existing.read_text() == "new\n" and not partials(environment.parent)


def test_an_upload_that_cannot_be_saved_leaves_the_file_it_would_replace_as_it_was(client, environment, runner,
                                                                                   monkeypatch):
    existing = environment.parent / "handbook.jsonl"
    existing.write_text("old\n")

    def disk_full(*_):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(os, "replace", disk_full)
    token = sign_in(client)
    response = upload(client, token, "handbook.jsonl", b"new\n", overwrite="on")
    assert response.status_code == 500 and "cannot be saved" in response.text
    assert existing.read_text() == "old\n" and not partials(environment.parent) and not runner.seen


@pytest.mark.parametrize("installed", [{"clm-server", "clm-derive"}])
def test_an_upload_is_not_saved_when_clm_load_documents_is_not_installed(client, environment, runner):
    token = sign_in(client)
    response = upload(client, token, "handbook.jsonl", b'{"id": "hb-1"}\n')
    assert response.status_code == 409 and "clm-load-documents is not installed" in response.text
    assert not (environment.parent / "handbook.jsonl").exists() and not runner.seen
    assert not partials(environment.parent)


def test_an_upload_is_not_saved_when_a_job_starts_while_it_arrives(client, environment, portal, monkeypatch):
    read = UploadFile.read

    async def read_while_a_job_starts(self, *args):
        await portal.jobs.lock.acquire()
        return await read(self, *args)
    monkeypatch.setattr(UploadFile, "read", read_while_a_job_starts)
    token = sign_in(client)
    response = upload(client, token, "handbook.jsonl", b'{"id": "hb-1"}\n')
    assert response.status_code == 409 and "wait" in response.text
    assert not (environment.parent / "handbook.jsonl").exists() and not partials(environment.parent)


def test_an_upload_whose_save_is_cut_off_leaves_no_partial_file(client, environment, runner, monkeypatch):
    async def written_then_cut_off(function, *args):
        function(*args)
        raise RuntimeError("the request ended while the file was written")
    monkeypatch.setattr(anyio.to_thread, "run_sync", written_then_cut_off)
    token = sign_in(client)
    with pytest.raises(RuntimeError):
        upload(client, token, "handbook.jsonl", b'{"id": "hb-1"}\n')
    assert not partials(environment.parent) and not (environment.parent / "handbook.jsonl").exists()
    assert not runner.seen


def test_derive_runs_clm_derive_run(client, environment, runner):
    token = sign_in(client)
    post(client, "/documents/derive", token, {"collection": "handbook", "backend": "claude",
                                              "model": "claude-opus-5-5"})
    finished(client)
    post(client, "/documents/derive", token, {"collection": "handbook"})
    finished(client)
    assert runner.seen == [["/stand-in/bin/clm-derive", "run", str(environment), "handbook", "--backend", "claude",
                            "--model", "claude-opus-5-5"],
                           ["/stand-in/bin/clm-derive", "run", str(environment), "handbook"]]
    assert post(client, "/documents/derive", token, {"collection": "handbook", "backend": "gpt"}).status_code == 400


@pytest.mark.parametrize("installed", [{"clm-server", "clm-load-documents"}])
def test_without_clm_derive_the_page_says_derivation_is_not_available(client, runner):
    token = sign_in(client)
    assert "Derivation is not available yet" in client.get("/documents").text
    page = post(client, "/documents/derive", token, {"collection": "handbook"}).text
    assert '<p class="failed">clm-derive is not installed' in page and not runner.seen


def test_a_running_job_sends_the_page_back_to_a_get_when_it_ends(client, runner):
    token = sign_in(client)
    runner.release.clear()
    try:
        page = post(client, "/documents/derive", token, {"collection": "handbook"}).text
        assert 'data-job-running="/documents"' in page
        assert 'data-job-running="/review"' in client.get("/review").text
    finally:
        runner.release.set()
    finished(client)
