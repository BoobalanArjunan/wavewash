"""The local web server's safety rules and the command line."""
import sys
import time

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from conftest import speechlike, write  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WAVEWASH_HOME", str(tmp_path / "home"))
    import importlib
    import wavewash.server as server
    importlib.reload(server)
    return TestClient(server.app, base_url="http://127.0.0.1:8420"), server


def test_health_and_ui_are_served(client):
    c, _ = client
    assert c.get("/api/health").json() == {"status": "ok"}
    r = c.get("/")
    assert r.status_code == 200 and "WaveWash" in r.text


def test_foreign_host_is_rejected(client):
    c, _ = client
    assert c.get("/api/browse", headers={"host": "evil.com"}).status_code == 403


def test_foreign_origin_and_cross_site_are_rejected(client):
    c, _ = client
    assert c.post("/api/open-folder", json={"path": "/tmp"},
                  headers={"origin": "https://evil.com"}).status_code == 403
    assert c.get("/api/browse", headers={"sec-fetch-site": "cross-site"}).status_code == 403


def test_download_rejects_bad_job_ids(client):
    c, _ = client
    assert c.get("/api/jobs/..%2F..%2Fetc/download").status_code in (400, 404)
    assert c.get("/api/jobs/doesnotexist/download").status_code == 404


def test_output_folder_rules(client, tmp_path):
    c, _ = client
    inp = tmp_path / "in"
    write(inp / "a.wav", speechlike())
    base = {"input_path": str(inp), "purpose": "asr"}
    assert c.post("/api/run", json={**base, "output_path": str(inp / "out")}).status_code == 400
    if sys.platform != "win32":
        assert c.post("/api/run", json={**base, "output_path": "/etc"}).status_code == 400
        assert c.post("/api/run", json={**base, "output_path": "/etc/wavewash-out"}).status_code == 400
    assert c.post("/api/run", json={**base, "output_path": str(tmp_path)}).status_code == 400  # contains input
    assert c.post("/api/run", json={"input_path": str(tmp_path / "nope"), "purpose": "asr"}).status_code == 400
    assert c.post("/api/run", json={**base, "purpose": "not_a_purpose"}).status_code == 400


def test_a_job_runs_to_completion_through_the_api(client, tmp_path):
    c, _ = client
    inp = tmp_path / "in"
    write(inp / "a.wav", speechlike(seed=1))
    write(inp / "dc.wav", speechlike(seed=2) + 0.1)
    out = tmp_path / "api_out"
    job = c.post("/api/run", json={"input_path": str(inp), "purpose": "asr", "output_path": str(out)}).json()["job_id"]
    for _ in range(90):
        status = c.get(f"/api/jobs/{job}").json()["status"]
        if status in ("done", "error"):
            break
        time.sleep(1)
    assert status == "done"
    res = c.get(f"/api/jobs/{job}/results").json()
    assert res["status_counts"]
    z = c.get(f"/api/jobs/{job}/download")
    assert z.status_code == 200 and z.content[:2] == b"PK"


def test_browse_lists_a_folder(client, tmp_path):
    c, _ = client
    write(tmp_path / "d" / "x.wav", speechlike())
    r = c.get("/api/browse", params={"path": str(tmp_path)}).json()
    assert any(e["name"] == "d" and e["is_dir"] for e in r["entries"])
    assert c.get("/api/browse", params={"path": str(tmp_path / "missing")}).status_code == 404


def test_server_refuses_non_loopback_bind():
    from wavewash.server import serve
    with pytest.raises(SystemExit):
        serve(host="0.0.0.0", open_browser=False)


def test_cli_version_and_doctor(capsys):
    from wavewash.cli import main
    assert main(["doctor"]) == 0
    assert "Result: all good" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--version"])
