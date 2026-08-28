import time

from agent.sandbox import CommandResult, HostSandbox


def test_run_success(tmp_path):
    sb = HostSandbox(tmp_path)
    r = sb.run("echo hello")
    assert isinstance(r, CommandResult)
    assert r.ok
    assert r.stdout.strip() == "hello"


def test_run_failure(tmp_path):
    sb = HostSandbox(tmp_path)
    r = sb.run('python -c "import sys; sys.exit(3)"')
    assert r.exit_code == 3


def test_run_timeout(tmp_path):
    sb = HostSandbox(tmp_path)
    try:
        sb.run('python -c "import time; time.sleep(5)"', timeout=1)
    except TimeoutError:
        return
    assert False, "应该抛 TimeoutError"


def test_background_lifecycle(tmp_path):
    (tmp_path / "bg.py").write_text(
        "import time\nprint('started', flush=True)\ntime.sleep(0.5)\nprint('done', flush=True)\n",
        encoding="utf-8",
    )
    sb = HostSandbox(tmp_path)
    pid = sb.start("python bg.py")
    collected = []
    status = "running"
    for _ in range(40):
        status, out = sb.poll(pid)
        if out:
            collected.append(out)
        if status.startswith("exited"):
            break
        time.sleep(0.05)
    combined = "".join(collected)
    assert status.startswith("exited")
    assert "started" in combined and "done" in combined


def test_kill_background(tmp_path):
    (tmp_path / "sleep.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    sb = HostSandbox(tmp_path)
    pid = sb.start("python sleep.py")
    sb.kill(pid)
    time.sleep(0.2)
    status, _ = sb.poll(pid)
    assert status.startswith("exited")
