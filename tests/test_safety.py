import pytest

from agent.safety import SafetyPolicy
from agent.tools.fs import ReadFileTool, WriteFileTool
from agent.tools.shell import RunCommandTool


@pytest.mark.parametrize(
    "cmd",
    ["rm -rf build", "git push origin main", "sudo apt install x", "git reset --hard HEAD", "shutdown -h now"],
)
def test_dangerous_detected(cmd):
    assert SafetyPolicy()._is_dangerous(cmd)


@pytest.mark.parametrize(
    "cmd",
    ["python test.py", "pytest -q", "git status", "echo hi", "rm build/out.txt"],
)
def test_safe_not_detected(cmd):
    assert not SafetyPolicy()._is_dangerous(cmd)


def test_graded_levels(tmp_path):
    sp = SafetyPolicy()
    assert sp.check(ReadFileTool(tmp_path), {"path": "x"})[0] is True
    assert sp.check(WriteFileTool(tmp_path), {"path": "x", "content": "1"})[0] is True
    allowed, _ = sp.check(RunCommandTool(tmp_path), {"command": "rm -rf /"})
    assert allowed is False


def test_confirmer_deny(tmp_path):
    sp = SafetyPolicy(confirmer=lambda c: False)
    allowed, reason = sp.check(RunCommandTool(tmp_path), {"command": "git push"})
    assert allowed is False
    assert "拒绝" in reason


def test_confirmer_allow(tmp_path):
    sp = SafetyPolicy(confirmer=lambda c: True)
    assert sp.check(RunCommandTool(tmp_path), {"command": "git push"})[0] is True
