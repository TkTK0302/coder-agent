"""真实端到端集成测试：需要 DEEPSEEK_API_KEY，未设置时自动跳过。

用法：
    DEEPSEEK_API_KEY=sk-xxx pytest tests/test_integration.py -v
"""

import os
from pathlib import Path

import pytest

from agent.config import Config, _load_dotenv
from agent.llm import LLMClient
from agent.loop import AgentLoop
from agent.tools import ToolRegistry
from agent.tools.fs import EditFileTool, ListFilesTool, ReadFileTool, SearchTool, WriteFileTool
from agent.tools.shell import RunCommandTool


# 让集成测试在 .env 已配置时也能运行（无需额外设置环境变量）
_load_dotenv(Path(__file__).resolve().parent.parent / ".env")

pytestmark = pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"), reason="需要 DEEPSEEK_API_KEY 才能跑真实集成测试"
)


def _build_agent(workdir):
    cfg = Config.from_env(workdir=workdir)
    reg = ToolRegistry()
    for t in (
        ReadFileTool(workdir),
        WriteFileTool(workdir),
        EditFileTool(workdir),
        ListFilesTool(workdir),
        SearchTool(workdir),
        RunCommandTool(workdir),
    ):
        reg.register(t)
    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model, cfg.temperature, cfg.max_retries)
    return AgentLoop(llm, reg, cfg.max_iters, token_budget=cfg.max_token_budget)


def test_real_end_to_end(tmp_path):
    agent = _build_agent(tmp_path)
    task = "在当前目录创建一个 calc.py，实现 add(a,b) 返回两数之和，然后用 python 运行验证 add(1,2)==3。"
    outcome = agent.run(task)
    assert outcome.reason in ("completed", "max_iters")
    assert (tmp_path / "calc.py").exists()
    content = (tmp_path / "calc.py").read_text(encoding="utf-8")
    assert "add" in content
