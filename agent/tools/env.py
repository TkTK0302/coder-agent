from __future__ import annotations

from pydantic import BaseModel

from agent.sandbox import Sandbox
from agent.schema import ToolResult
from agent.tools import Tool


class EnvInfoArgs(BaseModel):
    pass


class EnvInfoTool(Tool):
    name = "env_info"
    description = "查看执行环境：工作目录、Python 版本、已安装的 Python 依赖（前 25 个）。"
    args_model = EnvInfoArgs
    danger_level = "read"

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self.workdir = sandbox.workdir

    def run(self, args: EnvInfoArgs) -> ToolResult:
        lines = [f"工作目录: {self.workdir}"]
        r = self.sandbox.run(
            "python --version 2>&1; echo '---deps---'; pip list --format=freeze 2>/dev/null | head -25",
            timeout=30,
        )
        if r.exit_code != 0:
            lines.append(f"(环境探测失败: {r.stderr.strip()[:200]})")
        else:
            lines.append(r.stdout.strip())
        return ToolResult(output="\n".join(lines))
