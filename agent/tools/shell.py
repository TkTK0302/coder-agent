from __future__ import annotations

import subprocess

from pydantic import BaseModel, Field

from agent.schema import ToolResult
from agent.tools import Tool


class RunCommandArgs(BaseModel):
    command: str = Field(description="要执行的 shell 命令，在工作目录内运行")
    timeout: int = Field(default=60, description="超时秒数")


class RunCommandTool(Tool):
    name = "run_command"
    description = "在工作目录内执行一条 shell 命令，返回 stdout/stderr 与退出码。"
    args_model = RunCommandArgs
    danger_level = "exec"

    def run(self, args: RunCommandArgs) -> ToolResult:
        try:
            proc = subprocess.run(
                args.command,
                shell=True,
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.error(
                f"命令超时（>{args.timeout}s）", hint="拆分为更小的步骤，或增大 timeout"
            )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        if proc.returncode != 0:
            body = err or out or "(no output)"
            return ToolResult.error(f"退出码 {proc.returncode}: {body}", hint="根据错误信息修正后重试")
        return ToolResult(output=out or err or "(success, no output)")
