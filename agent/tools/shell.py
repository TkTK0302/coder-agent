from __future__ import annotations

from pydantic import BaseModel, Field

from agent.sandbox import Sandbox
from agent.schema import ToolResult
from agent.tools import Tool


class RunCommandArgs(BaseModel):
    command: str = Field(description="要执行的 shell 命令，在工作目录内运行")
    timeout: int = Field(default=60, description="超时秒数")


class RunCommandTool(Tool):
    name = "run_command"
    description = "执行一条 shell 命令，阻塞至结束或超时，返回 stdout/stderr 与退出码。适合短命令。"
    args_model = RunCommandArgs
    danger_level = "exec"

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self.workdir = sandbox.workdir

    def run(self, args: RunCommandArgs) -> ToolResult:
        try:
            result = self.sandbox.run(args.command, args.timeout)
        except TimeoutError as exc:
            return ToolResult.error(str(exc), hint="拆分为更小的步骤，或增大 timeout")
        out = result.stdout.strip()
        err = result.stderr.strip()
        if result.exit_code != 0:
            body = err or out or "(no output)"
            return ToolResult.error(f"退出码 {result.exit_code}: {body}", hint="根据错误信息修正后重试")
        return ToolResult(output=out or err or "(success, no output)")


class StartCommandArgs(BaseModel):
    command: str = Field(description="要后台启动的长时命令，如启动 web server 或长耗时编译")


class StartCommandTool(Tool):
    name = "start_command"
    description = "后台启动一个长时运行的命令（如启动 web server），返回进程 ID。之后用 check_command 查询增量输出。"
    args_model = StartCommandArgs
    danger_level = "exec"

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self.workdir = sandbox.workdir

    def run(self, args: StartCommandArgs) -> ToolResult:
        pid = self.sandbox.start(args.command)
        return ToolResult(output=f"已后台启动，进程 ID: {pid}。用 check_command 查询状态。")


class CheckCommandArgs(BaseModel):
    pid: str = Field(description="进程 ID（来自 start_command）")


class CheckCommandTool(Tool):
    name = "check_command"
    description = "查询后台进程的状态与自上次查询以来的增量输出。"
    args_model = CheckCommandArgs
    danger_level = "read"

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self.workdir = sandbox.workdir

    def run(self, args: CheckCommandArgs) -> ToolResult:
        status, output = self.sandbox.poll(args.pid)
        if status == "not_found":
            return ToolResult.error(f"进程不存在: {args.pid}", hint="检查进程 ID 是否正确")
        return ToolResult(output=f"状态: {status}\n{output.strip() or '(无新输出)'}")


class StopCommandArgs(BaseModel):
    pid: str = Field(description="进程 ID（来自 start_command）")


class StopCommandTool(Tool):
    name = "stop_command"
    description = "终止一个后台进程。"
    args_model = StopCommandArgs
    danger_level = "exec"

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self.workdir = sandbox.workdir

    def run(self, args: StopCommandArgs) -> ToolResult:
        self.sandbox.kill(args.pid)
        return ToolResult(output=f"已发送终止信号给 {args.pid}")
