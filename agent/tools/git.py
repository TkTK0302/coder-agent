from __future__ import annotations

from pydantic import BaseModel, Field

from agent.sandbox import Sandbox
from agent.schema import ToolResult
from agent.tools import Tool


class _NoArgs(BaseModel):
    pass


class GitLogArgs(BaseModel):
    n: int = Field(default=10, description="返回的提交条数")


class GitCommitArgs(BaseModel):
    message: str = Field(description="提交信息")


class GitRestoreArgs(BaseModel):
    path: str = Field(description="要回滚的文件路径（丢弃该文件的未提交改动）")


class _GitTool(Tool):
    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self.workdir = sandbox.workdir

    def _run_git(self, command: str, timeout: int = 30) -> ToolResult:
        r = self.sandbox.run(command, timeout=timeout)
        if r.exit_code != 0:
            return ToolResult.error(f"git 命令失败: {r.stderr.strip()[:300] or r.stdout.strip()[:300]}", hint="确认是否在 git 仓库中")
        return ToolResult(output=r.stdout.strip() or "(无输出)")


class GitStatusTool(_GitTool):
    name = "git_status"
    description = "查看 git 工作区状态（已修改/已暂存/未跟踪文件）。"
    args_model = _NoArgs
    danger_level = "read"

    def run(self, args: _NoArgs) -> ToolResult:
        return self._run_git("git status --short")


class GitDiffTool(_GitTool):
    name = "git_diff"
    description = "查看未暂存的代码改动。"
    args_model = _NoArgs
    danger_level = "read"

    def run(self, args: _NoArgs) -> ToolResult:
        return self._run_git("git diff --stat && git diff")


class GitLogTool(_GitTool):
    name = "git_log"
    description = "查看最近的提交历史。"
    args_model = GitLogArgs
    danger_level = "read"

    def run(self, args: GitLogArgs) -> ToolResult:
        return self._run_git(f"git log --oneline -n {args.n}")


class GitCommitTool(_GitTool):
    name = "git_commit"
    description = "提交当前所有改动（本地提交，可回滚）。"
    args_model = GitCommitArgs
    danger_level = "write"

    def run(self, args: GitCommitArgs) -> ToolResult:
        return self._run_git(f'git add -A && git commit -m "{args.message}"')


class GitRestoreTool(_GitTool):
    name = "git_restore"
    description = "回滚指定文件的未提交改动（丢弃该文件的本地修改）。"
    args_model = GitRestoreArgs
    danger_level = "write"

    def run(self, args: GitRestoreArgs) -> ToolResult:
        return self._run_git(f"git restore -- {args.path}")
