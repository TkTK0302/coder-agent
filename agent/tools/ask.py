from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from agent.schema import ToolResult
from agent.tools import Tool


class AskUserArgs(BaseModel):
    question: str = Field(description="要问用户的问题")


class AskUserTool(Tool):
    name = "ask_user"
    description = "需求模糊或缺少必要信息（如 API key、关键决策）时，向用户提问澄清，而不是瞎猜。"
    args_model = AskUserArgs
    danger_level = "read"

    def __init__(self, workdir: Path, interactive: bool = False):
        self.workdir = workdir
        self.interactive = interactive

    def run(self, args: AskUserArgs) -> ToolResult:
        if not self.interactive:
            return ToolResult.error(
                "非交互模式下无法向用户提问",
                hint="做出合理假设并继续，或在最终回答里明确说明你做了哪些假设",
            )
        ans = input(f"[需要澄清] {args.question}\n> ").strip()
        if not ans:
            return ToolResult.error("用户未回答", hint="做出合理假设并继续")
        return ToolResult(output=f"用户回答：{ans}")
