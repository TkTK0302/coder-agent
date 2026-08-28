from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from agent.schema import ToolResult
from agent.todo import TodoList
from agent.tools import Tool


class UpdateTodoArgs(BaseModel):
    todo_id: int = Field(description="任务 ID")
    status: str = Field(description="新状态：pending / in_progress / done")


class UpdateTodoTool(Tool):
    name = "update_todo"
    description = "更新任务清单中某项的进度状态（pending / in_progress / done）。"
    args_model = UpdateTodoArgs
    danger_level = "read"

    def __init__(self, todo: TodoList, workdir: Path):
        self.todo = todo
        self.workdir = workdir

    def run(self, args: UpdateTodoArgs) -> ToolResult:
        if args.status not in ("pending", "in_progress", "done"):
            return ToolResult.error(f"非法状态: {args.status}", hint="状态应为 pending/in_progress/done")
        if self.todo.update(args.todo_id, args.status):
            return ToolResult(output=f"已更新任务 {args.todo_id} 状态为 {args.status}")
        return ToolResult.error(f"任务 {args.todo_id} 不存在", hint="用 list 查看当前任务 ID")
