from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from agent.schema import ToolResult
from agent.tools import Tool


def _resolve(workdir: Path, path: str) -> Path:
    """Resolve a tool path relative to the workdir, keeping it inside the sandbox."""
    root = workdir.resolve()
    p = (root / path).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"path escapes the working directory: {path}")
    return p


class ReadFileArgs(BaseModel):
    path: str = Field(description="文件路径，相对于工作目录")


class ReadFileTool(Tool):
    name = "read_file"
    description = "读取一个文本文件的内容并返回。"
    args_model = ReadFileArgs
    danger_level = "read"

    def run(self, args: ReadFileArgs) -> ToolResult:
        p = _resolve(self.workdir, args.path)
        if not p.exists():
            return ToolResult.error(
                f"文件不存在: {args.path}",
                hint="先用 list_files 查看目录结构，或检查路径拼写",
            )
        if p.is_dir():
            return ToolResult.error(f"{args.path} 是目录", hint="用 list_files 查看目录内容")
        return ToolResult(output=p.read_text(encoding="utf-8", errors="replace"))


class WriteFileArgs(BaseModel):
    path: str = Field(description="文件路径，相对于工作目录")
    content: str = Field(description="要写入的完整内容")


class WriteFileTool(Tool):
    name = "write_file"
    description = "创建或覆盖写入一个文件，会自动创建不存在的父目录。"
    args_model = WriteFileArgs
    danger_level = "write"

    def run(self, args: WriteFileArgs) -> ToolResult:
        p = _resolve(self.workdir, args.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args.content, encoding="utf-8")
        return ToolResult(output=f"已写入 {args.path}（{len(args.content)} 字符）")
