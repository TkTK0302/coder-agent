from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from agent.schema import ToolResult
from agent.tools import Tool

# Directories skipped when walking/searching recursively.
_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv", ".idea", ".mypy_cache"}


def _resolve(workdir: Path, path: str) -> Path:
    """Resolve a tool path relative to the workdir, keeping it inside the sandbox."""
    root = workdir.resolve()
    p = (root / path).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"path escapes the working directory: {path}")
    return p


def _rel(workdir: Path, p: Path) -> str:
    rel = p.resolve().relative_to(workdir.resolve())
    return str(rel) if str(rel) else "."


# ---------- read_file ----------


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


# ---------- write_file ----------


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


# ---------- list_files ----------


class ListFilesArgs(BaseModel):
    path: str = Field(".", description="要列出的目录，相对于工作目录")
    recursive: bool = Field(False, description="是否递归列出所有子目录")


class ListFilesTool(Tool):
    name = "list_files"
    description = "列出目录中的文件和子目录（目录项末尾带 /）。"
    args_model = ListFilesArgs
    danger_level = "read"

    def run(self, args: ListFilesArgs) -> ToolResult:
        p = _resolve(self.workdir, args.path)
        if not p.exists():
            return ToolResult.error(f"目录不存在: {args.path}", hint="检查路径或先列出当前目录")
        if not p.is_dir():
            return ToolResult.error(f"{args.path} 不是目录", hint="用 read_file 读取，或列出其父目录")
        candidates = sorted(p.rglob("*")) if args.recursive else sorted(p.iterdir())
        lines: list[str] = []
        for item in candidates:
            if any(part in _SKIP_DIRS for part in item.relative_to(p).parts):
                continue
            rel = _rel(self.workdir, item)
            lines.append(rel + ("/" if item.is_dir() else ""))
        return ToolResult(output="\n".join(lines) if lines else "(空目录)")


# ---------- search ----------


class SearchArgs(BaseModel):
    pattern: str = Field(description="要搜索的正则表达式")
    path: str = Field(".", description="搜索的目录或文件，相对于工作目录")
    recursive: bool = Field(True, description="是否递归搜索子目录")


class SearchTool(Tool):
    name = "search"
    description = "用正则表达式搜索文件内容，返回 文件:行号:内容 列表。"
    args_model = SearchArgs
    danger_level = "read"

    def run(self, args: SearchArgs) -> ToolResult:
        try:
            regex = re.compile(args.pattern)
        except re.error as exc:
            return ToolResult.error(f"无效的正则表达式: {exc}", hint="修正正则表达式后重试")

        target = _resolve(self.workdir, args.path)
        if target.is_file():
            files = [target]
        elif target.is_dir():
            files = [f for f in (target.rglob("*") if args.recursive else target.iterdir()) if f.is_file()]
        else:
            return ToolResult.error(f"路径不存在: {args.path}", hint="检查路径")

        matches: list[str] = []
        for f in sorted(files):
            if any(part in _SKIP_DIRS for part in f.parts):
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # skip binary / unreadable files
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{_rel(self.workdir, f)}:{lineno}: {line.strip()}")

        if not matches:
            return ToolResult(output="(无匹配)")
        total = len(matches)
        if total > 200:
            matches = matches[:200]
            matches.append(f"... (共 {total} 条匹配，仅显示前 200 条)")
        return ToolResult(output="\n".join(matches))


# ---------- edit_file ----------


class EditArgs(BaseModel):
    path: str = Field(description="要编辑的文件路径，相对于工作目录")
    old_string: str = Field(description="要替换的原文，必须在文件中唯一出现")
    new_string: str = Field(description="替换后的新内容")


class EditFileTool(Tool):
    name = "edit_file"
    description = "对文件做一次精确字符串替换。old_string 必须唯一，否则报错要求更多上下文。"
    args_model = EditArgs
    danger_level = "write"

    def run(self, args: EditArgs) -> ToolResult:
        p = _resolve(self.workdir, args.path)
        if not p.exists():
            return ToolResult.error(f"文件不存在: {args.path}", hint="先用 read_file 确认内容")
        if p.is_dir():
            return ToolResult.error(f"{args.path} 是目录", hint="edit_file 只能编辑文件")
        content = p.read_text(encoding="utf-8", errors="replace")
        count = content.count(args.old_string)
        if count == 0:
            return ToolResult.error(
                "old_string 未在文件中找到", hint="先用 read_file 查看真实内容，注意缩进与换行"
            )
        if count > 1:
            return ToolResult.error(
                f"old_string 出现 {count} 次，不唯一", hint="提供更多上下文使其唯一后重试"
            )
        p.write_text(content.replace(args.old_string, args.new_string, 1), encoding="utf-8")
        return ToolResult(output=f"已替换 {args.path} 中的 1 处内容")
