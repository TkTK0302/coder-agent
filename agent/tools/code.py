from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel, Field

from agent.memory import MemoryStore
from agent.schema import ToolResult
from agent.tools import Tool

_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv", ".idea", ".mypy_cache"}


class SearchCodeArgs(BaseModel):
    query: str = Field(description="语义检索查询，如“处理用户登录的函数”")
    k: int = Field(default=5, description="返回的片段数量")


class SearchCodeTool(Tool):
    name = "search_code"
    description = "语义检索代码库中与查询最相关的代码片段（RAG），适合在大型代码库中定位实现。"
    args_model = SearchCodeArgs
    danger_level = "read"

    def __init__(self, memory: MemoryStore):
        self.memory = memory
        self.workdir = memory.workdir

    def run(self, args: SearchCodeArgs) -> ToolResult:
        n = self.memory.index()
        if n == 0:
            return ToolResult(output="(工作目录中无可索引的文本文件)")
        results = self.memory.search(args.query, k=args.k)
        if not results:
            return ToolResult(output="(无匹配)")
        out = []
        for r in results:
            out.append(f"# {r['path']}:{r['start_line']}-{r['end_line']} (score={r['score']:.3f})")
            out.append(r["text"].rstrip())
            out.append("")
        return ToolResult(output="\n".join(out))


class ListSymbolsArgs(BaseModel):
    path: str = Field(description="要分析的 Python 文件路径，相对于工作目录")


class ListSymbolsTool(Tool):
    name = "list_symbols"
    description = "用 AST 解析一个 Python 文件，列出其中的函数、类、导入及其行号。"
    args_model = ListSymbolsArgs
    danger_level = "read"

    def __init__(self, workdir: Path):
        self.workdir = workdir

    def run(self, args: ListSymbolsArgs) -> ToolResult:
        p = (self.workdir / args.path).resolve()
        if not p.is_relative_to(self.workdir.resolve()):
            return ToolResult.error("路径越界")
        if not p.exists():
            return ToolResult.error(f"文件不存在: {args.path}")
        try:
            text = p.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except SyntaxError as exc:
            return ToolResult.error(f"语法错误: {exc}")

        lines = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines.append(f"函数  {node.name}()  第 {node.lineno} 行")
            elif isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                lines.append(f"类    {node.name}  第 {node.lineno} 行  方法: {', '.join(methods) or '(无)'}")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                lines.append(f"导入  {ast.get_source_segment(text, node)}")
        return ToolResult(output="\n".join(lines) or "(无顶层符号)")


class FindDefinitionArgs(BaseModel):
    symbol: str = Field(description="要查找的函数名或类名")


class FindDefinitionTool(Tool):
    name = "find_definition"
    description = "在整个工作目录中查找函数或类的定义位置（基于 AST，非字符串匹配）。"
    args_model = FindDefinitionArgs
    danger_level = "read"

    def __init__(self, workdir: Path):
        self.workdir = workdir

    def run(self, args: FindDefinitionArgs) -> ToolResult:
        results = []
        for f in self.workdir.rglob("*.py"):
            if any(part in _SKIP_DIRS for part in f.parts):
                continue
            try:
                text = f.read_text(encoding="utf-8")
                tree = ast.parse(text)
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == args.symbol:
                    rel = str(f.relative_to(self.workdir))
                    seg = ast.get_source_segment(text, node) or ""
                    results.append(f"# {rel}:{node.lineno}\n{seg}")
        return ToolResult(output="\n\n".join(results) or f"(未找到符号 {args.symbol})")
