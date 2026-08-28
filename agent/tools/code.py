from __future__ import annotations

from pydantic import BaseModel, Field

from agent.memory import MemoryStore
from agent.schema import ToolResult
from agent.tools import Tool


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
