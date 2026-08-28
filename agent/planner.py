from __future__ import annotations

import json
import re

_PLAN_PROMPT = (
    "你正在处理一个编程任务。请先制定一个分步计划，以 JSON 数组形式返回，"
    "每个元素是一句步骤描述。例如：[\"分析现有代码\", \"实现核心逻辑\", \"编写测试并运行\"]。"
    "只输出 JSON 数组本身，不要多余解释。"
)


class Planner:
    """Plan-then-execute（创新③）。

    执行前把任务拆成一个分步计划（结构化 todo），供 loop 追踪进度、agent 逐步勾选。
    计划是 best-effort：失败时返回空列表，loop 仍可正常执行。
    """

    def __init__(self, llm):
        self.llm = llm

    def plan(self, task: str) -> list[str]:
        try:
            msg = self.llm.chat(
                [
                    {"role": "system", "content": "你是一个编程任务的规划器，只负责制定分步计划。"},
                    {"role": "user", "content": f"任务：{task}\n\n{_PLAN_PROMPT}"},
                ]
            )
            return self._parse_steps((msg.content or "").strip())
        except Exception:
            return []

    def _parse_steps(self, content: str) -> list[str]:
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if m:
            try:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    return [str(s).strip() for s in arr if str(s).strip()]
            except json.JSONDecodeError:
                pass
        # 回退：按行拆分，去掉编号/列表符号
        lines = []
        for line in content.splitlines():
            cleaned = line.strip().lstrip("-*0123456789.、) ")
            if cleaned:
                lines.append(cleaned)
        return lines
