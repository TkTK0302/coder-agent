from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class Tracer:
    """Observability + replayable trace (innovation 6).

    Two jobs:
      1. Print a readable, structured play-by-play ("思考 -> 决策 -> 执行 -> 结果")
         so a human can watch the agent reason — the backbone of the demo video.
      2. Optionally append every event to a JSONL file, giving a replayable,
         greppable record of exactly what the agent did and why.
    """

    def __init__(self, trace_path: Path | None = None, console: bool = True):
        self.console = console
        self._fh = None
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = trace_path.open("a", encoding="utf-8")

    # ----- low-level -----
    def _emit(self, event: dict) -> None:
        event["ts"] = datetime.now().isoformat(timespec="seconds")
        if self._fh is not None:
            self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._fh.flush()

    def _print(self, line: str = "") -> None:
        if self.console:
            print(line)

    # ----- high-level -----
    def start(self, task: str) -> None:
        self._emit({"type": "run_start", "task": task})
        self._print(f"任务: {task}")

    def plan(self, plan_text: str) -> None:
        self._emit({"type": "plan", "plan": plan_text})
        self._print(f"\n[计划]\n{plan_text}")

    def verify(self, ok: bool, detail: str) -> None:
        self._emit({"type": "verify", "ok": ok, "detail": detail})
        self._print(f"\n[验证] {'通过' if ok else '未通过'}: {detail}")

    def iteration(self, n: int) -> None:
        self._emit({"type": "iteration", "n": n})
        self._print(f"\n───── 第 {n} 轮 ─────")

    def model(self, content: str | None, tool_calls) -> None:
        self._emit(
            {
                "type": "model",
                "content": content,
                "tool_calls": [
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in (tool_calls or [])
                ],
            }
        )
        if tool_calls:
            names = ", ".join(tc.function.name for tc in tool_calls)
            self._print(f"模型决策: 调用工具 {names}")
        else:
            self._print(f"模型回答: {(content or '').strip()[:200]}")

    def tool(self, name: str, args: str, result) -> None:
        self._emit({"type": "tool", "name": name, "arguments": args, "result": result.to_content()})
        self._print(f"  > {name}  {args}")
        for line in result.to_content().splitlines():
            self._print(f"      {line}")

    def done(self, reason: str, iterations: int) -> None:
        self._emit({"type": "run_end", "reason": reason, "iterations": iterations})
        self._print(f"\n[结束] 原因={reason}，共 {iterations} 轮")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
