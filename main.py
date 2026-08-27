from __future__ import annotations

import argparse
from pathlib import Path

from agent.config import Config
from agent.llm import LLMClient
from agent.loop import AgentLoop
from agent.safety import SafetyPolicy
from agent.tools import ToolRegistry
from agent.tools.fs import EditFileTool, ListFilesTool, ReadFileTool, SearchTool, WriteFileTool
from agent.tools.shell import RunCommandTool
from agent.trace import Tracer


def _confirm(command: str) -> bool:
    ans = input(f"[安全确认] 允许执行这条危险命令？\n  {command}\n  (y/N): ").strip().lower()
    return ans in {"y", "yes"}


def build_agent(cfg: Config, interactive: bool = False, allow_dangerous: bool = False, tracer=None) -> AgentLoop:
    """Assemble the agent from its parts (tools + safety + llm + loop + tracer)."""
    # A SafetyPolicy gates destructive commands: confirmed interactively, or
    # auto-denied in one-shot mode. --allow-dangerous bypasses it entirely.
    safety = None if allow_dangerous else SafetyPolicy(confirmer=_confirm if interactive else None)

    registry = ToolRegistry(safety=safety)
    registry.register(ReadFileTool(cfg.workdir))
    registry.register(WriteFileTool(cfg.workdir))
    registry.register(EditFileTool(cfg.workdir))
    registry.register(ListFilesTool(cfg.workdir))
    registry.register(SearchTool(cfg.workdir))
    registry.register(RunCommandTool(cfg.workdir))

    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model, cfg.temperature, cfg.max_retries)
    return AgentLoop(llm, registry, cfg.max_iters, token_budget=cfg.max_token_budget, tracer=tracer)


def main() -> None:
    parser = argparse.ArgumentParser(description="自研 coding agent")
    parser.add_argument("task", nargs="?", help="任务描述；省略则进入交互模式")
    parser.add_argument("--workdir", default=".", help="工作目录（默认当前目录）")
    parser.add_argument("--allow-dangerous", action="store_true", help="关闭危险命令拦截")
    parser.add_argument("--trace", default=None, help="把完整轨迹写入 JSONL 文件（可回放）")
    args = parser.parse_args()

    cfg = Config.from_env(workdir=Path(args.workdir).resolve())
    interactive = args.task is None
    tracer = Tracer(trace_path=Path(args.trace) if args.trace else None)
    agent = build_agent(cfg, interactive=interactive, allow_dangerous=args.allow_dangerous, tracer=tracer)

    try:
        if not interactive:
            outcome = agent.run(args.task)
            print("\n===== 最终结果 =====")
            print(outcome.final_message)
        else:
            print("交互模式（输入 quit / exit 退出）")
            while True:
                try:
                    task = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not task:
                    continue
                if task.lower() in {"quit", "exit", "q"}:
                    break
                outcome = agent.run(task)
                print("\n===== 最终结果 =====")
                print(outcome.final_message)
    finally:
        tracer.close()


if __name__ == "__main__":
    main()
