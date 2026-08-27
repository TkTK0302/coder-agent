from __future__ import annotations

import argparse
from pathlib import Path

from agent.config import Config
from agent.llm import LLMClient
from agent.loop import AgentLoop
from agent.tools import ToolRegistry
from agent.tools.fs import ReadFileTool, WriteFileTool
from agent.tools.shell import RunCommandTool


def build_agent(cfg: Config) -> AgentLoop:
    """Assemble the agent from its parts (tools + llm + loop)."""
    registry = ToolRegistry()
    registry.register(ReadFileTool(cfg.workdir))
    registry.register(WriteFileTool(cfg.workdir))
    registry.register(RunCommandTool(cfg.workdir))

    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model, cfg.temperature, cfg.max_retries)
    return AgentLoop(llm, registry, cfg.max_iters)


def main() -> None:
    parser = argparse.ArgumentParser(description="自研 coding agent")
    parser.add_argument("task", nargs="?", help="任务描述；省略则进入交互模式")
    parser.add_argument("--workdir", default=".", help="工作目录（默认当前目录）")
    args = parser.parse_args()

    cfg = Config.from_env(workdir=Path(args.workdir).resolve())
    agent = build_agent(cfg)

    if args.task:
        outcome = agent.run(args.task)
        print("\n===== 最终结果 =====")
        print(outcome.final_message)
        return

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


if __name__ == "__main__":
    main()
