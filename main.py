from __future__ import annotations

import argparse
from pathlib import Path

from agent.config import Config
from agent.llm import LLMClient
from agent.loop import AgentLoop
from agent.memory import MemoryStore
from agent.planner import Planner
from agent.safety import SafetyPolicy
from agent.sandbox import create_sandbox
from agent.todo import TodoList
from agent.tools import ToolRegistry
from agent.tools.ask import AskUserTool
from agent.tools.code import FindDefinitionTool, ListSymbolsTool, SearchCodeTool
from agent.tools.env import EnvInfoTool
from agent.tools.fs import EditFileTool, ListFilesTool, ReadFileTool, SearchTool, WriteFileTool
from agent.tools.git import GitCommitTool, GitDiffTool, GitLogTool, GitRestoreTool, GitStatusTool
from agent.tools.shell import CheckCommandTool, RunCommandTool, StartCommandTool, StopCommandTool
from agent.tools.todo import UpdateTodoTool
from agent.trace import Tracer
from agent.verifier import Verifier


def _confirm(command: str) -> bool:
    ans = input(f"[安全确认] 允许执行这条危险命令？\n  {command}\n  (y/N): ").strip().lower()
    return ans in {"y", "yes"}


def build_agent(
    cfg: Config,
    interactive: bool = False,
    allow_dangerous: bool = False,
    tracer=None,
    plan: bool = True,
    verify: bool = True,
    memory_enabled: bool = True,
) -> AgentLoop:
    """Assemble the agent from its parts (tools + safety + llm + loop + planner/verifier)."""
    # A SafetyPolicy gates destructive commands: confirmed interactively, or
    # auto-denied in one-shot mode. --allow-dangerous bypasses it entirely.
    safety = None if allow_dangerous else SafetyPolicy(confirmer=_confirm if interactive else None)

    # 执行环境：host 直接执行（兜底），docker 走容器隔离
    sandbox = create_sandbox(cfg.sandbox_mode, cfg.workdir, cfg.docker_image)

    registry = ToolRegistry(safety=safety)
    registry.register(ReadFileTool(cfg.workdir))
    registry.register(WriteFileTool(cfg.workdir))
    registry.register(EditFileTool(cfg.workdir))
    registry.register(ListFilesTool(cfg.workdir))
    registry.register(SearchTool(cfg.workdir))
    registry.register(RunCommandTool(sandbox))
    registry.register(StartCommandTool(sandbox))
    registry.register(CheckCommandTool(sandbox))
    registry.register(StopCommandTool(sandbox))

    # 环境感知 + Git 版本控制
    registry.register(EnvInfoTool(sandbox))
    registry.register(GitStatusTool(sandbox))
    registry.register(GitDiffTool(sandbox))
    registry.register(GitLogTool(sandbox))
    registry.register(GitCommitTool(sandbox))
    registry.register(GitRestoreTool(sandbox))

    # 代码导航（AST）
    registry.register(ListSymbolsTool(cfg.workdir))
    registry.register(FindDefinitionTool(cfg.workdir))

    # 任务清单（进度追踪）+ 主动提问（Human-in-the-Loop）
    todo = TodoList()
    registry.register(UpdateTodoTool(todo, cfg.workdir))
    registry.register(AskUserTool(cfg.workdir, interactive=interactive))

    # 长期记忆（RAG）：可选，任务开始时自动索引并召回相关代码
    memory = MemoryStore(cfg.workdir, cfg.embed_model) if memory_enabled else None
    if memory is not None:
        registry.register(SearchCodeTool(memory))

    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model, cfg.temperature, cfg.max_retries)
    planner = Planner(llm) if plan else None
    verifier = Verifier(cfg.workdir) if verify else None
    agent = AgentLoop(
        llm,
        registry,
        cfg.max_iters,
        token_budget=cfg.max_token_budget,
        tracer=tracer,
        planner=planner,
        verifier=verifier,
        memory=memory,
        todo=todo,
    )
    agent.sandbox = sandbox  # 供 main() 清理（如删除 Docker 容器）
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="自研 coding agent")
    parser.add_argument("task", nargs="?", help="任务描述；省略则进入交互模式")
    parser.add_argument("--workdir", default=".", help="工作目录（默认当前目录）")
    parser.add_argument("--allow-dangerous", action="store_true", help="关闭危险命令拦截")
    parser.add_argument("--no-plan", action="store_true", help="关闭计划阶段")
    parser.add_argument("--no-verify", action="store_true", help="关闭完成前验证")
    parser.add_argument("--no-memory", action="store_true", help="关闭 RAG 长期记忆")
    parser.add_argument("--trace", default=None, help="把完整轨迹写入 JSONL 文件（可回放）")
    args = parser.parse_args()

    cfg = Config.from_env(workdir=Path(args.workdir).resolve())
    interactive = args.task is None
    tracer = Tracer(trace_path=Path(args.trace) if args.trace else None)
    agent = build_agent(
        cfg,
        interactive=interactive,
        allow_dangerous=args.allow_dangerous,
        tracer=tracer,
        plan=not args.no_plan,
        verify=not args.no_verify,
        memory_enabled=not args.no_memory,
    )

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
        agent.sandbox.close()


if __name__ == "__main__":
    main()
