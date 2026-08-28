from __future__ import annotations

from dataclasses import dataclass

from agent.context import ContextManager
from agent.llm import LLMClient
from agent.memory import MemoryStore
from agent.planner import Planner
from agent.todo import TodoList
from agent.tools import ToolRegistry
from agent.trace import Tracer
from agent.verifier import Verifier


SYSTEM_PROMPT = (
    "你是一个编程智能体（coding agent），在一个受控的工作目录中帮用户完成编程任务。\n"
    "你可以调用提供的工具来读写/编辑文件、列出目录、搜索内容、执行命令、查环境、管理 git、追踪任务进度。\n"
    "工作方式（ReAct）：每一步先思考(Thought)当前状态与目标，再行动(Action)调用工具，"
    "根据观察(Observation)结果决定下一步。\n"
    "规则：\n"
    "1. 完成任务后，用自然语言简要说明你做了什么、结果如何，且不要再调用工具。\n"
    "2. 工具报错时，先读错误信息，再决定如何修正，不要盲目重复同样的操作。\n"
    "3. 所有文件操作都在工作目录内进行。\n"
    "4. 有任务清单时，用 update_todo 及时标记进度。\n"
)


def _assistant_message(msg) -> dict:
    """Serialize an OpenAI assistant message (with possible tool_calls) for the next request."""
    m: dict = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        m["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return m


def _format_relevant(chunks: list[dict]) -> str:
    lines = ["以下是工作目录中与任务最相关的代码片段（RAG 召回）："]
    for c in chunks:
        lines.append(f"\n### {c['path']}:{c['start_line']}-{c['end_line']} (score={c['score']:.3f})\n{c['text']}")
    return "\n".join(lines)


@dataclass
class RunOutcome:
    final_message: str
    iterations: int
    reason: str  # "completed" | "max_iters" | "stuck"


class AgentLoop:
    """The core observe -> decide -> act loop.

    Flow: plan -> (observe -> decide -> act)* -> verify -> done.

    Termination is multi-condition (a single condition would either loop forever
    or stop too early):
      1. max_iters — a hard safety cap on the number of turns;
      2. finish_reason=stop with no tool_calls — the model signals it is done
         (then verified by the Verifier);
      3. stuck detection — the identical tool-call batch repeated too many times.
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        max_iters: int = 30,
        token_budget: int = 64_000,
        keep_recent: int = 6,
        stuck_threshold: int = 4,
        tracer: Tracer | None = None,
        planner: Planner | None = None,
        verifier: Verifier | None = None,
        max_verify_attempts: int = 3,
        memory: MemoryStore | None = None,
        todo: TodoList | None = None,
    ):
        self.llm = llm
        self.registry = registry
        self.max_iters = max_iters
        self.token_budget = token_budget
        self.keep_recent = keep_recent
        self.stuck_threshold = stuck_threshold
        self.tracer = tracer
        self.planner = planner
        self.verifier = verifier
        self.max_verify_attempts = max_verify_attempts
        self.memory = memory
        self.todo = todo

    def run(self, task: str) -> RunOutcome:
        ctx = ContextManager(SYSTEM_PROMPT, self.token_budget, self.llm, self.keep_recent)
        ctx.add({"role": "user", "content": task})
        if self.tracer:
            self.tracer.start(task)

        # RAG：任务开始时自动索引工作目录，并召回最相关的代码片段注入上下文。
        if self.memory is not None:
            self.memory.index()
            relevant = self.memory.search(task, k=4)
            if relevant:
                ctx.add({"role": "user", "content": _format_relevant(relevant)})
                if self.tracer:
                    self.tracer.memory(len(relevant))

        # Plan-then-execute：产出结构化 todo 清单并注入，供进度追踪。
        if self.planner is not None:
            steps = self.planner.plan(task)
            if steps:
                if self.todo is not None:
                    self.todo.reset(steps)
                    ctx.add({"role": "assistant", "content": self.todo.to_prompt()})
                    if self.tracer:
                        self.tracer.plan(self.todo.to_prompt())
                else:
                    plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
                    ctx.add({"role": "assistant", "content": f"计划：\n{plan_text}"})
                    if self.tracer:
                        self.tracer.plan(plan_text)

        prev_signature: tuple | None = None
        stuck_count = 0
        verify_attempts = 0
        prev_error_sig: str | None = None
        error_repeat = 0

        for i in range(self.max_iters):
            ctx.trim_if_needed()
            msg = self.llm.chat(ctx.messages, tools=self.registry.specs())
            ctx.add(_assistant_message(msg))
            if self.tracer:
                self.tracer.iteration(i + 1)
                self.tracer.model(msg.content, msg.tool_calls)

            if not msg.tool_calls:
                # Verify-before-done: don't trust the "done" claim blindly.
                if self.verifier is not None and verify_attempts < self.max_verify_attempts:
                    verify_attempts += 1
                    ok, detail = self.verifier.verify()
                    if self.tracer:
                        self.tracer.verify(ok, detail)
                    if not ok:
                        ctx.add({"role": "user", "content": f"验证未通过：\n{detail}\n请修复后重新提交。"})
                        continue
                if self.tracer:
                    self.tracer.done("completed", i + 1)
                return RunOutcome(
                    final_message=msg.content or "(无文字说明)",
                    iterations=i + 1,
                    reason="completed",
                )

            # Stuck detection: the exact same tool-call batch repeated too often.
            signature = tuple((tc.function.name, tc.function.arguments) for tc in msg.tool_calls)
            stuck_count = stuck_count + 1 if signature == prev_signature else 1
            prev_signature = signature
            if stuck_count >= self.stuck_threshold:
                if self.tracer:
                    self.tracer.done("stuck", i + 1)
                return RunOutcome(
                    final_message=(
                        f"检测到连续 {self.stuck_threshold} 次重复的工具调用，疑似陷入死循环，已停止。"
                    ),
                    iterations=i + 1,
                    reason="stuck",
                )

            results = []
            for tc in msg.tool_calls:
                result = self.registry.execute(tc.function.name, tc.function.arguments)
                results.append(result)
                if self.tracer:
                    self.tracer.tool(tc.function.name, tc.function.arguments, result)
                ctx.add({"role": "tool", "tool_call_id": tc.id, "content": result.to_content()})

            # 语义级死循环检测：连续多轮出现同样的错误信息（即使工具调用不同），
            # 捕捉「改代码→报错→撤销→重改→同样报错」这类非字面重复的循环。
            error_sig = next((r.output[:120] for r in results if r.status == "error"), None)
            error_repeat = error_repeat + 1 if error_sig is not None and error_sig == prev_error_sig else (1 if error_sig is not None else 0)
            prev_error_sig = error_sig
            if error_repeat >= self.stuck_threshold:
                if self.tracer:
                    self.tracer.done("stuck", i + 1)
                return RunOutcome(
                    final_message=(
                        f"检测到连续 {self.stuck_threshold} 轮出现同样的错误，疑似陷入「改→错→改→错」死循环，已停止。"
                    ),
                    iterations=i + 1,
                    reason="stuck",
                )

        if self.tracer:
            self.tracer.done("max_iters", self.max_iters)
        return RunOutcome(
            final_message="达到最大迭代次数，任务可能未完成。",
            iterations=self.max_iters,
            reason="max_iters",
        )
