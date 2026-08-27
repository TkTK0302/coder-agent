from __future__ import annotations

from dataclasses import dataclass

from agent.llm import LLMClient
from agent.tools import ToolRegistry


SYSTEM_PROMPT = (
    "你是一个编程智能体（coding agent），在一个受控的工作目录中帮用户完成编程任务。\n"
    "你可以调用提供的工具来读写文件、执行命令。\n"
    "规则：\n"
    "1. 完成任务后，用自然语言简要说明你做了什么、结果如何，且不要再调用工具。\n"
    "2. 工具报错时，先读错误信息，再决定如何修正，不要盲目重复同样的操作。\n"
    "3. 所有文件操作都在工作目录内进行。\n"
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


@dataclass
class RunOutcome:
    final_message: str
    iterations: int
    reason: str  # "completed" | "max_iters"


class AgentLoop:
    """The core observe -> decide -> act loop."""

    def __init__(self, llm: LLMClient, registry: ToolRegistry, max_iters: int = 30):
        self.llm = llm
        self.registry = registry
        self.max_iters = max_iters

    def run(self, task: str) -> RunOutcome:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        for i in range(self.max_iters):
            msg = self.llm.chat(messages, tools=self.registry.specs())
            messages.append(_assistant_message(msg))

            if not msg.tool_calls:
                return RunOutcome(
                    final_message=msg.content or "(无文字说明)",
                    iterations=i + 1,
                    reason="completed",
                )

            for tc in msg.tool_calls:
                result = self.registry.execute(tc.function.name, tc.function.arguments)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result.to_content()}
                )

        return RunOutcome(
            final_message="达到最大迭代次数，任务可能未完成。",
            iterations=self.max_iters,
            reason="max_iters",
        )
