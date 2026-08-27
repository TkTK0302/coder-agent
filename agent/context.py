from __future__ import annotations

import tiktoken

# cl100k_base is a close-enough proxy for DeepSeek's tokenizer (DeepSeek does not
# expose an official tiktoken-compatible encoder). It is exact enough for the
# mostly-ASCII code an agent deals with; we additionally set a conservative budget.
_ENC = tiktoken.get_encoding("cl100k_base")

_SUMMARY_PROMPT = (
    "把下面的对话历史压缩成一段简洁摘要。只保留：\n"
    "1. 用户最初要完成的任务目标；\n"
    "2. 已经完成的步骤和结果；\n"
    "3. 做出的关键决策；\n"
    "4. 当前状态与仍需完成的事项。\n"
    "省略每次工具调用的原始细节。\n\n"
    "<history>\n{history}\n</history>"
)


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


class ContextManager:
    """Manages the conversation window.

    Selective compression (innovation 2) — instead of naively dropping the oldest
    messages, we:
      - always keep the system prompt and the original task (the goal must never
        be forgotten);
      - keep the most recent `keep_recent` messages (the active working set);
      - summarize the middle into a single message when the token budget is hit.

    Rationale: the oldest turns are the least relevant to *what the agent is doing
    right now*, but they still carry decisions/facts worth remembering — so we
    compress them lossily rather than discard them.
    """

    def __init__(self, system_prompt: str, token_budget: int, llm, keep_recent: int = 6):
        self.system_prompt = system_prompt
        self.token_budget = token_budget
        self.llm = llm
        self.keep_recent = keep_recent
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def add(self, message: dict) -> None:
        self.messages.append(message)

    def token_count(self) -> int:
        return sum(count_tokens(m.get("content") or "") for m in self.messages)

    def trim_if_needed(self) -> None:
        while self.token_count() > self.token_budget and len(self.messages) > self.keep_recent + 2:
            if not self._compress_once():
                break

    def _compress_once(self) -> bool:
        # The tail must start on an assistant/user turn: a "tool" message has to
        # follow its assistant tool_call, so we can't cut right before one.
        start_tail = max(2, len(self.messages) - self.keep_recent)
        while start_tail < len(self.messages) and self.messages[start_tail]["role"] == "tool":
            start_tail += 1

        middle = self.messages[2:start_tail]
        if not middle:
            return False

        summary = self._summarize(middle)
        self.messages = (
            self.messages[:2]
            + [{"role": "user", "content": f"[早前对话摘要]\n{summary}"}]
            + self.messages[start_tail:]
        )
        return True

    def _summarize(self, history: list[dict]) -> str:
        text = "\n".join(
            f"{m['role']}: {m.get('content') or ''}" for m in history if m["role"] != "system"
        )
        text = text[:8000]  # cap what we feed the summarizer
        try:
            msg = self.llm.chat([{"role": "user", "content": _SUMMARY_PROMPT.format(history=text)}])
            return (msg.content or "").strip()
        except Exception:
            # Summarization failed — fall back to a lossy tail of the raw history.
            return "(摘要失败，以下为被裁剪历史)\n" + text[-2000:]
