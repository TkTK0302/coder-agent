from agent.context import ContextManager, count_tokens


class FakeLLM:
    def __init__(self, summary="【摘要】完成了一部分"):
        self.summary = summary

    def chat(self, messages, tools=None):
        class M:
            content = self.summary

        return M()


def test_count_tokens():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0


def test_trim_summarizes_middle():
    ctx = ContextManager("SYSTEM", token_budget=80, llm=FakeLLM(), keep_recent=3)
    ctx.add({"role": "user", "content": "任务：写一个计算器"})
    for i in range(10):
        ctx.add({"role": "assistant", "content": f"turn {i}"})
        ctx.add({"role": "tool", "tool_call_id": f"t{i}", "content": "x" * 100})
    ctx.add({"role": "assistant", "content": "recent"})
    ctx.add({"role": "tool", "tool_call_id": "tr", "content": "result"})

    before = len(ctx.messages)
    ctx.trim_if_needed()
    after = len(ctx.messages)

    assert after < before
    assert ctx.messages[0]["role"] == "system"
    assert ctx.messages[1]["role"] == "user" and "任务" in ctx.messages[1]["content"]
    assert "[早前对话摘要]" in ctx.messages[2]["content"]


def test_no_trim_when_under_budget():
    ctx = ContextManager("SYSTEM", token_budget=10_000, llm=FakeLLM(), keep_recent=3)
    ctx.add({"role": "user", "content": "hello"})
    ctx.add({"role": "assistant", "content": "hi"})
    n = len(ctx.messages)
    ctx.trim_if_needed()
    assert len(ctx.messages) == n
