from __future__ import annotations

_PLAN_PROMPT = (
    "你正在处理一个编程任务。请先制定一个简洁的分步计划（3～7 步），"
    "每步用一句话说明要做什么。只输出计划本身，不要多余解释。"
)


class Planner:
    """Plan-then-execute (innovation 3).

    Before the execute loop, one planning call turns the task into a short step
    list. The plan is injected into the conversation so the model follows it, and
    it gives the demo a visible "第 0 步：规划" phase. Planning is best-effort:
    if it fails, the loop still runs fine without a plan.
    """

    def __init__(self, llm):
        self.llm = llm

    def plan(self, task: str) -> str:
        try:
            msg = self.llm.chat(
                [
                    {"role": "system", "content": "你是一个编程任务的规划器，只负责制定分步计划。"},
                    {"role": "user", "content": f"任务：{task}\n\n{_PLAN_PROMPT}"},
                ]
            )
            return (msg.content or "").strip()
        except Exception:
            return ""
