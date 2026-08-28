from agent.planner import Planner


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)

        class M:
            content = self.response

        return M()


class FailingLLM:
    def chat(self, messages, tools=None):
        raise RuntimeError("boom")


def test_plan_parses_json():
    llm = FakeLLM('["第一步", "第二步", "第三步"]')
    planner = Planner(llm)
    assert planner.plan("任务") == ["第一步", "第二步", "第三步"]
    assert len(llm.calls) == 1


def test_plan_falls_back_to_lines():
    llm = FakeLLM("1. 创建文件\n2. 运行验证")
    planner = Planner(llm)
    assert planner.plan("任务") == ["创建文件", "运行验证"]


def test_plan_failure_returns_empty():
    planner = Planner(FailingLLM())
    assert planner.plan("任务") == []
