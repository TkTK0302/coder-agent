from agent.loop import AgentLoop
from agent.sandbox import HostSandbox
from agent.tools import ToolRegistry
from agent.tools.fs import EditFileTool, ListFilesTool, ReadFileTool, SearchTool, WriteFileTool
from agent.tools.shell import RunCommandTool


def make_registry(workdir):
    reg = ToolRegistry()
    for t in (
        ReadFileTool(workdir),
        WriteFileTool(workdir),
        EditFileTool(workdir),
        ListFilesTool(workdir),
        SearchTool(workdir),
        RunCommandTool(HostSandbox(workdir)),
    ):
        reg.register(t)
    return reg


def make_msg(content, tool_calls):
    class F:
        def __init__(self, n, a):
            self.name, self.arguments = n, a

    class TC:
        def __init__(self, n, a):
            self.id, self.function = "call_1", F(n, a)

    class M:
        pass

    m = M()
    m.content = content
    m.tool_calls = [TC(n, a) for n, a in (tool_calls or [])]
    return m


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, messages, tools=None):
        return self.responses.pop(0)


def test_completion(tmp_path):
    llm = FakeLLM(
        [
            make_msg(None, [("run_command", '{"command": "echo hi"}')]),
            make_msg("完成", None),
        ]
    )
    out = AgentLoop(llm, make_registry(tmp_path)).run("task")
    assert out.reason == "completed"
    assert out.final_message == "完成"


def test_stuck_detection(tmp_path):
    llm = FakeLLM([make_msg(None, [("run_command", '{"command": "ls"}')])] * 10)
    out = AgentLoop(llm, make_registry(tmp_path)).run("task")
    assert out.reason == "stuck"


def test_semantic_error_loop(tmp_path):
    # 不同的 old_string 都找不到 → 同样的错误信息反复出现（非字面重复的循环）
    (tmp_path / "x.py").write_text("hello\n", encoding="utf-8")
    responses = [
        make_msg(None, [("edit_file", f'{{"path": "x.py", "old_string": "missing{i}", "new_string": "x"}}')])
        for i in range(10)
    ]
    llm = FakeLLM(responses)
    out = AgentLoop(llm, make_registry(tmp_path)).run("task")
    assert out.reason == "stuck"


def test_max_iters(tmp_path):
    responses = [make_msg(None, [("run_command", f'{{"command": "echo {i}"}}')]) for i in range(30)]
    llm = FakeLLM(responses)
    out = AgentLoop(llm, make_registry(tmp_path), max_iters=30).run("task")
    assert out.reason == "max_iters"
