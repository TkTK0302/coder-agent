from agent.tools import ToolRegistry
from agent.tools.fs import EditFileTool, ListFilesTool, ReadFileTool, SearchTool, WriteFileTool
from agent.tools.shell import RunCommandTool


def make_registry(workdir, safety=None):
    reg = ToolRegistry(safety=safety)
    for t in (
        ReadFileTool(workdir),
        WriteFileTool(workdir),
        EditFileTool(workdir),
        ListFilesTool(workdir),
        SearchTool(workdir),
        RunCommandTool(workdir),
    ):
        reg.register(t)
    return reg


def test_write_then_read(tmp_path):
    reg = make_registry(tmp_path)
    assert reg.execute("write_file", {"path": "a/b.txt", "content": "hello"}).status == "success"
    r = reg.execute("read_file", {"path": "a/b.txt"})
    assert r.status == "success"
    assert r.output == "hello"


def test_read_missing(tmp_path):
    reg = make_registry(tmp_path)
    r = reg.execute("read_file", {"path": "nope.txt"})
    assert r.status == "error"
    assert "不存在" in r.output


def test_edit_requires_unique(tmp_path):
    reg = make_registry(tmp_path)
    reg.execute("write_file", {"path": "x.py", "content": "a\nb\na\n"})
    assert reg.execute("edit_file", {"path": "x.py", "old_string": "a", "new_string": "c"}).status == "error"
    r = reg.execute("edit_file", {"path": "x.py", "old_string": "b", "new_string": "B"})
    assert r.status == "success"
    assert "B" in reg.execute("read_file", {"path": "x.py"}).output


def test_search(tmp_path):
    reg = make_registry(tmp_path)
    reg.execute("write_file", {"path": "x.py", "content": "def foo():\n    return 1\n"})
    r = reg.execute("search", {"pattern": "def foo", "path": "."})
    assert r.status == "success"
    assert "def foo" in r.output


def test_path_escape_blocked(tmp_path):
    reg = make_registry(tmp_path)
    assert reg.execute("read_file", {"path": "../../../etc/passwd"}).status == "error"


def test_unknown_tool_returns_envelope(tmp_path):
    reg = make_registry(tmp_path)
    r = reg.execute("no_such_tool", "{}")
    assert r.status == "error"
    assert "未知工具" in r.output


def test_malformed_json_returns_envelope(tmp_path):
    reg = make_registry(tmp_path)
    assert reg.execute("run_command", "{bad json").status == "error"


def test_run_command(tmp_path):
    reg = make_registry(tmp_path)
    r = reg.execute("run_command", {"command": "echo ok"})
    assert r.status == "success"
    assert r.output == "ok"


def test_specs_are_valid_json_schema(tmp_path):
    reg = make_registry(tmp_path)
    specs = reg.specs()
    assert len(specs) == 6
    for spec in specs:
        fn = spec["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
