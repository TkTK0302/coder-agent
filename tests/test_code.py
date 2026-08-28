from agent.tools.code import FindDefinitionTool, ListSymbolsTool


def test_list_symbols(tmp_path):
    (tmp_path / "m.py").write_text(
        "import os\n\n\ndef foo():\n    return 1\n\n\nclass Bar:\n    def baz(self):\n        pass\n",
        encoding="utf-8",
    )
    r = ListSymbolsTool(tmp_path).run(type("A", (), {"path": "m.py"})())
    assert r.status == "success"
    assert "foo" in r.output and "Bar" in r.output and "baz" in r.output


def test_find_definition(tmp_path):
    (tmp_path / "a.py").write_text("def target():\n    return 42\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("x = 1\n", encoding="utf-8")
    r = FindDefinitionTool(tmp_path).run(type("A", (), {"symbol": "target"})())
    assert r.status == "success"
    assert "a.py" in r.output and "return 42" in r.output


def test_find_definition_missing(tmp_path):
    r = FindDefinitionTool(tmp_path).run(type("A", (), {"symbol": "nope"})())
    assert "未找到" in r.output
