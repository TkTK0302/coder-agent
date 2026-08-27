from agent.verifier import Verifier


def test_valid_python_passes(tmp_path):
    (tmp_path / "ok.py").write_text("print('hi')\n", encoding="utf-8")
    ok, detail = Verifier(tmp_path).verify()
    assert ok is True, detail


def test_syntax_error_fails(tmp_path):
    (tmp_path / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    ok, detail = Verifier(tmp_path).verify()
    assert ok is False
    assert "语法错误" in detail


def test_failing_test_fails(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 4\n", encoding="utf-8"
    )
    ok, detail = Verifier(tmp_path).verify()
    assert ok is False
    assert "测试失败" in detail


def test_empty_workdir_passes(tmp_path):
    ok, detail = Verifier(tmp_path).verify()
    assert ok is True
