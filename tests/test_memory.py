from agent.memory import MemoryStore


def _seed(tmp_path):
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef multiply(a, b):\n    return a * b\n",
        encoding="utf-8",
    )
    (tmp_path / "string_utils.py").write_text(
        "def reverse_string(s):\n    return s[::-1]\n\n\ndef capitalize_words(s):\n    return s.title()\n",
        encoding="utf-8",
    )


def test_index_returns_chunks(tmp_path):
    _seed(tmp_path)
    m = MemoryStore(tmp_path)
    n = m.index()
    assert n >= 4  # 每个文件的函数 + header 分块


def test_search_returns_relevant(tmp_path):
    _seed(tmp_path)
    m = MemoryStore(tmp_path)
    m.index()
    results = m.search("add two numbers together", k=3)
    assert results, "应召回结果"
    for r in results:
        assert "path" in r and "text" in r and "score" in r
    # 语义检索应把 math_utils 里的 add 函数排在前面
    assert any("add" in r["text"] and "math_utils" in r["path"] for r in results)


def test_search_empty_workdir(tmp_path):
    m = MemoryStore(tmp_path)
    assert m.index() == 0
    assert m.search("anything") == []
