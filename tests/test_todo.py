from agent.todo import TodoList


def test_reset_and_update():
    t = TodoList()
    t.reset(["分析代码", "实现逻辑", "编写测试"])
    assert len(t.items) == 3
    assert t.is_all_done() is False
    t.update(1, "done")
    t.update(2, "in_progress")
    assert t.done_count() == 1
    assert t.items[0]["status"] == "done"


def test_update_all_done():
    t = TodoList()
    t.reset(["a", "b"])
    t.update(1, "done")
    t.update(2, "done")
    assert t.is_all_done() is True


def test_update_invalid_id():
    t = TodoList()
    t.reset(["a"])
    assert t.update(99, "done") is False


def test_to_prompt_has_marks():
    t = TodoList()
    t.reset(["a", "b"])
    t.update(1, "done")
    prompt = t.to_prompt()
    assert "[x] 1. a" in prompt
    assert "[ ] 2. b" in prompt
