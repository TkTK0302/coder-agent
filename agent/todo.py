from __future__ import annotations

_STATUS_MARKS = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]"}


class TodoList:
    """任务清单状态机：追踪"完成了什么、正在做哪一步、下一步干什么"。"""

    def __init__(self):
        self.items: list[dict] = []

    def reset(self, steps: list[str]) -> None:
        self.items = [{"id": i + 1, "title": s, "status": "pending"} for i, s in enumerate(steps)]

    def update(self, todo_id: int, status: str) -> bool:
        for item in self.items:
            if item["id"] == todo_id:
                item["status"] = status
                return True
        return False

    def to_prompt(self) -> str:
        lines = ["当前任务清单（用 update_todo 更新进度）："]
        for item in self.items:
            mark = _STATUS_MARKS.get(item["status"], "[ ]")
            lines.append(f"{mark} {item['id']}. {item['title']}")
        return "\n".join(lines)

    def is_all_done(self) -> bool:
        return bool(self.items) and all(item["status"] == "done" for item in self.items)

    def done_count(self) -> int:
        return sum(1 for item in self.items if item["status"] == "done")
