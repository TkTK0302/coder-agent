from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ToolResult:
    """Structured envelope every tool returns (innovation 4).

    A uniform shape (status + output + hint) means the LLM always receives
    consistent feedback, so it can reason about errors instead of parsing ad-hoc
    strings. `hint` nudges the model toward recovery without hard-coding it.
    """

    status: Literal["success", "error"] = "success"
    output: str = ""
    hint: str | None = None

    @classmethod
    def error(cls, output: str, hint: str | None = None) -> "ToolResult":
        return cls(status="error", output=output, hint=hint)

    def to_content(self) -> str:
        parts = [f"status: {self.status}"]
        if self.output:
            parts.append(self.output.strip())
        if self.hint:
            parts.append(f"hint: {self.hint}")
        return "\n".join(parts)
