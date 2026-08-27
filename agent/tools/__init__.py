from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

from pydantic import BaseModel

from agent.schema import ToolResult


class Tool(ABC):
    """Base class for all tools.

    A tool exposes:
      - `name` / `description`   -> passed to the model so it knows when to use it
      - `args_model` (pydantic)  -> single source of truth: its JSON schema is
        sent to the LLM, and the same model validates incoming arguments.
      - `run()`                  -> local execution, returns a ToolResult envelope.
      - `danger_level`           -> graded safety level, consumed by the safety
        layer (milestone M5): read / write / exec / network / destructive.
    """

    name: str = ""
    description: str = ""
    args_model: Type[BaseModel] = BaseModel
    danger_level: str = "read"

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir

    @abstractmethod
    def run(self, args: BaseModel) -> ToolResult:
        ...

    def to_spec(self) -> dict:
        schema = self.args_model.model_json_schema()
        # Strip pydantic-specific keys that some providers reject; enforce a
        # closed object schema so the model doesn't invent extra fields.
        schema.pop("title", None)
        schema.setdefault("type", "object")
        schema["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def specs(self) -> list[dict]:
        return [tool.to_spec() for tool in self._tools.values()]

    def execute(self, name: str, raw_args: str | dict) -> ToolResult:
        """Validate raw arguments against the tool's schema, then run it.

        Every failure path — unknown tool, malformed JSON, invalid arguments, or a
        crashing tool — is converted into an error envelope rather than raising, so
        the loop can feed it back to the model to recover. This method never raises.
        """
        try:
            tool = self.get(name)
        except KeyError:
            return ToolResult.error(f"未知工具: {name}", hint="只调用已提供的工具")

        try:
            args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError) as exc:
            return ToolResult.error(f"工具参数不是合法 JSON: {exc}", hint="以合法 JSON 重试")

        try:
            args = tool.args_model.model_validate(args_dict)
        except Exception as exc:
            return ToolResult.error(f"参数校验失败: {exc}", hint="检查参数 schema 后重试")

        try:
            return tool.run(args)
        except Exception as exc:
            return ToolResult.error(
                f"工具执行失败: {exc}", hint="inspect the error and recover"
            )
