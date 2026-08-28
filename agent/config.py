from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — avoids an extra dependency.

    Only sets variables that are not already present in the environment, so a
    real environment variable always wins over the file. This is the single
    place where credentials are read; they are never hard-coded.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class Config:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_iters: int = 30
    max_retries: int = 3
    max_token_budget: int = 64_000
    workdir: Path = field(default_factory=Path.cwd)
    sandbox_mode: str = "host"  # host | docker
    docker_image: str = "python:3.12-slim"

    @classmethod
    def from_env(cls, workdir: Path | None = None) -> "Config":
        _load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "未找到 DEEPSEEK_API_KEY：请设置环境变量，或复制 .env.example 为 .env 并填入 key。"
            )
        return cls(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            max_iters=int(os.environ.get("AGENT_MAX_ITERS", "30")),
            max_retries=int(os.environ.get("AGENT_MAX_RETRIES", "3")),
            max_token_budget=int(os.environ.get("AGENT_TOKEN_BUDGET", "64000")),
            workdir=workdir or Path.cwd(),
            sandbox_mode=os.environ.get("AGENT_SANDBOX", "host"),
            docker_image=os.environ.get("DOCKER_IMAGE", "python:3.12-slim"),
        )
