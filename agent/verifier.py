from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SKIP = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv", ".idea", ".mypy_cache"}


class Verifier:
    """Verify-before-done (innovation 1).

    When the model claims the task is done, we don't just trust it — we run a
    verification pass over the workspace it produced:
      1. syntax-check every .py file (python -m py_compile);
      2. if tests exist, run pytest.
    Any failure is fed back to the model so it can fix the problem, instead of
    silently shipping broken code. This is the guard against "the LLM confidently
    outputs wrong-but-plausible code".
    """

    def __init__(self, workdir: Path):
        self.workdir = workdir

    def verify(self) -> tuple[bool, str]:
        py_files = [
            f for f in self.workdir.rglob("*.py")
            if not any(part in _SKIP for part in f.parts)
        ]

        for f in py_files:
            r = subprocess.run(
                [sys.executable, "-m", "py_compile", str(f)],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                return False, f"语法错误 {f.name}: {r.stderr.strip()[:400]}"

        test_files = [f for f in py_files if f.name.startswith("test_") or f.name.endswith("_test.py")]
        tests_dir = self.workdir / "tests"
        has_tests = bool(test_files) or (tests_dir.is_dir() and any(tests_dir.rglob("*.py")))
        if has_tests:
            # 显式指定 `.`：避免 pytest 向上找到项目根目录的 pytest.ini 而误跑项目自身测试
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "."],
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                return False, f"测试失败:\n{(r.stdout or r.stderr).strip()[:800]}"

        return True, "验证通过"
