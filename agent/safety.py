from __future__ import annotations

import re

# Destructive / irreversible commands that must be gated. These are the real
# risks of an agent with arbitrary code execution: mass deletion, force-pushing,
# privilege escalation, and bricking the machine.
_DANGEROUS_PATTERNS = [
    r"\brm\s+-[a-z]*[rf][a-z]*\b",       # rm -rf / rm -fr / rm -r
    r"\bgit\s+push\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[a-z]*f",
    r"\bgit\s+checkout\s+--\b",
    r"\bgit\s+rebase\b",
    r"\bsudo\b",
    r"\bsu\s+-",
    r"\b(shutdown|reboot|poweroff|mkfs|fdisk|dd|format)\b",
    r"\bchmod\s+-R\s+777\b",
    r"\bchown\b",
    r">\s*/dev/",
    r"\bdel\s+/[sq]\b",                   # Windows: del /s /q
    r"\brmdir\s+/[sq]\b",
]


class SafetyPolicy:
    """Graded safety policy (innovation 5).

    Tool `danger_level` tiers:
      - read/write  -> always allowed;
      - exec        -> allowed unless the command matches a destructive pattern;
      - (future: network/destructive -> could gate further).

    Deliberate choice: we do NOT gate network commands (curl/pip/git clone) — an
    agent must be able to install dependencies and fetch code to do its job. The
    genuinely irreversible risks are deletion, force-push, and privilege
    escalation, which is where the guardrail lives.

    A dangerous `exec` command is either confirmed by the user (interactive) or
    denied (non-interactive / one-shot), and the denial is fed back to the model
    as a recoverable error, not a crash.
    """

    def __init__(self, confirmer=None, auto_deny: bool = True):
        self.confirmer = confirmer  # callable(command: str) -> bool
        self.auto_deny = auto_deny

    def check(self, tool, args: dict) -> tuple[bool, str]:
        """Return (allowed, reason)."""
        if tool.danger_level in ("read", "write"):
            return True, ""
        if tool.danger_level == "exec":
            command = args.get("command", "")
            if not self._is_dangerous(command):
                return True, ""
            if self.confirmer is not None:
                return (True, "") if self.confirmer(command) else (False, "用户拒绝了该危险命令")
            if self.auto_deny:
                return False, "危险命令被安全策略自动拦截（未进入交互确认模式）"
            return True, ""
        return True, ""

    def _is_dangerous(self, command: str) -> bool:
        return any(re.search(p, command, re.IGNORECASE) for p in _DANGEROUS_PATTERNS)
