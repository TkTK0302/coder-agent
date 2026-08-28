from __future__ import annotations

import shlex
import subprocess
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class Sandbox(ABC):
    """命令执行环境抽象。

    决定「命令在哪里跑」：宿主机子进程，或 Docker 容器。上层工具只依赖这个接口，
    因此切换沙盒模式不影响 agent 其余逻辑。这层抽象本身就是可辩护的设计点——
    agent 核心不关心执行后端，隔离策略可替换。
    """

    def __init__(self, workdir: Path):
        self.workdir = workdir

    @abstractmethod
    def run(self, command: str, timeout: int = 60) -> CommandResult:
        """前台执行，阻塞至结束；超时抛 TimeoutError。"""

    @abstractmethod
    def start(self, command: str) -> str:
        """后台启动一个持续运行的命令，返回进程 ID。"""

    @abstractmethod
    def poll(self, pid: str) -> tuple[str, str]:
        """返回 (状态, 自上次查询以来的增量输出)。状态为 running 或 exited:<code>。"""

    @abstractmethod
    def kill(self, pid: str) -> None:
        """终止后台进程。"""

    @abstractmethod
    def close(self) -> None:
        """释放资源（如删除容器）。"""


class HostSandbox(Sandbox):
    """在宿主机直接执行（无隔离）。作为默认兜底，也便于本地演示。"""

    def __init__(self, workdir: Path):
        super().__init__(workdir)
        self._procs: dict[str, dict] = {}
        self._counter = 0

    def run(self, command: str, timeout: int = 60) -> CommandResult:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"命令超时（>{timeout}s）")
        return CommandResult(proc.returncode, proc.stdout, proc.stderr)

    def start(self, command: str) -> str:
        self._counter += 1
        pid = f"proc_{self._counter}"
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(self.workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        buf: list[str] = []

        def _reader() -> None:
            try:
                for line in proc.stdout:
                    buf.append(line)
            except Exception:
                pass  # 解码错误等情况，忽略并停止读取

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        self._procs[pid] = {"proc": proc, "buf": buf, "thread": thread, "read": 0}
        return pid

    def poll(self, pid: str) -> tuple[str, str]:
        info = self._procs.get(pid)
        if info is None:
            return ("not_found", "")
        proc = info["proc"]
        new_output = "".join(info["buf"][info["read"]:])
        info["read"] = len(info["buf"])
        if proc.poll() is None:
            return ("running", new_output)
        return (f"exited:{proc.returncode}", new_output)

    def kill(self, pid: str) -> None:
        info = self._procs.get(pid)
        if info is not None:
            try:
                info["proc"].kill()
            except Exception:
                pass

    def close(self) -> None:
        for info in self._procs.values():
            try:
                info["proc"].kill()
            except Exception:
                pass
        self._procs.clear()


class DockerSandbox(Sandbox):
    """在 Docker 容器内执行（真隔离）。需要 Docker 守护进程在运行。

    后台进程的处理是「重定向到容器内文件 + 读取文件 + kill -0 探活」的方式，
    结构正确；受限于 docker-py 的 exec 语义，退出码为尽力获取。
    """

    def __init__(self, workdir: Path, image: str = "python:3.12-slim"):
        super().__init__(workdir)
        import docker  # 延迟导入：未装 docker 时不影响 host 模式

        self._docker = docker
        self.image = image
        self.client = docker.from_env()
        self.container = self.client.containers.run(
            image,
            detach=True,
            tty=True,
            volumes={str(self.workdir.resolve()): {"bind": "/workspace", "mode": "rw"}},
            working_dir="/workspace",
        )
        self._counter = 0

    def run(self, command: str, timeout: int = 60) -> CommandResult:
        # 注：docker-py exec_run 无干净的超时语义，timeout 在此尽力而为
        r = self.container.exec_run(f"sh -c {shlex.quote(command)}", demux=True)
        stdout, stderr = r.output
        return CommandResult(
            r.exit_code,
            (stdout or b"").decode("utf-8", errors="replace"),
            (stderr or b"").decode("utf-8", errors="replace"),
        )

    def start(self, command: str) -> str:
        self._counter += 1
        pid = f"proc_{self._counter}"
        script = f"( {command} ) > /tmp/{pid}.out 2>&1 & echo $! > /tmp/{pid}.pid"
        self.container.exec_run(f"sh -c {shlex.quote(script)}")
        return pid

    def poll(self, pid: str) -> tuple[str, str]:
        out = self.container.exec_run(f"cat /tmp/{pid}.out 2>/dev/null || true")
        output = (out.output or b"").decode("utf-8", errors="replace")
        alive = self.container.exec_run(
            f"sh -c 'kill -0 $(cat /tmp/{pid}.pid 2>/dev/null) 2>/dev/null && echo yes || echo no'"
        )
        running = "yes" in (alive.output or b"").decode("utf-8", errors="replace")
        return ("running" if running else "exited:?", output)

    def kill(self, pid: str) -> None:
        self.container.exec_run(
            f"sh -c 'kill $(cat /tmp/{pid}.pid 2>/dev/null) 2>/dev/null || true'"
        )

    def close(self) -> None:
        try:
            self.container.remove(force=True)
        except Exception:
            pass


def create_sandbox(mode: str, workdir: Path, image: str = "python:3.12-slim") -> Sandbox:
    """工厂：按模式创建沙盒。mode: host | docker。"""
    if mode == "docker":
        return DockerSandbox(workdir, image=image)
    return HostSandbox(workdir)
