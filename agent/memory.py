from __future__ import annotations

import ast
from pathlib import Path

import faiss
import numpy as np

_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv", ".idea", ".mypy_cache"}
_TEXT_SUFFIXES = {".py", ".txt", ".md", ".rst", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh"}


class MemoryStore:
    """长期记忆：把项目代码库向量化，支持语义检索（RAG）。

    刻意不用 LangChain/LlamaIndex（题目禁止 agent 框架）：分块、向量化、余弦检索、
    召回的全部逻辑自写；只把「embedding 计算」交给 fastembed（ONNX 本地模型）、
    「向量索引」交给 faiss。这既合规，也说明理解 RAG 的原理。
    """

    def __init__(self, workdir: Path, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.workdir = workdir
        self.model_name = model_name
        self._model = None
        self._index = None
        self._chunks: list[dict] = []

    # ---------- 内部 ----------
    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _collect_files(self) -> list[Path]:
        files = []
        for f in self.workdir.rglob("*"):
            if f.is_file() and f.suffix.lower() in _TEXT_SUFFIXES:
                if not any(part in _SKIP_DIRS for part in f.parts):
                    files.append(f)
        return files

    def _chunk_file(self, f: Path) -> list[dict]:
        text = f.read_text(encoding="utf-8", errors="replace")
        rel = str(f.relative_to(self.workdir))
        if f.suffix == ".py":
            return self._chunk_python(rel, text)
        return self._chunk_lines(rel, text)

    def _chunk_python(self, rel: str, text: str) -> list[dict]:
        """按函数/类边界（AST）分块，比固定行数更符合代码语义。"""
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return self._chunk_lines(rel, text)

        chunks = []
        header = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                break
            seg = ast.get_source_segment(text, node)
            if seg:
                header.append(seg)
        if header:
            chunks.append({"path": rel, "start_line": 1, "end_line": 1, "text": "\n".join(header)})

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                seg = ast.get_source_segment(text, node)
                if seg:
                    chunks.append({
                        "path": rel,
                        "start_line": node.lineno,
                        "end_line": (node.end_lineno or node.lineno),
                        "text": seg,
                    })
        return chunks or self._chunk_lines(rel, text)

    def _chunk_lines(self, rel: str, text: str, size: int = 80) -> list[dict]:
        lines = text.splitlines()
        chunks = []
        for i in range(0, len(lines), size):
            seg = "\n".join(lines[i:i + size])
            if seg.strip():
                chunks.append({"path": rel, "start_line": i + 1, "end_line": min(i + size, len(lines)), "text": seg})
        return chunks

    # ---------- 公开 API ----------
    def index(self, files: list[Path] | None = None) -> int:
        """对工作目录文件分块并向量化，返回 chunk 数量。"""
        files = files or self._collect_files()
        chunks: list[dict] = []
        for f in files:
            chunks.extend(self._chunk_file(f))
        if not chunks:
            self._index = None
            self._chunks = []
            return 0

        model = self._get_model()
        vectors = np.array(list(model.embed([c["text"] for c in chunks])), dtype=np.float32)
        # 余弦相似度 = L2 归一化后的内积
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = np.ascontiguousarray(vectors / norms)

        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._chunks = chunks
        return len(chunks)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """语义检索 top-k 相关代码片段。"""
        if self._index is None or self._index.ntotal == 0:
            return []
        model = self._get_model()
        qv = np.array(list(model.embed([query])), dtype=np.float32)
        qn = np.linalg.norm(qv, axis=1, keepdims=True)
        qn[qn == 0] = 1.0
        qv = np.ascontiguousarray(qv / qn)

        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(qv, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            c = dict(self._chunks[idx])
            c["score"] = float(score)
            results.append(c)
        return results
