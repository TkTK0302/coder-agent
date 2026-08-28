# coding-agent

一个独立实现的编程智能体（coding agent）：通过与 LLM 交互，自主读写文件、执行命令、完成交给它的编程任务。**不依赖任何 agent 框架/SDK，核心逻辑全部自研。**

## 架构

核心循环 `loop.py` 不直接接触网络与文件系统，全部经分层模块隔离，便于单独测试与替换：

| 模块 | 职责 |
|------|------|
| `planner.py` | 计划阶段：任务拆解为结构化 todo |
| `llm.py` | 封装 DeepSeek（OpenAI 兼容协议），指数退避重试 |
| `context.py` | 上下文管理：token 预算 + 选择性压缩 |
| `memory.py` | RAG 长期记忆：AST 分块 + 向量检索 |
| `sandbox.py` | 执行沙盒：宿主机 / Docker 容器隔离 |
| `safety.py` | 分级安全策略：危险命令拦截 |
| `verifier.py` | 验证阶段：完成前跑语法检查 + 测试 |
| `todo.py` | 任务清单进度追踪 |
| `trace.py` | 逐步打印 + 可重放 JSONL 轨迹 |
| `loop.py` | 核心循环 + 多条件终止 |
| `tools/` | 20 个工具（文件/终端/Git/环境/AST 导航/提问） |

## 六个设计亮点

1. **计划→执行**：先产出结构化 todo，逐步勾选、追踪进度。
2. **验证后才算完成**：声称完成前自动做语法检查 + 跑测试，失败则继续修。
3. **选择性上下文压缩**：保留「系统提示 + 任务 + 最近消息」，中间摘要化，而非无脑丢最旧。
4. **工具结果信封**：统一 `status/output/hint` 结构，让模型据错误信息自主恢复。
5. **分级安全**：危险命令（`rm -rf`、`git push`、`sudo` 等）执行前需确认。
6. **可重放 trace**：每步留痕到 JSONL，可回放、可调试。

## 更多能力

- **Docker 沙盒**：`AGENT_SANDBOX=docker` 走容器隔离（默认 `host` 兜底）。
- **实时终端**：`start_command` / `check_command` / `stop_command` 支持长时进程。
- **RAG 长期记忆**：`search_code` 语义检索 + 任务开始自动召回相关片段。
- **Git 工具**：`git_status` / `git_diff` / `git_log` / `git_commit` / `git_restore`（试错 + 回滚）。
- **AST 导航**：`list_symbols` / `find_definition`（非字符串匹配）。
- **环境感知**：`env_info`。
- **主动提问**：需求模糊时 `ask_user` 澄清，而非瞎猜。
- **语义死循环检测**：识别「改→错→改→错」这类非字面重复的循环。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env          # 填入 DEEPSEEK_API_KEY
python main.py "你的任务" --workdir ./workdir   # 一次性模式
python main.py                                   # 交互模式
python -m pytest -q                              # 测试
```

## 目录结构

```
agent/
  planner.py     # 计划阶段
  llm.py         # LLM 客户端
  context.py     # 上下文管理
  memory.py      # RAG 长期记忆
  sandbox.py     # 执行沙盒
  schema.py      # 工具结果信封
  safety.py      # 安全策略
  verifier.py    # 验证阶段
  todo.py        # 任务清单
  trace.py       # 可观测 + 轨迹
  loop.py        # 核心循环
  tools/         # 工具集
main.py          # CLI 入口
tests/           # 测试
```
