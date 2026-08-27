# coding-agent

一个独立实现的编程智能体（coding agent）：通过与 LLM 交互，自主读写文件、执行命令，完成交给它的编程任务。**不依赖任何 agent 框架/SDK，核心逻辑全部自研。**

## 架构

核心循环 `loop.py` 不直接接触网络与文件系统，全部经分层模块隔离，便于单独测试与替换：

| 模块 | 职责 |
|------|------|
| `planner.py` | 计划阶段：把任务拆成分步计划 |
| `llm.py` | 封装 DeepSeek（OpenAI 兼容协议），指数退避重试 |
| `context.py` | 上下文管理：token 预算 + 选择性压缩 |
| `tools/` | read_file / write_file / edit_file / list_files / search / run_command |
| `schema.py` | 工具结果信封 + 参数校验 |
| `safety.py` | 分级安全策略：危险命令拦截 |
| `verifier.py` | 验证阶段：完成前跑语法检查 + 测试 |
| `trace.py` | 逐步打印 + 可重放 JSONL 轨迹 |
| `loop.py` | observe→decide→act 核心循环 + 多条件终止 |

## 六个设计亮点

1. **计划→执行**：先让模型产出分步计划，再逐步执行。
2. **验证后才算完成**：模型声称完成时，自动对产出做语法检查 + 跑测试，失败则继续修。
3. **选择性上下文压缩**：超预算时保留「系统提示 + 任务 + 最近消息」，中间摘要化，而非无脑丢最旧。
4. **工具结果信封**：统一 `status/output/hint` 结构，让模型据错误信息自主恢复。
5. **分级安全**：危险命令（`rm -rf`、`git push`、`sudo` 等）执行前需确认。
6. **可重放 trace**：每步留痕到 JSONL，可回放、可调试。

核心逻辑同样自研：原生 function calling（pydantic 单一 schema）、三重终止条件、指数退避错误处理。

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
  schema.py      # 工具结果信封
  safety.py      # 安全策略
  verifier.py    # 验证阶段
  trace.py       # 可观测 + 轨迹
  loop.py        # 核心循环
  tools/         # 工具集
main.py          # CLI 入口
tests/           # 测试
```
