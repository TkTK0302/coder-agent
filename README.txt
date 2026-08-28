编程智能体（coding agent）

一个独立实现的编程智能体：通过与 LLM 交互，自主读写文件、执行命令、完成编程任务。未使用任何 agent 框架/SDK，核心逻辑全部自研。

【架构】核心循环不直接接触网络与文件系统，经分层模块隔离：
- planner：任务拆解为结构化 todo（计划阶段）
- llm：封装 DeepSeek（OpenAI 兼容），指数退避重试
- context：token 预算 + 选择性上下文压缩
- memory：RAG 长期记忆（AST 分块 + 向量检索）
- sandbox：执行沙盒（宿主机 / Docker 容器）
- safety：分级安全，危险命令拦截
- verifier：验证阶段，完成前跑语法检查 + 测试
- todo：任务清单进度追踪
- trace：逐步打印 + 可回放轨迹
- loop：核心循环 + 多条件终止
- tools：20 个工具（文件/终端/Git/环境/AST 导航/提问）

【关键设计决策】
1. 计划→执行：结构化 todo + update_todo 追踪进度
2. 验证后才算完成：声称完成时先跑语法检查 + 测试，失败继续修
3. 上下文超预算时保留系统提示+任务+最近消息，中间摘要压缩
4. 工具报错以结构化信封回传，让模型参与恢复
5. 危险命令（rm -rf、git push、sudo 等）执行前需确认
6. 每步留痕到 JSONL，可回放调试

【运行】
pip install -r requirements.txt
复制 .env.example 为 .env，填入 DEEPSEEK_API_KEY
python main.py "任务描述" --workdir ./workdir   # 一次性模式
python main.py                                  # 交互模式
python -m pytest -q                             # 测试
# Docker 沙盒（需先装 Docker Desktop）：
#   set AGENT_SANDBOX=docker 后运行

【仓库地址】
https://github.com/TkTK0302/coder-agent
