编程智能体（coding agent）

一个独立实现的编程智能体：通过与 LLM 交互，自主读写文件、执行命令、完成编程任务。未使用任何 agent 框架/SDK，核心逻辑全部自研。

【架构】核心循环不直接接触网络与文件系统，经分层模块隔离：
- planner：计划阶段，把任务拆成分步计划
- llm：封装 DeepSeek（OpenAI 兼容协议），指数退避重试
- context：token 预算 + 选择性上下文压缩
- tools：read_file / write_file / edit_file / list_files / search / run_command
- schema：工具结果信封（status/output/hint）
- safety：分级安全策略，危险命令拦截
- verifier：验证阶段，完成前跑语法检查 + 测试
- trace：逐步打印 + 可回放 JSONL 轨迹
- loop：observe→decide→act 核心循环

【关键设计决策】
1. 计划→执行：先让模型产出分步计划，再逐步执行
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

【仓库地址】
https://github.com/TkTK0302/coder-agent
