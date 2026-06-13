# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
uv sync                     # 安装依赖
uv run agent-code "prompt"  # 单次模式
uv run agent-code           # 交互 REPL 模式
uv run black agent_code/    # 格式化代码
uv run pytest               # 运行测试
uv run pytest tests/test_foo.py::test_bar  # 运行单个测试
```

## 架构

**入口 → Agent Loop → 工具调用 → 权限门 → 执行**

`cli.py` → `main_command()` 解析 CLI 参数，创建 `AnthropicProvider`，进入单次或 REPL 模式。所有输入走 `run_user_input()`：先分发 slash 命令，未命中则进入 Agent Loop。

`agent.py` → `run_agent()` 核心循环：调用 `provider.complete()`，通过 `permissions.py` 处理工具调用的权限判定，经 `tools.py` 执行，结果持久化到 session。消息超过 40 条自动压缩（确定性算法，不调用 LLM — `compact_basic.py`）。

`model.py` → `AnthropicProvider` 封装 Anthropic SDK，支持自定义 `base_url`（默认指向 DeepSeek 的 Anthropic 兼容端点）。通过 `Provider` 协议实现可替换性。

`tools.py` → 17 个工具通过 `@register_tool` 装饰器注册。分为只读（自动放行）、写入（default 模式需确认）、交互（始终需确认）三类。关键工具：`file_write`、`file_edit`、`bash`、`read_file`、`web_fetch`、`web_search`、`memory_write`、`memory_recall`。

`permissions.py` → 三种模式：`default`（写入需确认）、`acceptEdits`（自动放行文件编辑）、`plan`（拒绝所有写入）。危险 bash 命令通过正则检测。

`fs_safety.py` → 路径沙箱（在 cwd 内解析）、二进制检测、256KB 读取限制、写前必须先读、mtime 冲突检测。`SkipPolicy` 遵循 `.gitignore`。

`session.py` → 追加写入 JSONL，存储在 `.agent/sessions/`。12 位 hex session ID。支持 `--resume` 和 `--continue`。

`slash.py` → REPL 命令注册表模式（`/help`、`/model`、`/context`、`/compact`、`/permissions`、`/plan`）。通过 `register()` 扩展。

`memdir/` → 长期记忆存储在 `.agent/memory/`，YAML frontmatter 格式文件，`MEMORY.md` 作为索引。`write_memory()` + `recall_memory()` 基于关键词评分检索。

## 约定

- Python 3.12+，写完整 type hints
- 中文注释和文档字符串
- `.env` 从包目录（`agent_code/.env`）加载，不依赖 cwd
- 测试框架：pytest，放在 `tests/` 目录
- 格式化：Black（target py312）
- 必需环境变量：`ANTHROPIC_AUTH_TOKEN`；可选：`ANTHROPIC_BASE_URL`（默认 DeepSeek）
- 项目根目录的 `AGENT.md` 会注入 system prompt 作为项目规则
