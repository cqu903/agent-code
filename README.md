# agent-code

基于 Typer 的 CLI agent，支持一次性 prompt 与 REPL 交互模式。

## 安装

```bash
uv sync
```

## 配置

复制环境变量模板并填入 API 凭证：

```bash
cp agent_code/.env.example agent_code/.env
```

## 运行

```bash
uv run agent-code "用 echo 工具说 hello"
```

或进入交互模式：

```bash
uv run agent-code
```
