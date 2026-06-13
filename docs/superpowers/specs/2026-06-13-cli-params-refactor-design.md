# CLI 参数重构：修复 cli.py 未定义变量和签名不匹配

## 问题

`cli.py` 的 `run_user_input` 引用了 `main_command` 作用域中不存在的变量（`provider`, `model`, `base_url`, `max_steps`），`run_once` 调用签名与定义不匹配，REPL 分支引用了不存在的 `handle_slash` 函数。

根因：添加 slash command 框架时，CLI 入参没有同步完成。

## 方案：最小改动（方案 A）

提升 `AnthropicProvider` 创建到 `main_command`，新增 CLI flags，修复调用链。

## 改动

### 1. `main_command` 新增 CLI flags

```python
model: str = typer.Option("glm-5.1", "--model", "-m", help="Model name"),
base_url: str | None = typer.Option(None, "--base-url", help="API base URL override"),
max_steps: int = typer.Option(99, "--max-steps", help="Max agent loop steps"),
```

在函数体内创建 provider 和推导 provider_name：

```python
llm_logger = create_llm_logger(log_dir)
provider = AnthropicProvider(model=model, base_url=base_url, llm_logger=llm_logger)
provider_name = _provider_name(provider.base_url)
```

### 2. `run_once` 签名变更

```python
def run_once(
    prompt: str,
    cwd: Path,
    provider: AnthropicProvider,
    max_steps: int,
    permission_mode: Literal["default", "acceptEdits", "plan"],
    session: Session | None = None,
    system_prompt: str | None = None,
) -> None:
```

- 接收外部传入的 `provider`（不再内部创建）
- 新增 `max_steps` 参数，传给 `run_agent`
- 移除 `log_dir` 参数（llm_logger 在外层创建）

### 3. `run_user_input` 修复

SlashContext 构造：

```python
SlashContext(
    cwd=resolved_cwd,
    permission_mode=permission_mode,
    model=model,
    provider=provider_name,
    session_id=session.session_id if session else None,
)
```

run_once 调用统一为：

```python
run_once(line, resolved_cwd, provider, max_steps, permission_mode, session=session, system_prompt=system_prompt)
```

### 4. REPL 循环修复

删除 `handle_slash` 引用，所有输入统一走 `run_user_input`：

```python
while True:
    line = console.input("[bold]>[/bold] ").strip()
    if not line:
        continue
    if line == "/exit":
        console.print("Bye.")
        return
    run_user_input(line)
```

### 5. 新增 `_provider_name` 辅助函数

```python
from urllib.parse import urlparse

def _provider_name(base_url: str) -> str:
    host = urlparse(base_url).hostname or "unknown"
    parts = host.split(".")
    return parts[-2] if len(parts) >= 2 else host
```

## 不改动的文件

- `slash.py` — SlashContext 结构不变
- `model.py` — AnthropicProvider 不变
- `agent.py` — run_agent 不变
