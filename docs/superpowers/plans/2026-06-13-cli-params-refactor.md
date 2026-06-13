# CLI 参数重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix undefined variable references and signature mismatches in `cli.py` by promoting `AnthropicProvider` creation to `main_command` and adding missing CLI flags.

**Architecture:** Move provider construction out of `run_once` into `main_command`. Add `--model`, `--base-url`, `--max-steps` CLI options. Fix `run_once` signature, `SlashContext` construction, and REPL loop to use the promoted variables.

**Tech Stack:** Python 3.12, typer, rich, anthropic SDK

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `agent_code/cli.py` | Modify | All changes live here: new CLI flags, `_provider_name` helper, `run_once` signature, `run_user_input` fix, REPL fix |

No other files are modified.

---

### Task 1: Add `_provider_name` helper and update imports

**Files:**
- Modify: `agent_code/cli.py:1-15` (imports) and add new function after `render_header`

- [ ] **Step 1: Add `urlparse` import**

Change line 3 from:
```python
from pathlib import Path
```
to:
```python
from pathlib import Path
from urllib.parse import urlparse
```

- [ ] **Step 2: Add `_provider_name` function after `render_header`**

Insert after line 30:

```python
def _provider_name(base_url: str) -> str:
    host = urlparse(base_url).hostname or "unknown"
    parts = host.split(".")
    return parts[-2] if len(parts) >= 2 else host
```

- [ ] **Step 3: Verify no syntax errors**

Run: `python -c "from agent_code.cli import _provider_name; print(_provider_name('https://api.deepseek.com/anthropic'))"`
Expected: `deepseek`

- [ ] **Step 4: Commit**

```bash
git add agent_code/cli.py
git commit -m "Add _provider_name helper to extract display name from base_url"
```

---

### Task 2: Update `run_once` signature — accept `provider` and `max_steps`, remove `log_dir`

**Files:**
- Modify: `agent_code/cli.py:33-56`

- [ ] **Step 1: Replace `run_once` function**

Replace the entire `run_once` function (lines 33-56) with:

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
    if session:
        suffix = " (resumed)" if session.resumed else ""
        console.print(f"[dim]session: {session.session_id}{suffix}[/dim]")
    run_agent(
        prompt,
        provider,
        tool_registry,
        max_steps=max_steps,
        cwd=cwd,
        permission_mode=permission_mode,
        session=session,
        system_prompt=system_prompt,
    )
```

Key changes:
- Removed `log_dir` parameter (llm_logger created externally now)
- Added `provider: AnthropicProvider` parameter (replaces internal creation)
- Added `max_steps: int` parameter (passed to `run_agent`)
- Removed internal `llm_logger` and `provider` creation

- [ ] **Step 2: Verify no import errors**

Run: `python -c "from agent_code.cli import run_once; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add agent_code/cli.py
git commit -m "Update run_once signature: accept provider and max_steps from caller"
```

---

### Task 3: Add CLI flags and promote provider creation to `main_command`

**Files:**
- Modify: `agent_code/cli.py:59-95` (`main_command` function)

- [ ] **Step 1: Add new CLI options to `main_command` signature**

Replace the `main_command` function definition (lines 59-79) with:

```python
def main_command(
    prompt: str = typer.Argument("", help="Prompt to send to the agent."),
    cwd: Path = typer.Option(Path.cwd(), "--cwd", "-C"),
    log_dir: Path | None = typer.Option(
        None,
        "--log-dir",
        help="Directory for LLM request/response JSONL logs.",
        envvar="AGENT_CODE_LOG_DIR",
    ),
    permission_mode: Literal["default", "acceptEdits", "plan"] = typer.Option(
        "default",
        "--permission-mode",
        help="Permission mode: default, acceptEdits, plan",
    ),
    model: str = typer.Option(
        "glm-5.1", "--model", "-m", help="Model name."
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", help="API base URL override."
    ),
    max_steps: int = typer.Option(
        99, "--max-steps", help="Max agent loop steps."
    ),
    resume: str | None = typer.Option(
        None, "--resume", help="按 session id 恢复指定会话"
    ),
    continue_: bool = typer.Option(
        False, "--continue", "-c", help="回复 cwd 最近一次会话"
    ),
) -> None:
```

- [ ] **Step 2: Replace `main_command` body after `text = prompt.strip()`**

Replace lines 93-95 (`text = prompt.strip()` through `system_prompt = build_system_prompt(resolved_cwd)`) with:

```python
    text = prompt.strip()

    llm_logger = create_llm_logger(log_dir)
    if llm_logger:
        console.print(f"[dim]llm log: {llm_logger.path}[/dim]\n")
    provider = AnthropicProvider(
        model=model, base_url=base_url, llm_logger=llm_logger
    )
    provider_name = _provider_name(provider.base_url)
    system_prompt = build_system_prompt(resolved_cwd)
```

- [ ] **Step 3: Verify CLI flags work**

Run: `python -m agent_code.cli --help`
Expected: output shows `--model`, `--base-url`, `--max-steps` options

- [ ] **Step 4: Commit**

```bash
git add agent_code/cli.py
git commit -m "Add --model, --base-url, --max-steps CLI flags and promote provider creation"
```

---

### Task 4: Fix `run_user_input` — SlashContext construction and `run_once` calls

**Files:**
- Modify: `agent_code/cli.py:97-145` (the `run_user_input` inner function)

- [ ] **Step 1: Replace the entire `run_user_input` function**

Replace the `run_user_input` function with:

```python
    def run_user_input(line: str) -> None:
        """
        统一处理用户输入：先走 slash dispatch，未命中再进入 Agent Loop。
        REPL 用户输入和 cron pending prompt 都必须走这个入口。
        """
        nonlocal session
        slash_result = dispatch_slash(
            line,
            SlashContext(
                cwd=resolved_cwd,
                permission_mode=permission_mode,
                model=model,
                provider=provider_name,
                session_id=session.session_id if session else None,
            ),
        )
        if slash_result.handled:
            if slash_result.message:
                console.print(slash_result.message)
            if slash_result.should_query:
                if session is None:
                    session = Session.create(resolved_cwd)
                run_once(
                    slash_result.prompt,
                    resolved_cwd,
                    provider,
                    max_steps,
                    permission_mode,
                    session=session,
                    system_prompt=system_prompt,
                )
            return
        if session is None:
            session = Session.create(resolved_cwd)
        run_once(
            line,
            resolved_cwd,
            provider,
            max_steps,
            permission_mode,
            session=session,
            system_prompt=system_prompt,
        )
```

Key fixes:
- `SlashContext(provider=provider_name)` — uses derived string, not undefined variable
- `SlashContext(model=model)` — now references the CLI flag
- Both `run_once` calls match the new signature: `(prompt, cwd, provider, max_steps, permission_mode, session, system_prompt)`

- [ ] **Step 2: Verify no syntax errors**

Run: `python -c "from agent_code.cli import main_command; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add agent_code/cli.py
git commit -m "Fix run_user_input: correct SlashContext and run_once call signatures"
```

---

### Task 5: Fix REPL loop — remove `handle_slash` reference

**Files:**
- Modify: `agent_code/cli.py` (the REPL while loop at end of `main_command`)

- [ ] **Step 1: Replace REPL loop**

Replace the REPL loop (the `if text:` block through the end of `main_command`) with:

```python
    if text:
        run_user_input(text.strip())
        return
    # REPL 分支——命令后没跟 prompt，走交互循环
    render_header(resolved_cwd)
    if session:
        suffix = " (resumed)" if session.resumed else ""
        console.print(f"[dim]session: {session.session_id}{suffix}[/dim]")
    console.print("输入 /help 查看命令，输入 /exit 退出。")
    while True:
        line = console.input("[bold]>[/bold] ").strip()
        if not line:
            continue
        if line == "/exit":
            console.print("Bye.")
            return
        run_user_input(line)
```

Key fix: removed `if line.startswith("/") and handle_slash(line):` — all input goes through `run_user_input`, which handles slash dispatch internally.

- [ ] **Step 2: Verify full file parses**

Run: `python -c "from agent_code.cli import main; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add agent_code/cli.py
git commit -m "Fix REPL loop: remove handle_slash, route all input through run_user_input"
```

---

### Task 6: Smoke test

- [ ] **Step 1: Verify --help shows all new flags**

Run: `python -m agent_code.cli --help`
Expected: shows `--model`, `--base-url`, `--max-steps` alongside existing flags

- [ ] **Step 2: Verify _provider_name edge cases**

Run:
```bash
python -c "
from agent_code.cli import _provider_name
assert _provider_name('https://api.deepseek.com/anthropic') == 'deepseek'
assert _provider_name('https://api.anthropic.com') == 'anthropic'
assert _provider_name('http://localhost:1234') == 'localhost'
assert _provider_name('unknown') == 'unknown'
print('all passed')
"
```
Expected: `all passed`
