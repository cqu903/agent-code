from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pathspec

# 文本文件后缀白名单: 直接放行，不用peek文件夹
TEXT_SUFFIXES = {
    ".py",
    ".pyi",
    ".md",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".cfg",
    ".ini",
    ".env",
    ".sh",
    ".bash",
    ".zsh",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".sql",
    ".lock",
    ".gitignore",
}

# 单文件大小上限，超过就拒绝读取整个文件
MAX_READ_BYTE = 256 * 1024
# 单次工具 observation 上限。模型上下文有限，过长直接截尾。
DEFAULT_MAX_CHARS = 8000
# 默认跳过的目录名。任意祖先目录命中名单，整条路径都被剔除。
DEFAULT_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)


@dataclass
class SkipPolicy:
    # 默认skip集合
    skip_dirs: frozenset[str] = DEFAULT_SKIP_DIRS
    gitignore: pathspec.PathSpec | None = None

    @classmethod
    def default(cls, gitignore: pathspec.PathSpec | None = None) -> "SkipPolicy":
        return cls(gitignore=gitignore)


@dataclass
class ReadFileState:
    # path -> (mtime_ns, char_count)。Day 4 的 read-before-edit 要靠它判断
    # "模型读过这个文件之后，文件在磁盘上是不是又被改过"。今天先只做记录。
    entries: dict[Path, tuple[int, int]] = field(default_factory=dict)

    def record(self, path: Path, content: str) -> None:
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return
        self.entries[path] = (mtime_ns, len(content))


def resolve_in_cwd(cwd: Path, user_path: str) -> Path:
    # 把模型给的相对路径解析成绝对路径，并强制返回cwd子树
    # 越界直接报错，由调用方包成observation
    candidate = (cwd / user_path).resolve()
    cwd_resolved = cwd.resolve()
    try:
        candidate.relative_to(cwd_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes cwd: {user_path}") from exc
    return candidate


def ensure_text_file(path: Path) -> None:
    # 白名单后缀直接放行，其他文件peek首1 kb，看到NUL就当二进制拒绝
    if path.suffix.lower() in TEXT_SUFFIXES:
        return
    with path.open("rb") as f:
        if f.read(1024).find(b"\x00") != -1:
            raise ValueError(f"binary file: {path.name}")


def ensure_whithin_size(path: Path, max_bytes: int = MAX_READ_BYTE) -> None:
    # 整个文件读取的硬上限
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"file too large:{size} bytes > {max_bytes} bytes;"
            f"read a smaller file or use grep instead"
        )


def should_skip(rel_path: Path, policy: SkipPolicy) -> bool:
    # 判断相对路径 rel_path 是否应该被跳过（不读、不列等）：路径的任意一段目录/文件名若出现在 policy.skip_dirs 里，就返回 True。

    if any(part in policy.skip_dirs for part in rel_path.parts):
        return True
    if policy.gitignore is not None and policy.gitignore.match_file(str(rel_path)):
        return True
    return False


def truncate_output(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[truncated {len(text)-max_chars} chars]"


def load_gitignore(cwd: Path) -> pathspec.PathSpec | None:
    # 只读cwd根的.gitignore
    gitignore = cwd / ".gitignore"
    if not gitignore.exists():
        return None
    lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    return pathspec.PathSpec.from_lines("gitignore", lines)
