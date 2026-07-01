from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._frontmatter import _split_frontmatter, _unquote


@dataclass
class OutputStyle:
    name: str
    description: str
    body: str
    path: Path


def list_output_styles(cwd: Path) -> list[OutputStyle]:
    styles_dir = cwd / ".agent" / "output-styles"
    if not styles_dir.exists():
        return []

    styles: list[OutputStyle] = []
    for path in sorted(styles_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        fields, body = _split_frontmatter(text)
        name = _unquote(fields.get("name", path.stem)).strip()
        description = _unquote(fields.get("description", "")).strip()
        styles.append(
            OutputStyle(name=name, description=description, body=body, path=path)
        )
    return styles


def load_output_style(cwd: Path, name: str) -> OutputStyle | None:
    for style in list_output_styles(cwd):
        if style.name == name:
            return style
    return None


def render_output_style(cwd: Path, name: str | None) -> str:
    if not name:
        return ""
    style = load_output_style(cwd, name)
    if style is None:
        return ""
    return f'<output-style name="{style.name}">\n{style.body}\n</output-style>'
