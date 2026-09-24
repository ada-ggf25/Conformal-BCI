"""Markdown lint checks for project documentation files via pymarkdownlnt."""

import pathlib
import shutil
import subprocess

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]

# MD013: line length
_DISABLED_RULES = "MD013"
_ALLOWED_HTML_ELEMENTS = "!--,![CDATA[,!DOCTYPE,div,img,p"

# MkDocs pages additionally need MD033.
_MKDOCS_DISABLED_RULES = f"{_DISABLED_RULES},MD033"

_README_FILES = [
    _REPO / "README.md",
    _REPO / "src" / "README.md",
    _REPO / "tests" / "README.md",
    _REPO / "analyses" / "README.md",
    _REPO / "paper" / "README.md",
    _REPO / ".github" / "WORKFLOWS.md",
]

# Globbed rather than listed: a new MkDocs page is picked up automatically, so the
# nav and the lint scope cannot drift apart.
_MKDOCS_FILES = sorted((_REPO / "docs").rglob("*.md"))

_DOC_FILES = [(p, _DISABLED_RULES) for p in _README_FILES] + [
    (p, _MKDOCS_DISABLED_RULES) for p in _MKDOCS_FILES
]


def test_pymarkdown_available():
    """The pymarkdownlnt binary is installed."""
    assert shutil.which("pymarkdown") is not None, (
        "pymarkdownlnt is required for markdown linting. Install with: uv sync"
    )


@pytest.mark.parametrize(
    ("doc_file", "disabled_rules"),
    _DOC_FILES,
    ids=[str(p.relative_to(_REPO)) for p, _ in _DOC_FILES],
)
def test_markdown_lint(doc_file, disabled_rules):
    """Each tracked documentation file passes markdown lint."""
    if shutil.which("pymarkdown") is None:
        pytest.skip("pymarkdownlnt not installed")

    result = subprocess.run(
        [
            "pymarkdown",
            "--disable-rules",
            disabled_rules,
            "--set",
            f"plugins.md033.allowed_elements={_ALLOWED_HTML_ELEMENTS}",
            "scan",
            str(doc_file),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"pymarkdownlnt found issues in {doc_file.relative_to(_REPO)}:\n\n"
        f"{result.stdout}{result.stderr}"
    )
