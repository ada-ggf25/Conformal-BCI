"""American English spelling checks for project documentation files via aspell.

Covers every tracked README.md, .github/WORKFLOWS.md, and all MkDocs source
pages under docs/.
"""

import pathlib
import re
import shutil
import subprocess

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_WORDLIST = pathlib.Path(__file__).resolve().parent / "docs.wordlist"

_README_FILES = [
    _REPO / "README.md",
    _REPO / "src" / "README.md",
    _REPO / "tests" / "README.md",
    _REPO / "analyses" / "README.md",
    _REPO / "paper" / "README.md",
    _REPO / ".github" / "WORKFLOWS.md",
]

_MKDOCS_FILES = sorted((_REPO / "docs").rglob("*.md"))

_ALL_FILES = _README_FILES + _MKDOCS_FILES


def _preprocess(content: str) -> str:
    """Strip markup syntax, keeping prose for spell-checking."""
    # Fenced code blocks
    content = re.sub(r"```.*?```", " ", content, flags=re.DOTALL)
    # Inline code
    content = re.sub(r"`[^`]+`", " ", content)
    # MathJax display blocks \[...\]
    content = re.sub(r"\\\[.*?\\\]", " ", content, flags=re.DOTALL)
    # MathJax inline \(...\)
    content = re.sub(r"\\\(.*?\\\)", " ", content, flags=re.DOTALL)
    # mkdocstrings directives ::: and their indented option blocks
    content = re.sub(r"^:::[^\n]*(\n[ \t]+[^\n]*)*", " ", content, flags=re.MULTILINE)
    # Admonition headers !!! type "optional title"
    content = re.sub(r"^!!!.*$", " ", content, flags=re.MULTILINE)
    # Material for MkDocs / Octicons icon shortcodes :icon-name-24:
    content = re.sub(r":[a-z][a-z0-9-]*:", " ", content)
    # HTML tags
    content = re.sub(r"<[^>]+>", " ", content)
    # Markdown table separator rows (|---|:--:|) — alignment markup, no prose
    content = re.sub(r"^\s*\|[\s:|-]+\|\s*$", " ", content, flags=re.MULTILINE)
    # Markdown table cells.
    content = re.sub(
        r"^\s*\|(.*)\|\s*$",
        lambda row: " " + row.group(1).replace("|", " ") + " ",
        content,
        flags=re.MULTILINE,
    )
    # URLs
    content = re.sub(r"https?://\S+", " ", content)
    # ATX headings
    content = re.sub(r"^#+\s+", "", content, flags=re.MULTILINE)
    # Bold / italic markers (keep inner text)
    content = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", content)
    # Horizontal rules
    content = re.sub(r"^-{3,}$", "", content, flags=re.MULTILINE)
    # Markdown links (keep link text)
    content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
    # Unordered list markers
    content = re.sub(r"^\s*[-*+]\s+", " ", content, flags=re.MULTILINE)
    # Ordered list markers
    content = re.sub(r"^\s*\d+\.\s+", " ", content, flags=re.MULTILINE)
    # Ordinal suffixes — aspell drops the digits of "15th" and flags the "th"
    content = re.sub(r"\b\d+(?:st|nd|rd|th)\b", " ", content)
    return content


def _run_aspell(prose: str) -> set[str]:
    """Return the set of words aspell does not recognize in the given prose."""
    result = subprocess.run(
        ["aspell", "--mode=none", "--lang=en_US", "list"],
        input=prose,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 0, f"aspell failed.\n\nOutput:\n{result.stdout}"
    return {w.strip() for w in result.stdout.splitlines() if w.strip()}


def _load_wordlist() -> set[str]:
    """Return the accepted words from the shared docs wordlist."""
    if not _WORDLIST.exists():
        return set()
    lines = _WORDLIST.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}


def test_aspell_available():
    """The aspell binary is installed."""
    assert shutil.which("aspell") is not None, (
        "aspell is required to check American English spelling. "
        "Install with: sudo apt-get install aspell aspell-en"
    )


@pytest.mark.parametrize(
    "doc_file", _ALL_FILES, ids=lambda p: str(p.relative_to(_REPO))
)
def test_american_english_spelling(doc_file):
    """Each documentation file spell-checks clean against the shared wordlist."""
    if shutil.which("aspell") is None:
        pytest.skip("aspell not installed")

    prose = _preprocess(doc_file.read_text(encoding="utf-8"))
    flagged = _run_aspell(prose) - _load_wordlist()

    assert not flagged, (
        f"Possible spelling errors in {doc_file.relative_to(_REPO)}.\n\n"
        "If these are valid technical terms or proper nouns, add them to "
        "tests/docs/docs.wordlist.\n\n"
        f"Flagged words ({len(flagged)}):\n"
        + "\n".join(f"  {w}" for w in sorted(flagged))
    )
