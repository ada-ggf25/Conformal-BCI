"""Check American English spelling of the paper LaTeX source: main.tex and every file it
``\\input``s (the NeurIPS checklist).
"""

import pathlib
import re
import shutil
import subprocess

import pytest

repo_dir = pathlib.Path(__file__).resolve().parents[2]
paper_dir = repo_dir / "paper"
main_tex = paper_dir / "main.tex"
wordlist_path = paper_dir / "paper.wordlist"


def tex_sources() -> list[pathlib.Path]:
    """main.tex plus every file it pulls in with \\input or \\include."""
    names = re.findall(
        r"\\(?:input|include)\{([^}]+)\}", main_tex.read_text(encoding="utf-8")
    )
    extra = [paper_dir / (n if n.endswith(".tex") else f"{n}.tex") for n in names]
    return [main_tex, *extra]


_STRIP_ENVS = ["tikzpicture"]


def preprocess(source: str) -> str:
    """Remove the LaTeX preamble and code-heavy environments."""
    source = re.sub(r"^.*?\\begin\{document\}", "", source, flags=re.DOTALL)
    for env in _STRIP_ENVS:
        source = re.sub(
            rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}",
            "",
            source,
            flags=re.DOTALL,
        )
    return source


def run_aspell(path: pathlib.Path = main_tex):
    """Return the set of words aspell does not recognize in one LaTeX source."""
    if shutil.which("aspell") is None:
        pytest.fail(
            "aspell is required to check American English spelling. "
            "Install with: sudo apt-get install aspell aspell-en"
        )

    source = preprocess(path.read_text(encoding="utf-8"))

    result = subprocess.run(
        ["aspell", "--mode=tex", "--lang=en_US", "list"],
        input=source,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert result.returncode == 0, f"aspell failed.\n\nOutput:\n{result.stdout}"
    return {w.strip() for w in result.stdout.splitlines() if w.strip()}


def load_wordlist():
    """Return the accepted words from the paper wordlist."""
    if not wordlist_path.exists():
        return set()
    lines = wordlist_path.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}


def test_aspell_available():
    """The aspell binary is installed."""
    assert shutil.which("aspell") is not None, (
        "aspell is not installed. Install with: sudo apt-get install aspell aspell-en"
    )


def test_wordlist_exists():
    """The paper wordlist file is present."""
    assert wordlist_path.is_file(), (
        f"Custom wordlist not found at {wordlist_path}. "
        "Create it to allow domain-specific terms."
    )


def test_every_input_file_exists():
    """Each file main.tex inputs is present, so none is silently skipped by the checks."""
    for path in tex_sources():
        assert path.is_file(), f"{path.name} is input by main.tex but missing"


@pytest.mark.parametrize("path", tex_sources(), ids=lambda p: p.name)
def test_american_english_spelling(path):
    """Each LaTeX source spell-checks clean once the wordlist is applied."""
    unknown = run_aspell(path)
    allowed = load_wordlist()
    flagged = unknown - allowed

    assert not flagged, (
        "Possible spelling errors (or missing wordlist entries) found "
        f"in {path.name}.\n\n"
        "If these are valid technical terms, add them to "
        "paper/paper.wordlist.\n\n"
        f"Flagged words ({len(flagged)}):\n"
        + "\n".join(f"  {w}" for w in sorted(flagged))
    )
