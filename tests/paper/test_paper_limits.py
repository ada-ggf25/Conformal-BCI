"""Check paper limits from the requirements."""

import pathlib
import re
import shutil
import subprocess
import tempfile

import pytest

repo_dir = pathlib.Path(__file__).resolve().parents[2]
paper_dir = repo_dir / "paper"
main_tex = paper_dir / "main.tex"

PAPER_WORD_LIMIT = 5000
FIGURE_TABLE_LIMIT = 10

# Count \cite, \ref, \eqref and \autoref as one word each.
TEXCOUNT_RULES = "".join(
    f"%TC:macro \\{macro} [0]\n%TC:macrocount \\{macro} 1\n"
    for macro in ("cite", "ref", "eqref", "autoref")
)


def main_body_source():
    """Return main.tex trimmed to the counted body.

    Excludes the title page, AI acknowledgments, acknowledgments (including the NeurIPS
    ``ack`` environment), appendix, glossary, bibliography and the paper checklist.
    """
    source = main_tex.read_text(encoding="utf-8")
    stop_match = re.search(r"\\appendix|\\section\*\{Glossary\}|\\bibliography", source)
    source = source[: stop_match.start()] if stop_match else source
    source = re.sub(
        r"\\begin\{titlepage\}.*?\\end\{titlepage\}", "", source, flags=re.DOTALL
    )
    source = re.sub(r"\\begin\{ack\}.*?\\end\{ack\}", "", source, flags=re.DOTALL)
    source = re.sub(
        r"\\section\*?\{(?:AI )?Acknowledgments?\}.*?(?=\\section|\Z)",
        "",
        source,
        flags=re.DOTALL,
    )
    return source


def count_words(tmp_path, source):
    """Return texcount's counted text-word total for a LaTeX source string."""
    if shutil.which("texcount") is None:
        pytest.fail("texcount is required to check the paper word limits")

    tex_path = tmp_path / "body.tex"
    tex_path.write_text(TEXCOUNT_RULES + source, encoding="utf-8")
    result = subprocess.run(
        ["texcount", "-merge", "-brief", str(tex_path)],
        cwd=tex_path.parent,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 0, (
        f"texcount failed while checking the paper limits.\n\n{result.stdout}"
    )
    match = re.search(r"^\s*(\d+)\+", result.stdout, re.MULTILINE)
    assert match, f"Could not parse texcount output:\n{result.stdout}"
    return int(match.group(1))


def figure_table_counts():
    """Return (figure count, table count) for the main body."""
    body_source = main_body_source()
    figures = re.findall(r"\\begin\{figure\*?\}", body_source)
    tables = re.findall(r"\\begin\{table\*?\}", body_source)
    return len(figures), len(tables)


def print_wordcount_summary():
    """Print the current word and float counts without asserting, for make wordcount."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        word_count = count_words(pathlib.Path(tmp_dir), main_body_source())

    figure_count, table_count = figure_table_counts()
    print("=== Paper limits count ===")
    print(f"Report words: {word_count} / {PAPER_WORD_LIMIT}")
    print(
        f"Figures + tables: {figure_count + table_count} / {FIGURE_TABLE_LIMIT} "
        f"({figure_count} figures + {table_count} tables)"
    )


def test_report_word_limit(tmp_path):
    """The counted body word total is within the limit."""
    word_count = count_words(tmp_path, main_body_source())

    assert word_count <= PAPER_WORD_LIMIT, (
        f"Paper has {word_count} counted words; "
        f"limit is {PAPER_WORD_LIMIT}.\n\n"
        "The check excludes: title page, AI acknowledgments, acknowledgments, "
        "section headings, equations, captions, footnotes, references, "
        "appendices, and the glossary/list-of-abbreviations section."
    )


def test_figure_and_table_limit():
    """Main-body figures plus tables are within the limit."""
    figure_count, table_count = figure_table_counts()
    combined = figure_count + table_count

    assert combined <= FIGURE_TABLE_LIMIT, (
        f"Paper has {combined} main-body figures/tables "
        f"({figure_count} figures + {table_count} tables); "
        f"limit is {FIGURE_TABLE_LIMIT}."
    )


if __name__ == "__main__":
    print_wordcount_summary()
