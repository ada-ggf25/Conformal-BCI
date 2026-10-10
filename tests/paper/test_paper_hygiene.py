"""Bibliography and figure hygiene for the paper, checks the strict LaTeX build cannot make.

The build already fails on an undefined citation or a missing image. It does not notice a
bibliography entry nobody cites (BibTeX drops it silently), a key defined twice, an image in
``figures/`` that the paper no longer uses, or an image that exists on this machine but was never
committed (the build passes here and breaks on a fresh clone). It also keeps internal research-vault
labels, which a reader cannot resolve, out of the manuscript.
"""

import pathlib
import re
import subprocess

repo_dir = pathlib.Path(__file__).resolve().parents[2]
paper_dir = repo_dir / "paper"
main_tex = paper_dir / "main.tex"
bib_path = paper_dir / "references.bib"
figures_dir = paper_dir / "figures"


def _strip_comments(source: str) -> str:
    """Drop LaTeX comments (an unescaped % to the end of the line)."""
    return re.sub(r"(?<!\\)%.*", "", source)


def tex_source() -> str:
    """main.tex plus every file it inputs, with comments removed."""
    main = main_tex.read_text(encoding="utf-8")
    parts = [main]
    for name in re.findall(r"\\(?:input|include)\{([^}]+)\}", _strip_comments(main)):
        path = paper_dir / (name if name.endswith(".tex") else f"{name}.tex")
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8"))
    return _strip_comments("\n".join(parts))


def cited_keys() -> set[str]:
    """Every key passed to a \\cite-family command (\\cite, \\citep, \\citet, \\citeauthor ...)."""
    groups = re.findall(
        r"\\[a-zA-Z]*cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex_source()
    )
    return {k.strip() for g in groups for k in g.split(",") if k.strip()}


def bib_keys() -> list[str]:
    """Every entry key in references.bib, in file order (duplicates kept)."""
    text = bib_path.read_text(encoding="utf-8")
    return re.findall(r"^\s*@(?!comment|string|preamble)[a-zA-Z]+\s*\{\s*([^,\s]+)\s*,", text,
                      flags=re.MULTILINE | re.IGNORECASE)  # fmt: skip


def included_graphics() -> set[str]:
    """File names passed to \\includegraphics (relative to the graphics path)."""
    return set(
        re.findall(r"\\includegraphics\*?(?:\[[^\]]*\])?\{([^}]+)\}", tex_source())
    )


def _resolve(name: str) -> pathlib.Path | None:
    """The figures/ file an \\includegraphics name refers to (extension optional)."""
    direct = figures_dir / name
    if direct.is_file():
        return direct
    matches = sorted(figures_dir.glob(f"{name}.*"))
    return matches[0] if matches else None


def test_bibliography_keys_are_unique():
    """No key is defined twice (BibTeX compares keys case-insensitively)."""
    lowered = [k.lower() for k in bib_keys()]
    duplicates = sorted({k for k in lowered if lowered.count(k) > 1})
    assert not duplicates, f"Duplicate references.bib keys: {duplicates}"


def test_every_bibliography_entry_is_cited():
    """Uncited entries are dead weight BibTeX drops silently; cite them or delete them."""
    uncited = sorted(set(bib_keys()) - cited_keys())
    assert not uncited, f"references.bib entries never cited in the paper: {uncited}"


def test_every_figure_file_is_used():
    """Every image in figures/ is included by the paper; a stale one is a leftover."""
    used = {p.name for n in included_graphics() if (p := _resolve(n)) is not None}
    unused = sorted(
        p.name for p in figures_dir.iterdir() if p.is_file() and p.name not in used
    )
    assert not unused, f"Images in paper/figures/ the paper never includes: {unused}"


def test_every_included_figure_is_committed():
    """An image that exists only locally builds here and breaks every fresh clone and CI."""
    tracked = set(
        subprocess.run(
            ["git", "ls-files", "paper/figures"], cwd=repo_dir, check=True,
            capture_output=True, text=True,
        ).stdout.split()
    )  # fmt: skip
    problems = []
    for name in sorted(included_graphics()):
        path = _resolve(name)
        if path is None:
            problems.append(f"{name}: no such file in paper/figures/")
        elif str(path.relative_to(repo_dir)) not in tracked:
            problems.append(f"{name}: not tracked by git")
    assert not problems, (
        "Included figures that a fresh clone would not have:\n" + "\n".join(problems)
    )


# Decision records (ADR-012), experiment notes (EXP-004) and scenario names (scen.1).
VAULT_LABEL = re.compile(r"\b(?:ADR|EXP)-\d+\b|\bscen\.\s?\d")


def test_no_vault_labels():
    """The paper never cites internal vault labels; describe the thing instead."""
    found = sorted(set(VAULT_LABEL.findall(tex_source())))
    assert not found, f"Internal vault labels in the paper source: {found}"
