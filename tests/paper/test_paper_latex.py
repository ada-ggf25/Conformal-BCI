"""Test the paper LaTeX source build."""

import os
import pathlib
import subprocess

import pytest

# This test is reached through `make paper-test`.
_BUILD_ENV = {**os.environ, "MAKEFLAGS": "", "MFLAGS": ""}

repo_dir = pathlib.Path(__file__).resolve().parents[2]
paper_dir = repo_dir / "paper"


def test_source_dir():
    """The paper source directory exists."""
    assert paper_dir.is_dir()


def test_main_tex():
    """main.tex exists in the paper source directory."""
    assert (paper_dir / "main.tex").is_file()


def test_references_bib():
    """references.bib exists in the paper source directory."""
    assert (paper_dir / "references.bib").is_file()


def test_compiles_with_warnings_as_errors():
    """make strict compiles the report with LaTeX warnings promoted to errors."""
    command = ["make", "strict"]

    try:
        result = subprocess.run(
            command,
            cwd=paper_dir,
            env=_BUILD_ENV,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        pytest.fail("make is required to run the strict paper build")

    build_output = result.stdout or ""
    assert result.returncode == 0, (
        "Strict LaTeX build failed. The Makefile strict target treats "
        "warnings as errors.\n\n"
        f"Command: {' '.join(command)}\n\n"
        "Last 12000 characters of build output:\n"
        f"{build_output[-12000:]}"
    )
