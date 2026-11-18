#!/usr/bin/env python3
"""Turn the percent-format pipeline into week13_pipeline.ipynb."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "week13_pipeline.py").read_text()
chunks = re.split(r"(?m)^# %%(?: \[markdown\])?[ \t]*\n", source)
cells = []

for chunk in chunks:
    lines = chunk.splitlines()
    if not lines or all(not line.strip() for line in lines):
        continue
    is_markdown = all(
        (not line.strip()) or line.startswith("#") for line in lines
    ) and any(line.startswith("# ") or line.startswith("#\n") or line == "#" for line in lines)
    # A code cell can contain comments. Treat a chunk as markdown only when
    # every non-empty line is a comment AND none of them is an assignment-style
    # code comment mixed with Python. The source marks markdown by a leading
    # "# %% [markdown]" split, so those chunks are comments only.
    if is_markdown and not any(line.startswith("# %%") for line in lines):
        md_lines = []
        for line in lines:
            if line.startswith("# "):
                md_lines.append(line[2:])
            elif line.strip() == "#":
                md_lines.append("")
            elif line.strip() == "":
                md_lines.append("")
            else:
                is_markdown = False
                break
        if is_markdown:
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [line + "\n" for line in md_lines],
                }
            )
            continue
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in lines],
        }
    )

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (week13-spark)",
            "language": "python",
            "name": "week13-spark",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}
out = ROOT / "week13_pipeline.ipynb"
out.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {out} with {len(cells)} cells")
