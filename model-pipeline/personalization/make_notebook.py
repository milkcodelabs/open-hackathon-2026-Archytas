"""A notebook script (cells split on '# %%') -> the .ipynb next to it, for Colab. Stdlib only.

    python make_notebook.py                     # finetune.py -> finetune.ipynb
    python make_notebook.py few_sentences.py    # few_sentences.py -> few_sentences.ipynb
"""

import json
import sys
from pathlib import Path

here = Path(__file__).parent
script = here / (sys.argv[1] if len(sys.argv) > 1 else "finetune.py")
src = script.read_text(encoding="utf-8")
parts = src.split("\n# %%\n")
cells = []
header = parts[0].strip()
md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in header.splitlines())
cells.append({"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)})
for p in parts[1:]:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": p.strip("\n").splitlines(keepends=True)})
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
script.with_suffix(".ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(script.with_suffix(".ipynb").name, len(cells), "cells")
