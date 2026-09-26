"""finetune.py (cells split on '# %%') -> finetune.ipynb for Colab. Stdlib only."""

import json
from pathlib import Path

here = Path(__file__).parent
src = (here / "finetune.py").read_text(encoding="utf-8")
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
(here / "finetune.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(len(cells), "cells")
