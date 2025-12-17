Project Overview

This repository contains an Equilibrium Problem with Equilibrium Constraints (EPEC) that models the strategic policy interactions of major solar-PV market players. The model captures how firms and regulators react to each other's decisions, forming a nested optimization structure.

Documentation (Sphinx + MyST)
- Edit math in Python docstrings and rebuild docs to update both HTML and LaTeX/PDF outputs.
- Build HTML: `sphinx-build -b html docs docs/_build/html`
- Build LaTeX/PDF: `sphinx-build -b latex docs docs/_build/latex` then run `pdflatex` (or `latexmk`) in `docs/_build/latex`.
- Docs source lives in `docs/`; autodoc pulls APIs from `src/epec`.