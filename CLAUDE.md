# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Exam project for a numerical methods course: 7 original Jupyter notebooks, each built from a technique shown in class but reimplemented with the student's own functions/examples/data, plus a `src/` package of the reusable code factored out of them.

## Layout

- `codici/` — the professor's original notebooks and `.m` files (`Bezier.ipynb`, `SVDforimages.ipynb`, `Least_squares_with_torch.ipynb`, `newton_vett.ipynb`, `newton_min.ipynb`, `DerivateParziali.ipynb`, etc.). Reference material only — never edit these, and never move/copy them into `notebooks/`.
- `notebooks/` — the 7 notebooks being delivered: `bezier_spoiler.ipynb`, `svd_recommender.ipynb`, `least_squares.ipynb`, `newton_vett.ipynb`, `newton_min.ipynb`, `partial_derivatives.ipynb`, `power_method_pagerank.ipynb`. This folder must contain only finished work of the student's own, nothing from `codici/`.
- `src/` — reusable functions factored out of the notebooks, one module per topic:
  - `bezier.py` — `bezier_cubic` + derivatives, curvature, potential-flow field (used only by `bezier_spoiler.ipynb`).
  - `svd_tools.py` — rank-`k` SVD truncation and reconstruction-error helpers, shared between `svd_recommender.ipynb` and the PCR step of `least_squares.ipynb`.
  - `least_squares.py` — polynomial fit via QR, Givens-rotation update, condition-number computation; internally uses `svd_tools` for the PCR part instead of duplicating it.
  - `symbolic_utils.py` — sympy helper(s) that turn a symbolic expression into a lambdified numeric function; shared between `newton_vett.ipynb` (Jacobian), `newton_min.ipynb` (gradient + Hessian) and `partial_derivatives.ipynb` (gradient + Hessian of the Step 4 comparison function).
  - `newton.py` — `newton` (scalar), `newton_vett` (systems), `newton_damped_vett` (damped minimization); built on top of `symbolic_utils`.
  - `calculus.py` — `classify_critical_points` (stationary points + Hessian-eigenvalue classification into minimum/maximum/saddle), used only by `partial_derivatives.ipynb`.
  - `gradient_descent.py` — `gradient_descent` (vanilla gradient descent with a fixed learning rate), used only by `partial_derivatives.ipynb`.
  - `power_method.py` — `power_method` (raw power iteration) and `power_method_norm` (normalized power iteration), used only by `power_method_pagerank.ipynb`.
- `plans/` — process documentation, one plan per notebook (`plan_bezier_spoiler.md`, `plan_svd_recommender.md`, `plan_least_squares.md`, `plan_newton_vett.md`, `plan_newton_min.md`, `plan_partial_derivatives.md`, `plan_power_method_pagerank.md`) plus `plan_project_structure.md`, which governs this folder layout and the migration of code from notebooks into `src/`. Treat `plan_project_structure.md` as the source of truth for where things belong and what step of the migration is currently done.

## Import convention

Every notebook in `notebooks/` imports shared code from `src/` the same way, in its first cell:

```python
import sys, pathlib
sys.path.append(str(pathlib.Path.cwd().parent))
from src.bezier import bezier_cubic
```

## Working conventions specific to this repo

- Each of the 6 notebook plans explicitly forbids reusing the professor's exact functions, systems, or datasets — every example (test functions, matrices, synthetic data) must be different from the corresponding one in `codici/`, even though the underlying technique is the same. When implementing a step from one of the plans, check its "Nota trasversale" for the specific examples to avoid.
- Data is generated synthetically inside the notebooks (fixed random seed) rather than loaded from external CSVs/images, specifically so notebooks keep working after being moved into `notebooks/`.
- When moving a function from a notebook into `src/`, the notebook must be switched to import it (per the convention above) and then re-run top to bottom ("Restart & Run All"); the resulting outputs (numbers, plots) must match exactly what the notebook produced before the migration — the migration is only supposed to change where the code lives, never its behavior.
