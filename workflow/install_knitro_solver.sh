#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/.venvs/gamspy-epec/bin/activate"

export PYTHONUTF8=1

python -m pip install -r requirements.txt
python -m gamspy list solvers --installables
python -m gamspy install solver KNITRO
python -m gamspy list solvers --all
