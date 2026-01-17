$activate = Join-Path $env:USERPROFILE ".venvs\\gamspy-epec\\Scripts\\Activate.ps1"
& $activate

# GAMSPy CLI output contains unicode; avoid cp1252 issues on Windows consoles.
$env:PYTHONUTF8 = "1"

python -m pip install -r requirements.txt
python -m gamspy list solvers --installables
python -m gamspy install solver KNITRO
python -m gamspy list solvers --all
