from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    lock_path = project_root / "requirements-lock.txt"

    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    lock_path.write_text(result.stdout, encoding="utf-8")
    print(f"Wrote {lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

