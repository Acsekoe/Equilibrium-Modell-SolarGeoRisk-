from __future__ import annotations

import importlib
import sys
from importlib import metadata


PACKAGES: list[tuple[str, str]] = [
    ("gamspy", "gamspy"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("openpyxl", "openpyxl"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
    ("pyarrow", "pyarrow"),
    ("rich", "rich"),
]


def main() -> int:
    failures: list[str] = []

    for import_name, dist_name in PACKAGES:
        try:
            importlib.import_module(import_name)
            version = metadata.version(dist_name)
            print(f"{dist_name}=={version}")
        except Exception as exc:
            failures.append(f"{dist_name}: {exc!r}")

    if failures:
        print("\nImport/version check failed:", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

