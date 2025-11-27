#!/usr/bin/env python3
"""Run CodeSutra example scripts in CI-friendly way.

This script will run `examples/numpy_example.codesutra` and
`examples/pandas_example.codesutra` only if their Python deps are available.
It exits with non-zero when an executed example fails. If examples are
skipped due to missing deps, the script exits 0 (graceful skip).
"""
import subprocess
import sys
import importlib.util


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_example(path: str) -> int:
    print(f"Running example: {path}")
    res = subprocess.run([sys.executable, "src/main.py", path])
    return res.returncode


def main():
    examples = [
        ("examples/numpy_example.codesutra", "numpy"),
        ("examples/pandas_example.codesutra", "pandas"),
    ]

    ran_any = False
    failures = 0

    for path, dep in examples:
        if has_module(dep):
            ran_any = True
            code = run_example(path)
            if code != 0:
                print(f"Example failed: {path} (exit {code})", file=sys.stderr)
                failures += 1
        else:
            print(f"Skipping {path}: missing dependency '{dep}'")

    if failures > 0:
        sys.exit(2)

    if not ran_any:
        print("No examples run: required Python packages are not installed. Skipping.")
        sys.exit(0)

    print("Examples completed successfully")
    sys.exit(0)


if __name__ == '__main__':
    main()
