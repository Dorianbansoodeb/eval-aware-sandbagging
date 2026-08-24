#!/usr/bin/env python3
"""Grade-lock the committed eval without pytest.

    PYTHONPATH=. python tests/run_audit.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _callables(mod):
    names = []
    for name in sorted(dir(mod)):
        if name.startswith("test_") and callable(getattr(mod, name)):
            names.append(name)
    return names


def main() -> int:
    from tests import test_all_generations, test_score

    failed = []
    passed = 0
    n_rows = len(test_all_generations.GENS)
    print(f"Auditing {n_rows} generations and scorer unit tests\n")
    for mod in (test_score, test_all_generations):
        for name in _callables(mod):
            fn = getattr(mod, name)
            label = f"{mod.__name__}.{name}"
            try:
                fn()
            except Exception:
                failed.append(label)
                print(f"FAIL  {label}")
                traceback.print_exc()
                print()
            else:
                passed += 1
                print(f"ok    {label}")
    print()
    if failed:
        print(f"{len(failed)} failed, {passed} passed, {n_rows} rows audited")
        return 1
    print(f"{passed} passed, 0 failed, {n_rows} rows audited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
