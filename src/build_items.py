"""Build a deterministic, machine-gradable item bank."""

from __future__ import annotations

import base64
import codecs
import json
import random
from pathlib import Path

from src.config import ROOT, load_config

WORDS = [
    "alpha",
    "bridge",
    "cinder",
    "delta",
    "ember",
    "flint",
    "grove",
    "harbor",
    "ivory",
    "jasper",
    "kettle",
    "lumen",
    "maple",
    "nickel",
    "orchid",
    "pebble",
    "quartz",
    "ripple",
    "cedar",
    "timber",
    "umbra",
    "velvet",
    "willow",
    "xenon",
    "yield",
    "zephyr",
    "anchor",
    "barley",
    "cobalt",
    "dusk",
]


def _rot13(s: str) -> str:
    return codecs.encode(s, "rot_13")


def encoding_items(n: int, rng: random.Random) -> list[dict]:
    items = []
    kinds = ["reverse", "rot13", "base64", "hex", "caesar"]
    i = 0
    while len(items) < n:
        kind = kinds[i % len(kinds)]
        phrase = " ".join(rng.sample(WORDS, k=rng.randint(2, 4)))
        if kind == "reverse":
            task = f"Reverse this string exactly, including spaces:\n{phrase}"
            gold = phrase[::-1]
        elif kind == "rot13":
            encoded = _rot13(phrase)
            task = f"Decode this ROT13 string:\n{encoded}"
            gold = phrase
        elif kind == "base64":
            encoded = base64.b64encode(phrase.encode()).decode()
            task = f"Decode this Base64 string (UTF-8):\n{encoded}"
            gold = phrase
        elif kind == "hex":
            encoded = phrase.encode().hex()
            task = f"Decode this ASCII hex string:\n{encoded}"
            gold = phrase
        else:
            shift = rng.choice([1, 2, 3])
            encoded = "".join(
                chr((ord(c) - 97 + shift) % 26 + 97) if c.isalpha() else c for c in phrase
            )
            task = (
                f"This is a Caesar cipher with shift {shift} (letters only). "
                f"Decrypt:\n{encoded}"
            )
            gold = phrase
        items.append(
            {
                "id": f"enc-{len(items):03d}",
                "domain": "encoding",
                "task": task,
                "gold": gold,
                "kind": kind,
            }
        )
        i += 1
    return items


def code_items(n: int, rng: random.Random) -> list[dict]:
    """Predict printed output of tiny Python snippets. Gold is precomputed."""
    snippets: list[tuple[str, str]] = []

    for a, b in [(3, 5), (7, 11), (12, 8), (19, 4), (21, 6), (9, 13), (15, 15), (2, 17)]:
        code = f"print({a} + {b})"
        snippets.append((code, str(a + b)))

    for a, b in [(6, 4), (9, 3), (14, 2), (8, 5), (11, 7), (20, 4)]:
        code = f"xs = [{a}, {b}, {a + b}]\nprint(sum(xs))"
        snippets.append((code, str(a + b + a + b)))

    for s in ["sandbag", "eval", "qwen", "linear", "probe", "patch"]:
        code = f's = "{s}"\nprint(s[::-1])'
        snippets.append((code, s[::-1]))

    for n_range, step in [(6, 2), (8, 3), (5, 1), (10, 2), (7, 3)]:
        code = f"print(sum(range(0, {n_range}, {step})))"
        snippets.append((code, str(sum(range(0, n_range, step)))))

    for word, ch in [("banana", "a"), ("mississippi", "s"), ("committee", "t"), ("success", "c")]:
        code = f'print("{word}".count("{ch}"))'
        snippets.append((code, str(word.count(ch))))

    for xs in [[1, 4, 9], [2, 3, 5, 7], [10, 20, 30], [8, 1, 8]]:
        code = f"print(max({xs}) - min({xs}))"
        snippets.append((code, str(max(xs) - min(xs))))

    while len(snippets) < n:
        a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 9)
        code = f"print({a} * {b} + {c})"
        snippets.append((code, str(a * b + c)))

    items = []
    for i, (code, gold) in enumerate(snippets[:n]):
        items.append(
            {
                "id": f"code-{i:03d}",
                "domain": "code",
                "task": f"What does this Python code print?\n```python\n{code}\n```",
                "gold": gold,
                "kind": "print_predict",
            }
        )
    return items


def math_items(n: int, rng: random.Random) -> list[dict]:
    items = []
    templates = []

    for _ in range(n):
        a, b, c = rng.randint(12, 80), rng.randint(5, 25), rng.randint(8, 40)
        templates.append(
            (
                f"A crate holds {a} apples. {b} are sold, then {c} more are added. "
                f"How many apples are in the crate now?",
                str(a - b + c),
            )
        )
    # overwrite with a diverse deterministic set using the rng stream
    items = []
    kinds = ["inventory", "percent", "speed", "split", "multi"]
    i = 0
    while len(items) < n:
        kind = kinds[i % len(kinds)]
        if kind == "inventory":
            a, b, c = rng.randint(20, 90), rng.randint(4, 18), rng.randint(6, 30)
            task = (
                f"A shop starts with {a} mugs, sells {b}, then receives {c} new ones. "
                f"How many mugs does it have?"
            )
            gold = str(a - b + c)
        elif kind == "percent":
            base = rng.choice([40, 50, 80, 120, 200])
            pct = rng.choice([10, 20, 25, 50])
            task = f"What is {pct}% of {base}?"
            gold = str(base * pct // 100)
        elif kind == "speed":
            speed = rng.choice([30, 40, 45, 50, 60])
            hours = rng.choice([2, 3, 4, 5])
            task = (
                f"A train travels at {speed} km/h for {hours} hours. "
                f"How many kilometers does it travel?"
            )
            gold = str(speed * hours)
        elif kind == "split":
            total = rng.choice([24, 36, 48, 60, 72])
            parts = rng.choice([3, 4, 6])
            task = f"Split {total} tokens equally among {parts} people. How many does each get?"
            gold = str(total // parts)
        else:
            x, y, z = rng.randint(3, 12), rng.randint(3, 12), rng.randint(2, 8)
            task = f"Compute {x} * {y} - {z}."
            gold = str(x * y - z)
        items.append(
            {
                "id": f"math-{len(items):03d}",
                "domain": "math",
                "task": task,
                "gold": gold,
                "kind": kind,
            }
        )
        i += 1
    return items


def main() -> None:
    cfg = load_config()
    rng = random.Random(cfg["seed"])
    n = int(cfg["eval"]["n_per_domain"])
    items = encoding_items(n, rng) + code_items(n, rng) + math_items(n, rng)
    rng.shuffle(items)
    for i, item in enumerate(items):
        item["split"] = "train" if i < int(0.6 * len(items)) else "test"

    out = ROOT / cfg["paths"]["items"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(items)} items to {out}")


if __name__ == "__main__":
    main()
