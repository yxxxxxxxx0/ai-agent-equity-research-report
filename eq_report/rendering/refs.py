"""How reference markers are printed: sorted, with runs of three or more as a range.

``[4][5][6][7]`` prints as ``[4]-[7]``; a pair stays ``[4][5]``; unrelated numbers
are written out (``[1][4][9]``). Sources that are the same document already share
one number (see synthesis/citations.py), so there is nothing to collapse by source here.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable


def format_refs(refs: Iterable[int], fmt: Callable[[int], str] = lambda n: f"[{n}]") -> str:
    numbers = sorted(set(refs))
    parts: list[str] = []
    i = 0
    while i < len(numbers):
        j = i
        while j + 1 < len(numbers) and numbers[j + 1] == numbers[j] + 1:
            j += 1
        run = numbers[i:j + 1]
        parts.append(fmt(run[0]) if len(run) == 1 else
                     "".join(fmt(n) for n in run) if len(run) == 2 else
                     f"{fmt(run[0])}-{fmt(run[-1])}")
        i = j + 1
    return "".join(parts)


_TOKEN = re.compile(r"\[(\d+)\](?:-\[(\d+)\])?")


def refs_in(text: str) -> set[int]:
    """Every reference number printed in ``text``, expanding ``[a]-[b]`` ranges."""
    found: set[int] = set()
    for first, last in _TOKEN.findall(text):
        found.update(range(int(first), int(last or first) + 1))
    return found
