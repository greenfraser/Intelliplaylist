from __future__ import annotations

from pathlib import Path
from typing import List
import random
import clingo

# Wrapper for running the clingo ASP solver.
# This file loads the generated ASP facts and playlist rules, runs clingo,
# and returns the selected track IDs from the answer set.

# Runs clingo on the generated facts and rules file, then extracts selected track IDs.
def run_clingo(
    facts_file: Path,
    rules_file: Path,
    playlist_size: int = 20,
    timeout_seconds: float = 5.0,
) -> List[str]:
    ctl = clingo.Control([
        "-c",
        f"n={playlist_size}",

        "--opt-mode=ignore",
        "--models=1",

        "--rand-freq=0.25",
        "--seed",
        str(random.randint(1, 1_000_000)),
    ])

    ctl.load(str(facts_file))
    ctl.load(str(rules_file))
    ctl.ground([("base", [])])

    last_model = None

    def on_model(model: clingo.Model) -> None:
        nonlocal last_model
        last_model = model.symbols(shown=True)

    handle = ctl.solve(on_model=on_model, async_=True)

    finished = handle.wait(timeout_seconds)
    if not finished:
        handle.cancel()

    result = handle.get()

    if not result.satisfiable or last_model is None:
        return []

    ids: List[str] = []
    for sym in last_model:
        if sym.name == "in_playlist" and len(sym.arguments) == 1:
            arg = sym.arguments[0]
            if arg.type == clingo.SymbolType.String:
                ids.append(arg.string)

    return ids
