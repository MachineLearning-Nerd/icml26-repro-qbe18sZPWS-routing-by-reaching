#!/usr/bin/env python
"""Training entry point for single-objective GFlowNets (the ingredients for Mixing).

Usage:
    python train_gfn.py <task> [args...]

Tasks:
    qm9        Single-objective QM9 GFN (gap | qed | sa)
    seh_frag   Single-objective SEH fragment GFN (seh | qed | sa)
"""
import importlib
import sys

TASKS = {
    "qm9":      "gfn_composition.tasks.qm9",
    "seh_frag": "gfn_composition.tasks.seh_frag",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    task = sys.argv[1]
    if task not in TASKS:
        print(f"Unknown task: {task!r}\n", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    sys.argv = [sys.argv[0], *sys.argv[2:]]
    importlib.import_module(TASKS[task]).main()


if __name__ == "__main__":
    main()
