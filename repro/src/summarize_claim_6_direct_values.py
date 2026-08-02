#!/usr/bin/env python3
"""Export representative direct state values from the frozen Claim 6 audit."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path


FIELDS = [
    "seed",
    "operator",
    "g_rank",
    "state_index",
    "g",
    "u_m",
    "n_m",
    "delta",
    "one_over_z_m",
    "delta_over_one_over_z_m",
    "induced_probability",
    "target_probability",
]


def render_samples(path: Path, top_k: int) -> str:
    audit = json.loads(path.read_text())
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()

    for row in audit["rows"]:
        values = row["state_values"]
        state_indices = values["state_index"]
        if state_indices != list(range(len(state_indices))):
            raise ValueError("state_index must enumerate the frozen arrays in order")
        if len(state_indices) != 1024:
            raise ValueError("each audit row must contain all 1,024 terminal states")

        total_g = sum(values["g"])
        order = sorted(state_indices, key=lambda index: (-values["g"][index], index))
        for rank, index in enumerate(order[:top_k], start=1):
            g = values["g"][index]
            u_m = values["u_m"][index]
            n_m = values["n_m"][index]
            delta = values["delta"][index]
            induced = values["induced_probability"][index]
            reference = row["one_over_z_m"]
            if abs(delta - u_m / n_m) > 1e-12:
                raise ValueError("recorded delta does not equal u_m / n_m")
            if abs(delta - induced / g) > 1e-4:
                raise ValueError("recorded delta does not match induced probability / G")

            writer.writerow(
                {
                    "seed": row["seed"],
                    "operator": row["operator"],
                    "g_rank": rank,
                    "state_index": index,
                    "g": format(g, ".17g"),
                    "u_m": format(u_m, ".17g"),
                    "n_m": format(n_m, ".17g"),
                    "delta": format(delta, ".17g"),
                    "one_over_z_m": format(reference, ".17g"),
                    "delta_over_one_over_z_m": format(delta / reference, ".17g"),
                    "induced_probability": format(induced, ".17g"),
                    "target_probability": format(g / total_g, ".17g"),
                }
            )

    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evidence/claim-6/direct_state_audit.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/claim-6/direct_state_samples.csv"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    rendered = render_samples(args.input, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"wrote {len(rendered.splitlines()) - 1} verified rows to {args.output}")


if __name__ == "__main__":
    main()
