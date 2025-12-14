#!/usr/bin/env python3
"""Quick analysis of a single K_{s,t} file."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from analyze_conjectures import (
    parse_structure, analyze_structure, get_structure_pattern,
    check_conjecture_1, check_conjecture_2, get_clique_partition
)

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_one.py extremal_K23.json")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    with open(filepath) as f:
        data = json.load(f)

    s = data["s"]
    t = data["t"]

    print("=" * 80)
    print(f"K_{{{s},{t}}}-FREE EXTREMAL COGRAPHS")
    print("=" * 80)
    print()

    for n_str in sorted(data.get("extremal_by_n", {}).keys(), key=int):
        n = int(n_str)
        n_data = data["extremal_by_n"][n_str]
        ex = n_data["ex"]
        count = n_data["count"]

        print(f"\nn={n:2d}, ex={ex:3d}, count={count:2d}")
        print("-" * 80)

        for struct_data in n_data["structures"]:
            struct_str = struct_data["structure"]

            try:
                node = parse_structure(struct_str)
                pattern = get_structure_pattern(node)
                conj1 = check_conjecture_1(node, n, s, t)
                conj2 = check_conjecture_2(node, n, s, t)

                c1 = "✓" if conj1["holds"] else "✗"
                c2 = "✓" if conj2["holds"] else "✗"

                print(f"  [{c1}][{c2}] {struct_str}")

                # Show details about the structure
                if node.op == "product":
                    left = node.children[0]
                    right = node.children[1]
                    left_cliques = get_clique_partition(left)
                    right_cliques = get_clique_partition(right)

                    print(f"       P({left.n}, {right.n})", end="")
                    if left_cliques:
                        print(f" | Left: cliques {left_cliques}", end="")
                    else:
                        print(f" | Left: {get_structure_pattern(left)}", end="")
                    if right_cliques:
                        print(f" | Right: cliques {right_cliques}")
                    else:
                        print(f" | Right: {get_structure_pattern(right)}")

                    if not conj1["holds"] and conj1["expected_sizes"]:
                        print(f"       Expected: P({conj1['expected_sizes'][0]}, {conj1['expected_sizes'][1]})")

            except Exception as e:
                print(f"  [ERR] {struct_str}: {e}")

if __name__ == "__main__":
    main()
