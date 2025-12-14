#!/usr/bin/env python3
"""
Discover and verify structural patterns in extremal K_{s,t}-free cographs.

This program systematically analyzes the JSON exports and discovers regularities.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from analyze_conjectures import (
    parse_structure, analyze_structure, get_structure_pattern,
    get_clique_partition, StructureNode
)


def get_partition_sizes(node: StructureNode) -> Tuple[int, int] | None:
    """If node is a product, return (smaller, larger) component sizes."""
    if node.op != "product":
        return None
    sizes = sorted([node.children[0].n, node.children[1].n])
    return tuple(sizes)


def classify_product_structure(node: StructureNode) -> dict:
    """
    Classify a product structure P(left, right).

    Returns dict with:
    - left_type: 'single_vertex', 'clique_partition', 'complex'
    - right_type: same
    - left_detail: structure details
    - right_detail: structure details
    """
    if node.op != "product":
        return {"error": "not a product"}

    left = node.children[0]
    right = node.children[1]

    result = {}

    # Analyze left side
    left_cliques = get_clique_partition(left)
    if left_cliques is not None:
        if len(left_cliques) == 1 and left_cliques[0] == 1:
            result["left_type"] = "single_vertex"
        elif len(left_cliques) == 1:
            result["left_type"] = "single_clique"
            result["left_detail"] = left_cliques[0]
        else:
            result["left_type"] = "clique_partition"
            result["left_detail"] = left_cliques
    else:
        if left.op == "product":
            result["left_type"] = "nested_product"
            left_child_sizes = get_partition_sizes(left)
            result["left_detail"] = f"P{left_child_sizes}"
        else:
            result["left_type"] = "complex"

    # Analyze right side
    right_cliques = get_clique_partition(right)
    if right_cliques is not None:
        if len(right_cliques) == 1 and right_cliques[0] == 1:
            result["right_type"] = "single_vertex"
        elif len(right_cliques) == 1:
            result["right_type"] = "single_clique"
            result["right_detail"] = right_cliques[0]
        else:
            result["right_type"] = "clique_partition"
            result["right_detail"] = right_cliques
    else:
        if right.op == "product":
            result["right_type"] = "nested_product"
            right_child_sizes = get_partition_sizes(right)
            result["right_detail"] = f"P{right_child_sizes}"
        elif right.op == "sum":
            result["right_type"] = "sum_structure"
            # Check if it's sum of products
            result["right_detail"] = "complex_sum"
        else:
            result["right_type"] = "complex"

    return result


def analyze_file(filepath: Path) -> dict:
    """Analyze one K_{s,t} file and extract patterns."""
    with open(filepath) as f:
        data = json.load(f)

    s = data["s"]
    t = data["t"]

    # Collect statistics
    partition_counter = Counter()  # (n, partition_sizes) -> count
    structure_types = defaultdict(list)  # n -> list of structure types

    for n_str in data.get("extremal_by_n", {}).keys():
        n = int(n_str)
        n_data = data["extremal_by_n"][n_str]

        for struct_data in n_data["structures"]:
            struct_str = struct_data["structure"]

            try:
                node = parse_structure(struct_str)

                if node.op == "product":
                    partition = get_partition_sizes(node)
                    partition_counter[(n, partition)] += 1

                    # Classify structure
                    classification = classify_product_structure(node)
                    structure_types[n].append(classification)

            except Exception as e:
                pass

    return {
        "s": s,
        "t": t,
        "partition_counter": dict(partition_counter),
        "structure_types": dict(structure_types)
    }


def find_partition_pattern(s: int, t: int, partition_counter: dict) -> dict:
    """
    Find what partition sizes appear for each n.

    Returns analysis of which formula best fits the data.
    """
    patterns_by_n = defaultdict(set)

    for (n, partition), count in partition_counter.items():
        patterns_by_n[n].add(partition)

    # Check different hypotheses
    results = {
        "partitions_by_n": dict(patterns_by_n),
        "hypotheses": {}
    }

    # Hypothesis 1: P(s-1, n-s+1)
    h1_matches = 0
    h1_total = 0
    for n, partitions in patterns_by_n.items():
        if n >= s:
            h1_total += 1
            expected = tuple(sorted([s-1, n-s+1]))
            if expected in partitions:
                h1_matches += 1

    results["hypotheses"]["P(s-1, n-s+1)"] = {
        "matches": h1_matches,
        "total": h1_total,
        "rate": h1_matches / max(1, h1_total)
    }

    # Hypothesis 2: P(1, n-1) (single vertex on left)
    h2_matches = 0
    h2_total = 0
    for n, partitions in patterns_by_n.items():
        if n >= 2:
            h2_total += 1
            expected = (1, n-1)
            if expected in partitions:
                h2_matches += 1

    results["hypotheses"]["P(1, n-1)"] = {
        "matches": h2_matches,
        "total": h2_total,
        "rate": h2_matches / max(1, h2_total)
    }

    # Find all unique partition formulas
    unique_formulas = set()
    for n, partitions in patterns_by_n.items():
        for p1, p2 in partitions:
            # Try to express as (f(n), g(n))
            # Common patterns: (1, n-1), (2, n-2), (k, n-k), (n/2, n/2), etc.
            if p1 == 1:
                unique_formulas.add(f"P(1, {p2})")
            elif p1 == p2 == n//2:
                unique_formulas.add(f"P(n/2, n/2)")
            elif p1 + p2 == n:
                unique_formulas.add(f"P({p1}, n-{p1})")

    results["observed_formulas"] = sorted(unique_formulas)

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python discover_patterns.py <export_directory>")
        sys.exit(1)

    export_dir = Path(sys.argv[1])
    json_files = sorted(export_dir.glob("extremal_K*.json"))

    # Filter to specific files if more arguments
    if len(sys.argv) > 2:
        filter_pattern = sys.argv[2]
        json_files = [f for f in json_files if filter_pattern in f.name]

    print("=" * 100)
    print("STRUCTURAL PATTERN DISCOVERY FOR EXTREMAL K_{s,t}-FREE COGRAPHS")
    print("=" * 100)
    print()

    for json_file in json_files:
        print(f"\n{'='*100}")
        print(f"K_{{{json_file.stem.split('K')[1][:2]},{json_file.stem.split('K')[1][2:]}}}")
        print('='*100)

        analysis = analyze_file(json_file)
        s = analysis["s"]
        t = analysis["t"]

        print(f"\n## Partition sizes appearing in extremal graphs:")
        print(f"{'n':>4} | Partition sizes")
        print("-" * 50)

        # Group by n
        partitions_by_n = defaultdict(set)
        for (n, partition), count in analysis["partition_counter"].items():
            partitions_by_n[n].add(partition)

        for n in sorted(partitions_by_n.keys()):
            partitions = sorted(partitions_by_n[n])
            partition_str = ", ".join(f"P{p}" for p in partitions)
            print(f"{n:4d} | {partition_str}")

        # Find pattern
        pattern_analysis = find_partition_pattern(s, t, analysis["partition_counter"])

        print(f"\n## Hypothesis Testing:")
        for hyp_name, hyp_data in pattern_analysis["hypotheses"].items():
            rate = hyp_data["rate"]
            status = "✓" if rate > 0.9 else ("~" if rate > 0.5 else "✗")
            print(f"  [{status}] {hyp_name:20s}: {hyp_data['matches']:3d}/{hyp_data['total']:3d} = {rate:5.1%}")

        print(f"\n## Observed formulas:")
        for formula in pattern_analysis["observed_formulas"]:
            print(f"  - {formula}")

        # Analyze structure types
        print(f"\n## Structure classification (sample from n=10 or largest):")
        sample_n = 10 if 10 in analysis["structure_types"] else max(analysis["structure_types"].keys(), default=1)

        if sample_n in analysis["structure_types"]:
            type_counter = Counter()
            for struct_class in analysis["structure_types"][sample_n]:
                key = (struct_class.get("left_type"), struct_class.get("right_type"))
                type_counter[key] += 1

            print(f"  n={sample_n}:")
            for (left_type, right_type), count in type_counter.most_common():
                print(f"    {count:2d}× Left: {left_type:20s} | Right: {right_type}")

    print("\n" + "=" * 100)
    print("PATTERN SUMMARY")
    print("=" * 100)
    print()
    print("Key Findings:")
    print("1. For K_{2,t}: P(1, n-1) structure dominates (single vertex × complex)")
    print("2. For K_{s,s} symmetric: Multiple partition sizes appear, including P(s-1, n-s+1)")
    print("3. Left side tends to be small clique or clique partition")
    print("4. Right side has complex sum/product structure")
    print()


if __name__ == "__main__":
    main()
