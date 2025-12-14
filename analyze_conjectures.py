#!/usr/bin/env python3
"""
Analyze extremal K_{s,t}-free cograph structures and check conjectures.

Conjectures to test:
1. For s <= t, extremal graph is P(s-1, n-s+1)
2. Structure is either:
   a) Complete multipartite on smaller side with product partner
   b) Smaller side fully connected, other side has cliques
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, List, Tuple
from collections import defaultdict, Counter


@dataclass
class StructureNode:
    """Parsed structure tree."""
    op: Literal["vertex", "sum", "product"]
    n: int  # Number of vertices
    children: List['StructureNode']

    def __repr__(self):
        if self.op == "vertex":
            return f"V({self.n})"
        elif self.op == "sum":
            return f"S({self.children[0]},{self.children[1]})"
        else:  # product
            return f"P({self.children[0]},{self.children[1]})"

    def to_short_str(self):
        """Compact representation."""
        if self.op == "vertex":
            return "1" if self.n == 1 else f"{self.n}"
        op_char = "S" if self.op == "sum" else "P"
        return f"{op_char}({self.children[0].to_short_str()},{self.children[1].to_short_str()})"


def parse_structure(s: str) -> StructureNode:
    """
    Parse structure string like 'P(S(1,1),P(1,1))' into tree.

    Grammar:
        struct ::= '1' | 'S(' struct ',' struct ')' | 'P(' struct ',' struct ')'
    """
    s = s.strip()

    if s == '1':
        return StructureNode(op="vertex", n=1, children=[])

    # Check for S(...) or P(...)
    if s[0] in ['S', 'P']:
        op = "sum" if s[0] == 'S' else "product"

        # Find the matching comma at the top level
        # Parse S( or P(
        assert s[1] == '(', f"Expected '(' after {s[0]}"

        # Find matching comma and closing paren
        depth = 0
        comma_pos = -1
        for i in range(2, len(s)):
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
            elif s[i] == ',' and depth == 0:
                comma_pos = i
                break

        assert comma_pos > 0, f"Could not find comma in {s}"
        assert s[-1] == ')', f"Expected ')' at end of {s}"

        left_str = s[2:comma_pos]
        right_str = s[comma_pos+1:-1]

        left = parse_structure(left_str)
        right = parse_structure(right_str)

        n = left.n + right.n
        return StructureNode(op=op, n=n, children=[left, right])

    raise ValueError(f"Could not parse structure: {s}")


def count_vertices(node: StructureNode) -> int:
    """Count total vertices in structure."""
    return node.n


def analyze_structure(node: StructureNode, depth=0) -> dict:
    """
    Analyze structural properties of a cograph.

    Returns:
        dict with properties like:
        - last_op: 'sum', 'product', or 'vertex'
        - depth: tree depth
        - component_sizes: if last op is sum or product
        - is_complete_multipartite: for product graphs
        - clique_partition: for sum graphs
    """
    info = {
        "last_op": node.op,
        "depth": depth,
        "n": node.n,
    }

    if node.op == "vertex":
        info["is_clique"] = True
        info["clique_size"] = node.n
        return info

    # Recurse on children
    left_info = analyze_structure(node.children[0], depth + 1)
    right_info = analyze_structure(node.children[1], depth + 1)

    info["left"] = left_info
    info["right"] = right_info
    info["component_sizes"] = [node.children[0].n, node.children[1].n]

    if node.op == "sum":
        # Sum = disjoint union
        # Check if it's a clique partition (all components are cliques)
        left_cliques = get_clique_partition(node.children[0])
        right_cliques = get_clique_partition(node.children[1])

        if left_cliques is not None and right_cliques is not None:
            info["is_clique_partition"] = True
            info["clique_sizes"] = left_cliques + right_cliques
        else:
            info["is_clique_partition"] = False

    elif node.op == "product":
        # Product = complete join
        # Check if sides are clique partitions
        left_cliques = get_clique_partition(node.children[0])
        right_cliques = get_clique_partition(node.children[1])

        info["left_cliques"] = left_cliques
        info["right_cliques"] = right_cliques

        if left_cliques is not None and right_cliques is not None:
            info["is_complete_multipartite"] = True
            info["partition_sizes"] = left_cliques + right_cliques
        else:
            info["is_complete_multipartite"] = False

    return info


def get_clique_partition(node: StructureNode) -> List[int] | None:
    """
    If node represents a clique partition (sum of vertices), return clique sizes.
    Otherwise return None.
    """
    if node.op == "vertex":
        return [node.n]

    if node.op == "sum":
        left = get_clique_partition(node.children[0])
        right = get_clique_partition(node.children[1])

        if left is not None and right is not None:
            return left + right
        return None

    # Product is not a clique partition
    return None


def is_single_vertex(node: StructureNode) -> bool:
    """Check if node is a single vertex."""
    return node.op == "vertex" and node.n == 1


def get_structure_pattern(node: StructureNode) -> str:
    """
    Get a pattern description of the structure.

    Examples:
    - "vertex"
    - "sum_of_cliques"
    - "product(sum_of_cliques, sum_of_cliques)"
    - "product(vertex, complex)"
    """
    if node.op == "vertex":
        return f"clique_{node.n}"

    if node.op == "sum":
        cliques = get_clique_partition(node)
        if cliques:
            return f"sum_of_cliques_{sorted(cliques, reverse=True)}"
        else:
            left = get_structure_pattern(node.children[0])
            right = get_structure_pattern(node.children[1])
            return f"sum({left}, {right})"

    # Product
    left = get_structure_pattern(node.children[0])
    right = get_structure_pattern(node.children[1])
    return f"product({left}, {right})"


def check_conjecture_1(node: StructureNode, n: int, s: int, t: int) -> dict:
    """
    Conjecture 1: For s <= t, extremal graph is P(s-1, n-s+1).

    This only makes sense when n >= s, otherwise s-1 or n-s+1 could be invalid.

    Returns:
        dict with 'holds', 'expected_sizes', 'actual_sizes', 'details'
    """
    result = {
        "holds": False,
        "expected_sizes": None,
        "actual_sizes": None,
        "details": ""
    }

    # Check if root is a product
    if node.op != "product":
        result["details"] = f"Not product"
        return result

    actual_sizes = sorted([node.children[0].n, node.children[1].n])
    result["actual_sizes"] = actual_sizes

    # Conjecture only applies when n >= s
    if n < s:
        result["details"] = f"n<s, P({actual_sizes[0]},{actual_sizes[1]})"
        return result

    expected_sizes = sorted([s - 1, n - s + 1])
    result["expected_sizes"] = expected_sizes

    if actual_sizes == expected_sizes:
        result["holds"] = True
        result["details"] = f"✓"
    else:
        result["details"] = f"P({actual_sizes[0]},{actual_sizes[1]})≠P({expected_sizes[0]},{expected_sizes[1]})"

    return result


def check_conjecture_2(node: StructureNode, n: int, s: int, t: int) -> dict:
    """
    Conjecture 2: Structure is either:
    a) Complete multipartite graph (product of clique partitions)
    b) One side fully connected, other side is clique partition

    Returns detailed analysis.
    """
    result = {
        "holds": False,
        "case": None,
        "details": ""
    }

    if node.op != "product":
        result["details"] = "Not a product graph"
        return result

    left = node.children[0]
    right = node.children[1]

    left_cliques = get_clique_partition(left)
    right_cliques = get_clique_partition(right)

    # Case a: Both sides are clique partitions
    if left_cliques is not None and right_cliques is not None:
        result["holds"] = True
        result["case"] = "complete_multipartite"
        result["details"] = f"Complete multipartite: partition {left_cliques} × {right_cliques}"
        result["left_cliques"] = left_cliques
        result["right_cliques"] = right_cliques
        return result

    # Case b: One side fully connected (single clique), other is clique partition
    if left_cliques is not None and len(left_cliques) == 1 and right_cliques is None:
        result["holds"] = True
        result["case"] = "one_clique_vs_complex"
        result["details"] = f"Left is single clique {left_cliques[0]}, right is complex"
        result["left_cliques"] = left_cliques
        return result

    if right_cliques is not None and len(right_cliques) == 1 and left_cliques is None:
        result["holds"] = True
        result["case"] = "one_clique_vs_complex"
        result["details"] = f"Right is single clique {right_cliques[0]}, left is complex"
        result["right_cliques"] = right_cliques
        return result

    # Neither case holds
    result["details"] = f"Neither case: left_cliques={left_cliques}, right_cliques={right_cliques}"

    return result


def analyze_extremal_file(filepath: Path) -> dict:
    """Load and analyze a single extremal JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    s = data["s"]
    t = data["t"]

    results = {
        "s": s,
        "t": t,
        "by_n": {}
    }

    for n_str, n_data in data.get("extremal_by_n", {}).items():
        n = int(n_str)
        ex = n_data["ex"]
        count = n_data["count"]
        structures = n_data["structures"]

        # Analyze each extremal structure
        structure_analyses = []
        for struct_data in structures:
            struct_str = struct_data["structure"]
            edges = struct_data["edges"]

            # Parse structure
            try:
                node = parse_structure(struct_str)
                info = analyze_structure(node)
                pattern = get_structure_pattern(node)

                # Check conjectures
                conj1 = check_conjecture_1(node, n, s, t)
                conj2 = check_conjecture_2(node, n, s, t)

                structure_analyses.append({
                    "structure": struct_str,
                    "edges": edges,
                    "parsed": node,
                    "info": info,
                    "pattern": pattern,
                    "conjecture_1": conj1,
                    "conjecture_2": conj2
                })
            except Exception as e:
                structure_analyses.append({
                    "structure": struct_str,
                    "edges": edges,
                    "error": str(e)
                })

        results["by_n"][n] = {
            "ex": ex,
            "count": count,
            "structures": structure_analyses
        }

    return results


def main():
    """Main analysis program."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_conjectures.py <export_directory>")
        print("  Analyzes extremal_K*.json files in the directory")
        sys.exit(1)

    export_dir = Path(sys.argv[1])

    if not export_dir.exists():
        print(f"Error: Directory {export_dir} does not exist")
        sys.exit(1)

    # Find all extremal JSON files
    json_files = sorted(export_dir.glob("extremal_K*.json"))

    if not json_files:
        print(f"No extremal_K*.json files found in {export_dir}")
        sys.exit(1)

    print("=" * 80)
    print("EXTREMAL K_{s,t}-FREE COGRAPH STRUCTURE ANALYSIS")
    print("=" * 80)
    print()
    print(f"Found {len(json_files)} files to analyze")
    print()

    # Track statistics
    conj1_stats = {"total": 0, "holds": 0, "fails": 0}
    conj2_stats = {"total": 0, "holds": 0, "fails": 0}
    pattern_counts = Counter()
    exceptions = []

    for json_file in json_files:
        print(f"\n{'='*80}")
        print(f"Analyzing: {json_file.name}")
        print('='*80)

        results = analyze_extremal_file(json_file)
        s = results["s"]
        t = results["t"]

        print(f"\nK_{{{s},{t}}}-free extremal graphs:")
        print(f"{'n':>4} {'ex':>6} {'cnt':>4} {'C1':>3} {'C2':>3} {'Pattern':<40} {'Details'}")
        print("-" * 80)

        for n in sorted(results["by_n"].keys()):
            n_data = results["by_n"][n]

            for i, struct_analysis in enumerate(n_data["structures"]):
                if "error" in struct_analysis:
                    print(f"{n:4d} {n_data['ex']:6d} {n_data['count']:4d} ERR ERR {struct_analysis['error']}")
                    continue

                conj1 = struct_analysis["conjecture_1"]
                conj2 = struct_analysis["conjecture_2"]
                pattern = struct_analysis["pattern"]

                # Update stats
                conj1_stats["total"] += 1
                if conj1["holds"]:
                    conj1_stats["holds"] += 1
                else:
                    conj1_stats["fails"] += 1
                    exceptions.append({
                        "file": json_file.name,
                        "s": s, "t": t, "n": n,
                        "conjecture": 1,
                        "structure": struct_analysis["structure"],
                        "details": conj1
                    })

                conj2_stats["total"] += 1
                if conj2["holds"]:
                    conj2_stats["holds"] += 1
                else:
                    conj2_stats["fails"] += 1
                    exceptions.append({
                        "file": json_file.name,
                        "s": s, "t": t, "n": n,
                        "conjecture": 2,
                        "structure": struct_analysis["structure"],
                        "details": conj2
                    })

                pattern_counts[pattern] += 1

                # Print row
                c1_mark = "✓" if conj1["holds"] else "✗"
                c2_mark = "✓" if conj2["holds"] else "✗"

                if i == 0:
                    print(f"{n:4d} {n_data['ex']:6d} {n_data['count']:4d} ", end="")
                else:
                    print(f"{'':4s} {'':6s} {'':4s} ", end="")

                print(f"{c1_mark:>3} {c2_mark:>3} {pattern:<40.40s}", end="")

                # Add details for failures
                if not conj1["holds"]:
                    print(f" {conj1['details']}", end="")
                if not conj2["holds"]:
                    print(f" {conj2['details']}", end="")

                print()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nConjecture 1: P({s-1}, {n-(s-1)}) structure")
    print(f"  Total cases: {conj1_stats['total']}")
    print(f"  Holds: {conj1_stats['holds']} ({100*conj1_stats['holds']/max(1,conj1_stats['total']):.1f}%)")
    print(f"  Fails: {conj1_stats['fails']} ({100*conj1_stats['fails']/max(1,conj1_stats['total']):.1f}%)")

    print(f"\nConjecture 2: Complete multipartite or special structure")
    print(f"  Total cases: {conj2_stats['total']}")
    print(f"  Holds: {conj2_stats['holds']} ({100*conj2_stats['holds']/max(1,conj2_stats['total']):.1f}%)")
    print(f"  Fails: {conj2_stats['fails']} ({100*conj2_stats['fails']/max(1,conj2_stats['total']):.1f}%)")

    print(f"\nMost common patterns:")
    for pattern, count in pattern_counts.most_common(10):
        print(f"  {count:4d} × {pattern}")

    if exceptions:
        print(f"\n{'='*80}")
        print(f"EXCEPTIONS ({len(exceptions)} total)")
        print('='*80)

        # Group by conjecture
        conj1_exceptions = [e for e in exceptions if e["conjecture"] == 1]
        conj2_exceptions = [e for e in exceptions if e["conjecture"] == 2]

        if conj1_exceptions:
            print(f"\nConjecture 1 failures ({len(conj1_exceptions)}):")
            for exc in conj1_exceptions[:20]:  # Show first 20
                print(f"  K_{{{exc['s']},{exc['t']}}}, n={exc['n']}: {exc['details']['details']}")
                print(f"    Structure: {exc['structure']}")

        if conj2_exceptions:
            print(f"\nConjecture 2 failures ({len(conj2_exceptions)}):")
            for exc in conj2_exceptions[:20]:  # Show first 20
                print(f"  K_{{{exc['s']},{exc['t']}}}, n={exc['n']}: {exc['details']['details']}")
                print(f"    Structure: {exc['structure']}")


if __name__ == "__main__":
    main()
