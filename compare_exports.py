#!/usr/bin/env python3
"""
Compare extremal numbers between two export directories to ensure consistency.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_extremal_data(json_path):
    """Load extremal data from JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def compare_directories(dir1, dir2):
    """
    Compare all matching extremal_*.json files between two directories.

    Returns: (success, differences)
    """
    dir1 = Path(dir1)
    dir2 = Path(dir2)

    # Find all extremal_*.json files in both directories
    files1 = {f.name: f for f in dir1.glob("extremal_*.json")}
    files2 = {f.name: f for f in dir2.glob("extremal_*.json")}

    # Find common files
    common_files = set(files1.keys()) & set(files2.keys())

    if not common_files:
        print(f"No common files found between {dir1} and {dir2}")
        return True, []

    print(f"Comparing {len(common_files)} common files...")
    print(f"Files: {sorted(common_files)}")
    print()

    all_success = True
    differences = []

    for filename in sorted(common_files):
        path1 = files1[filename]
        path2 = files2[filename]

        # Extract s,t from filename (e.g., "extremal_K23.json" -> s=2, t=3)
        # Format is extremal_K{s}{t}.json
        basename = filename.replace("extremal_K", "").replace(".json", "")
        if len(basename) == 2:
            s, t = int(basename[0]), int(basename[1])
        else:
            # Handle double digits
            print(f"Warning: Cannot parse {filename}, skipping")
            continue

        print(f"Checking K_{{{s},{t}}}-free graphs ({filename})...")

        try:
            data1 = load_extremal_data(path1)
            data2 = load_extremal_data(path2)

            # Get extremal_by_n for both
            ex_by_n1 = data1.get("extremal_by_n", {})
            ex_by_n2 = data2.get("extremal_by_n", {})

            # Find common n values
            n_values1 = set(int(k) for k in ex_by_n1.keys())
            n_values2 = set(int(k) for k in ex_by_n2.keys())

            # Compare up to minimum n
            max_n1 = max(n_values1) if n_values1 else 0
            max_n2 = max(n_values2) if n_values2 else 0
            min_max_n = min(max_n1, max_n2)

            print(f"  Directory 1 max n: {max_n1}")
            print(f"  Directory 2 max n: {max_n2}")
            print(f"  Comparing up to n={min_max_n}")

            # Compare extremal numbers (must match) and structure counts (informational only)
            edge_mismatches = []
            count_differences = []

            for n in range(2, min_max_n + 1):
                n_str = str(n)

                if n_str not in ex_by_n1 or n_str not in ex_by_n2:
                    continue

                ex1 = ex_by_n1[n_str]["ex"]
                ex2 = ex_by_n2[n_str]["ex"]
                count1 = ex_by_n1[n_str]["count"]
                count2 = ex_by_n2[n_str]["count"]

                # Extremal number mismatch is an ERROR
                if ex1 != ex2:
                    edge_mismatches.append((n, ex1, ex2))
                    all_success = False
                    differences.append({
                        "file": filename,
                        "s": s,
                        "t": t,
                        "n": n,
                        "type": "edges",
                        "dir1_value": ex1,
                        "dir2_value": ex2
                    })

                # Structure count difference is INFORMATIONAL (not an error)
                if count1 != count2:
                    count_differences.append((n, count1, count2))
                    differences.append({
                        "file": filename,
                        "s": s,
                        "t": t,
                        "n": n,
                        "type": "count",
                        "dir1_value": count1,
                        "dir2_value": count2
                    })

            if edge_mismatches:
                print(f"  ❌ EXTREMAL NUMBER MISMATCH (ERROR):")
                for n, ex1, ex2 in edge_mismatches:
                    print(f"     n={n}: dir1={ex1} edges, dir2={ex2} edges")

            if count_differences:
                print(f"  ℹ️  Structure count differences (informational):")
                for n, c1, c2 in count_differences:
                    print(f"     n={n}: dir1={c1} structures, dir2={c2} structures")

            if not edge_mismatches and not count_differences:
                print(f"  ✓ All extremal numbers and counts match (n=2 to {min_max_n})")
            elif not edge_mismatches:
                print(f"  ✓ All extremal numbers match (n=2 to {min_max_n})")

            print()

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            all_success = False
            print()

    return all_success, differences


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_exports.py <dir1> <dir2>")
        print()
        print("Example:")
        print("  python compare_exports.py exports_lattice exports")
        sys.exit(1)

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]

    print("=" * 70)
    print("EXTREMAL NUMBER COMPARISON")
    print("=" * 70)
    print(f"Directory 1: {dir1}")
    print(f"Directory 2: {dir2}")
    print()

    success, differences = compare_directories(dir1, dir2)

    print("=" * 70)
    if success:
        print("✓ SUCCESS: All extremal numbers and structure counts match!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("❌ FAILURE: Inconsistencies found!")
        print("=" * 70)
        print()
        print("Summary of differences:")

        edge_diffs = [d for d in differences if d.get('type') == 'edges']
        count_diffs = [d for d in differences if d.get('type') == 'count']

        if edge_diffs:
            print("\nExtremal number (edge count) differences:")
            for diff in edge_diffs:
                print(f"  K_{{{diff['s']},{diff['t']}}}-free, n={diff['n']}: "
                      f"{diff['dir1_value']} edges vs {diff['dir2_value']} edges")

        if count_diffs:
            print("\nStructure count differences:")
            for diff in count_diffs:
                print(f"  K_{{{diff['s']},{diff['t']}}}-free, n={diff['n']}: "
                      f"{diff['dir1_value']} structures vs {diff['dir2_value']} structures")

        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
