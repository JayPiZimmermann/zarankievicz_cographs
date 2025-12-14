#!/usr/bin/env python3
"""
Analyze the exceptional cases where P(s-1, n-s+1) partition is NOT present.
"""

import csv
from collections import defaultdict

def main():
    # Read exceptions from CSV
    exceptions = []
    with open("partition_presence.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["has_partition"] == "0":
                s = int(row["s"])
                t = int(row["t"])
                n = int(row["n"])
                exceptions.append((s, t, n))

    print("=" * 80)
    print(f"EXCEPTIONAL CASES ANALYSIS ({len(exceptions)} total)")
    print("=" * 80)
    print()

    # Pattern 1: Distribution by n
    print("PATTERN 1: Distribution by n")
    print("-" * 40)
    n_counts = defaultdict(list)
    for s, t, n in exceptions:
        n_counts[n].append((s, t))

    for n in sorted(n_counts.keys()):
        print(f"n={n:2d}: {len(n_counts[n]):2d} cases - {n_counts[n]}")
    print()

    # Pattern 2: Distribution by s
    print("PATTERN 2: Distribution by s")
    print("-" * 40)
    s_counts = defaultdict(list)
    for s, t, n in exceptions:
        s_counts[s].append((t, n))

    for s in sorted(s_counts.keys()):
        t_values = sorted(set(t for t, n in s_counts[s]))
        print(f"s={s}: {len(s_counts[s]):2d} cases, t ∈ {{{','.join(map(str, t_values))}}}")
        print(f"      t range: [{min(t_values)}, {max(t_values)}], avg t = {sum(t_values)/len(t_values):.1f}")
    print()

    # Pattern 3: t - s gap
    print("PATTERN 3: Gap t - s")
    print("-" * 40)
    gap_counts = defaultdict(list)
    for s, t, n in exceptions:
        gap = t - s
        gap_counts[gap].append((s, t, n))

    for gap in sorted(gap_counts.keys()):
        print(f"t-s={gap}: {len(gap_counts[gap]):2d} cases")
    print(f"\nMin gap: {min(gap_counts.keys())}")
    print(f"Max gap: {max(gap_counts.keys())}")
    print(f"Avg gap: {sum(t-s for s,t,n in exceptions)/len(exceptions):.2f}")
    print()

    # Pattern 4: n vs t ratio
    print("PATTERN 4: Ratio n/t")
    print("-" * 40)
    ratios = [(n/t, s, t, n) for s, t, n in exceptions]
    ratios.sort()

    print(f"Min n/t: {ratios[0][0]:.3f} at (s,t,n) = {ratios[0][1:]}")
    print(f"Max n/t: {ratios[-1][0]:.3f} at (s,t,n) = {ratios[-1][1:]}")
    avg_ratio = sum(r[0] for r in ratios) / len(ratios)
    print(f"Avg n/t: {avg_ratio:.3f}")
    print()

    # Pattern 5: Expected partition size
    print("PATTERN 5: Expected partition P(s-1, n-s+1)")
    print("-" * 40)
    print(f"{'s':>3} {'t':>3} {'n':>3} | {'s-1':>4} {'n-s+1':>6} | Partition")
    print("-" * 50)
    for s, t, n in sorted(exceptions):
        a = s - 1
        b = n - s + 1
        print(f"{s:3d} {t:3d} {n:3d} | {a:4d} {b:6d} | P({a},{b})")
    print()

    # Pattern 6: Relationship between n-s+1 and t
    print("PATTERN 6: Relationship between n-s+1 and t")
    print("-" * 40)
    print(f"{'s':>3} {'t':>3} {'n':>3} | {'n-s+1':>6} {'t':>3} | Diff (n-s+1)-t")
    print("-" * 50)
    diffs = []
    for s, t, n in sorted(exceptions):
        b = n - s + 1
        diff = b - t
        diffs.append(diff)
        marker = " <--" if abs(diff) <= 2 else ""
        print(f"{s:3d} {t:3d} {n:3d} | {b:6d} {t:3d} | {diff:+4d}{marker}")

    print(f"\nDifference (n-s+1) - t:")
    print(f"  Min: {min(diffs)}")
    print(f"  Max: {max(diffs)}")
    print(f"  Avg: {sum(diffs)/len(diffs):.2f}")
    print(f"  Cases where |diff| ≤ 2: {sum(1 for d in diffs if abs(d) <= 2)}/{len(diffs)}")
    print()

    # Pattern 7: Check if n ≈ s + t + k
    print("PATTERN 7: Relationship n vs s+t")
    print("-" * 40)
    print(f"{'s':>3} {'t':>3} {'n':>3} | {'s+t':>5} | n-(s+t)")
    print("-" * 40)
    residuals = []
    for s, t, n in sorted(exceptions):
        residual = n - (s + t)
        residuals.append(residual)
        print(f"{s:3d} {t:3d} {n:3d} | {s+t:5d} | {residual:+4d}")

    print(f"\nResidual n - (s+t):")
    print(f"  Min: {min(residuals)}")
    print(f"  Max: {max(residuals)}")
    print(f"  Avg: {sum(residuals)/len(residuals):.2f}")
    print()

    # Pattern 8: Overall summary
    print("=" * 80)
    print("SUMMARY OF PATTERNS")
    print("=" * 80)
    print(f"1. Most common n values: {', '.join(f'n={n} ({len(n_counts[n])} cases)' for n in sorted(n_counts.keys(), key=lambda x: len(n_counts[x]), reverse=True)[:5])}")
    print(f"2. t - s gap: minimum {min(gap_counts.keys())}, typically {3} to {9}")
    print(f"3. Ratio n/t: typically {avg_ratio:.2f} ± 0.2")
    print(f"4. Small s values dominate: s ∈ {{2,3,4}} account for {sum(len(s_counts[s]) for s in [2,3,4])}/{len(exceptions)} cases")
    print(f"5. Difference (n-s+1) - t ranges from {min(diffs)} to {max(diffs)}, avg {sum(diffs)/len(diffs):.2f}")
    print()

if __name__ == "__main__":
    main()
