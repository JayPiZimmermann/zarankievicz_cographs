#!/usr/bin/env python3
"""
CLI for extremal K_{s,t}-free cographs.

Usage:
    python main.py run --name test1 --N 20 --T 8   # Incremental build with saves
    python main.py check --name test1              # Check conjecture on run
    python main.py runs                            # List all runs
    python main.py build --N 20 --T 5              # Old-style single-file build
    python main.py analyze --n 10 --s 2 --t 3      # Analyze extremal graphs
"""

import argparse
import sys
from pathlib import Path

from src.registry import Registry
from src.builder import build_up_to, estimate_complexity
from src.cache import (
    save_registry, load_registry, cache_exists, list_caches,
    find_best_cache, clear_cache, cache_info
)
from src.export import (
    export_extremal_table, export_extremal_for_biclique,
    export_graphs_graph6, print_extremal_table, analyze_extremal,
    summarize_extremal, export_all
)
from src.incremental_cache import IncrementalRegistry, list_runs
from src.incremental_builder import build_incremental, check_conjecture, analyze_extremal_structure
from src.fast_builder import build_fast, check_conjecture_fast, export_extremal_analysis
from src.compact_storage import FastRegistry
from src.partition_builder import build_range, check_conjecture_partition


def cmd_build(args):
    """Build registry up to N vertices."""
    N = args.N
    T = args.T

    print(f"Building extremal cographs up to n={N}")
    if T:
        print(f"Pruning graphs containing K_{{{T},{T}}}")
    print(f"Complexity estimate: {estimate_complexity(N)}")
    print()

    # Check for existing cache to resume from
    registry = None
    if not args.fresh:
        best_cache = find_best_cache(N, T)
        if best_cache:
            print(f"Found existing cache: {best_cache}")
            registry, metadata = load_registry(best_cache)
            print(f"  Loaded registry with max_n={registry.max_n()}, {registry.total_graphs()} graphs")
            print()

    def progress(current_n, total_n, added, profiles, total):
        print(f"  n={current_n:3d}: {added:5d} new, {profiles:5d} profiles, {total:5d} total graphs")

    print("Building...")
    registry = build_up_to(N, T, registry, progress_callback=progress)

    print()
    print(f"Done! Registry has {registry.total_graphs()} graphs")

    # Save cache
    if args.save:
        path = save_registry(registry, N=N, T=T)
        print(f"Saved to: {path}")

    return registry


def cmd_export(args):
    """Export extremal data."""
    # Load registry
    registry = _load_registry_for_query(args)

    output_dir = Path(args.output) if args.output else Path("exports")

    if args.all:
        # Export everything
        paths = export_all(registry, output_dir, args.s_max, args.t_max)
        print(f"Exported to {output_dir}:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    elif args.s and args.t:
        # Export specific (s,t)
        if args.format == "csv":
            path = output_dir / f"extremal_K{args.s}{args.t}.csv"
            export_extremal_table(registry, args.s, args.t, path)
            print(f"Exported to: {path}")
        elif args.format == "json":
            path = output_dir / f"extremal_K{args.s}{args.t}.json"
            export_extremal_for_biclique(registry, args.s, args.t, path)
            print(f"Exported to: {path}")
        elif args.format == "graph6":
            if not args.n:
                print("Error: --n required for graph6 format")
                return
            path = output_dir / f"graphs_n{args.n}_K{args.s}{args.t}.g6"
            count = export_graphs_graph6(registry, args.n, args.s, args.t, path)
            print(f"Exported {count} graphs to: {path}")
    else:
        # Default: print table
        print_extremal_table(registry, args.s_max, args.t_max)


def cmd_analyze(args):
    """Analyze extremal graphs."""
    registry = _load_registry_for_query(args)

    if args.s and args.t:
        if args.n:
            # Specific n, s, t
            graphs = registry.get_avoiding(args.n, args.s, args.t)
            if not graphs:
                print(f"No K_{{{args.s},{args.t}}}-free graphs on {args.n} vertices")
                return

            print(f"Extremal K_{{{args.s},{args.t}}}-free cographs on {args.n} vertices:")
            print(f"  ex({args.n}, K_{{{args.s},{args.t}}}) = {graphs[0][0]}")
            print(f"  Number of extremal graphs: {len(graphs)}")
            print()

            for i, (edges, cotree) in enumerate(graphs):
                analysis = analyze_extremal(cotree)
                print(f"Graph {i+1}:")
                print(f"  Structure: {analysis['structure_str']}")
                print(f"  Last operation: {analysis['last_op']}")
                if analysis['component_sizes']:
                    print(f"  Component sizes: {analysis['component_sizes']}")
                    print(f"  Component edges: {analysis['component_edges']}")
                print(f"  Depth: {analysis['depth']}")
                print()
        else:
            # Summary for all n
            summary = summarize_extremal(registry, args.s, args.t)
            print(f"Summary of K_{{{args.s},{args.t}}}-free extremal cographs:")
            print()
            for entry in summary:
                print(f"n={entry['n']:3d}: ex={entry['ex']:4d}, count={entry['count']:3d}, ops={entry['last_ops']}")
                for struct in entry['structures'][:2]:
                    print(f"         {struct}")
                if len(entry['structures']) > 2:
                    print(f"         ... (+{len(entry['structures']) - 2} more)")
    else:
        # General statistics
        stats = registry.statistics()
        print("Registry statistics:")
        print(f"  Maximum n: {stats['max_n']}")
        print(f"  Total graphs: {stats['total_graphs']}")
        print()
        print("By vertex count:")
        for n, info in sorted(stats['by_n'].items()):
            print(f"  n={n:3d}: {info['profiles']:5d} profiles, {info['graphs']:5d} graphs")


def cmd_cache(args):
    """Manage cache files."""
    if args.show:
        caches = list_caches()
        if not caches:
            print("No cache files found")
            return

        print("Available caches:")
        for cache in caches:
            info = cache_info(cache['path'])
            meta = info['metadata']
            size_kb = info['size_bytes'] / 1024
            print(f"  {cache['path'].name}")
            print(f"    N={cache['N']}, T={cache['T']}")
            print(f"    Graphs: {meta.get('total_graphs', '?')}, Size: {size_kb:.1f} KB")
            print(f"    Saved: {meta.get('saved_at', '?')}")
            print()

    elif args.clear:
        count = clear_cache()
        print(f"Cleared {count} cache files")

    elif args.info:
        path = Path(args.info)
        if not path.exists():
            print(f"File not found: {path}")
            return
        info = cache_info(path)
        print(f"Cache: {path}")
        print(f"  Size: {info['size_bytes'] / 1024:.1f} KB")
        print(f"  Vertex counts: {min(info['vertex_counts'])} to {max(info['vertex_counts'])}")
        for key, val in info['metadata'].items():
            print(f"  {key}: {val}")


def cmd_run(args):
    """Incremental build with per-n saves."""
    print(f"Incremental build: run='{args.name}', N={args.N}, T={args.T}")
    print(f"Exports for s,t in [2..{args.s_max}] x [2..{args.t_max}]")
    print()

    def progress(n, N, added, skipped, profiles, total, cumulative):
        print(f"  n={n:3d}: +{added:5d} graphs, {skipped:5d} profile pairs skipped, "
              f"{profiles:4d} profiles, {total:5d} for n, {cumulative:7d} cumulative")

    registry = build_incremental(
        run_name=args.name,
        N=args.N,
        T=args.T,
        s_max=args.s_max,
        t_max=args.t_max,
        progress_callback=progress
    )

    print()
    print(f"Done! Run '{args.name}' complete.")
    print(f"  Max n: {registry.max_n()}")
    print(f"  Exports in: {registry.exports_dir}")


def cmd_check(args):
    """Check conjecture on extremal graph structure."""
    # Load the run
    registry = IncrementalRegistry(args.name, N=100, T=None)  # N/T will be loaded from metadata
    if not registry.exists():
        print(f"Run '{args.name}' not found")
        return

    start_n = registry.load_existing()
    print(f"Loaded run '{args.name}' with n up to {registry.max_n()}")
    print()

    # Check conjecture
    results = check_conjecture(
        registry,
        s_min=args.s_min,
        s_max=args.s_max,
        t_max=args.t_max
    )

    print("=" * 70)
    print("CONJECTURE CHECK: Are all extremal K_{s,t}-free graphs connected?")
    print("                  (i.e., is the last operation always 'product'?)")
    print("=" * 70)
    print()

    if results["all_connected"]:
        print("RESULT: YES - All extremal graphs for s,t >= 2 are connected (product)")
    else:
        print("RESULT: NO - Found exceptions:")
        for exc in results["exceptions"]:
            print(f"  n={exc['n']}, K_{{{exc['s']},{exc['t']}}}: "
                  f"{exc['structure']} (last_op={exc['last_op']})")

    print()
    print("Component size analysis (min component size in P(a,b) for connected graphs):")
    print("-" * 70)

    for key, st_result in results["by_st"].items():
        if not st_result["typical_first_component"]:
            continue
        print(f"\n{key}:")
        sizes_by_n = {}
        for n, size in st_result["typical_first_component"]:
            if n not in sizes_by_n:
                sizes_by_n[n] = []
            sizes_by_n[n].append(size)

        for n in sorted(sizes_by_n.keys()):
            sizes = sizes_by_n[n]
            if len(set(sizes)) == 1:
                print(f"  n={n:3d}: min_component = {sizes[0]}")
            else:
                print(f"  n={n:3d}: min_component in {set(sizes)}")


def cmd_runs(args):
    """List all runs."""
    runs = list_runs()
    if not runs:
        print("No runs found")
        return

    print("Available runs:")
    print("-" * 70)
    for run in runs:
        status = "COMPLETE" if run["completed_n"] >= run["N"] else f"at n={run['completed_n']}/{run['N']}"
        print(f"  {run['name']}")
        print(f"    N={run['N']}, T={run['T']}, status: {status}")
        print(f"    Graphs: {run['total_graphs']:,}, Last update: {run['last_update']}")
        print()


def cmd_fast(args):
    """Fast parallel build."""
    from multiprocessing import cpu_count

    N = args.N
    T = args.T
    workers = args.workers or cpu_count()

    print(f"Fast parallel build: N={N}, T={T}, workers={workers}")
    print(f"Checkpoint dir: {args.checkpoint_dir or 'disabled'}")
    if args.export_dir:
        print(f"Export dir: {args.export_dir} (incremental)")
        print(f"Exporting K_{{s,t}} for s,t in [2..{args.s_max}] x [2..{args.t_max}]")
    if args.profile_domination:
        print("Profile domination: ENABLED (batch mode)")
    if args.profile_domination_lattice:
        print("Profile domination: ENABLED (lattice mode - pre-filter combinations)")
    if args.depth_domination:
        print("Depth domination: ENABLED")
    print()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    export_dir = Path(args.export_dir) if args.export_dir else None

    def progress(n, total_n, added, profiles, total_n_graphs, cumulative, elapsed):
        print(f"  n={n:3d}: +{added:6d} graphs, {profiles:5d} profiles, "
              f"{total_n_graphs:6d} for n, {cumulative:8d} total, {elapsed:.1f}s")

    registry = build_fast(
        N=N,
        T=T,
        num_workers=workers,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        export_dir=export_dir,
        s_max=args.s_max,
        t_max=args.t_max,
        use_profile_domination=args.profile_domination,
        use_profile_domination_lattice=args.profile_domination_lattice,
        use_depth_domination=args.depth_domination,
        progress_callback=progress
    )

    print()
    print(f"Done! Total graphs: {registry.total_graphs():,}")

    # Check conjecture
    if args.check:
        print("\nChecking conjecture...")
        results = check_conjecture_fast(registry, s_min=2, s_max=args.s_max, t_max=args.t_max)

        if results["all_connected"]:
            print("RESULT: All extremal graphs for s,t >= 2 are CONNECTED (product)")
        else:
            print("RESULT: Found DISCONNECTED extremal graphs:")
            for exc in results["exceptions"]:
                print(f"  n={exc['n']}, K_{{{exc['s']},{exc['t']}}}: {exc['structure']}")

        print("\nComponent size patterns:")
        for key, st_result in results["by_st"].items():
            if st_result["component_sizes"]:
                sizes = st_result["component_sizes"]
                # Group by n
                by_n = {}
                for n, size in sizes:
                    if n not in by_n:
                        by_n[n] = set()
                    by_n[n].add(size)
                print(f"  {key}: ", end="")
                sample = [(n, sorted(s)) for n, s in sorted(by_n.items())[:5]]
                print(", ".join(f"n={n}→{sizes}" for n, sizes in sample), "...")

    return registry


def cmd_partition(args):
    """Partition-based build: compute each n from n'+n'' partitions."""
    from multiprocessing import cpu_count

    start_n = args.start_n
    end_n = args.end_n
    T = args.T
    S = args.S
    S_max = args.S_max
    workers = args.workers or cpu_count()

    print(f"Partition-based build: n=[{start_n}..{end_n}], workers={workers}")

    # Display pruning configuration
    if S is not None and S_max is not None:
        print(f"Pruning: K_{{{S},{S_max}}} (--S={S}, --S-max={S_max})")
        print(f"Profile truncation: storing only up to index {S}")
    elif T is not None:
        print(f"Pruning: K_{{{T},{T}}} (legacy --T={T})")
    else:
        print(f"Pruning: disabled")

    print(f"Checkpoint dir: {args.checkpoint_dir or 'disabled'}")
    if args.export_dir:
        print(f"Export dir: {args.export_dir} (incremental)")
        effective_s = S if S is not None else args.s_max
        if S is not None and S_max is None:
            print(f"Exporting K_{{s,t}} for s in [1..{effective_s}], t in [s..n] (N-independent)")
        else:
            effective_t = S_max if S_max is not None else args.t_max
            print(f"Exporting K_{{s,t}} for s in [1..{effective_s}], t in [s..{effective_t}]")
    if args.profile_domination:
        print("Profile domination: ENABLED (batch mode)")
    if args.profile_domination_lattice:
        print("Profile domination: ENABLED (lattice mode - pre-filter combinations)")
    if args.depth_domination:
        print("Depth domination: ENABLED")
    print()
    print("NOTE: Each n is computed independently from partitions n'+n''=n")
    print("      This makes computation independent of end_n, enabling true incremental updates")
    print()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    export_dir = Path(args.export_dir) if args.export_dir else None

    def progress(n, end_n, added, profiles, total_n_graphs, cumulative, elapsed):
        print(f"  n={n:3d}: +{added:6d} graphs, {profiles:5d} profiles at n, "
              f"{total_n_graphs:6d} graphs at n, {cumulative:8d} total, {elapsed:.1f}s")

    registry = build_range(
        start_n=start_n,
        end_n=end_n,
        registry=None,
        T=T,
        num_workers=workers,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        export_dir=export_dir,
        s_max=args.s_max,
        t_max=args.t_max,
        use_profile_domination=args.profile_domination,
        use_profile_domination_lattice=args.profile_domination_lattice,
        use_depth_domination=args.depth_domination,
        progress_callback=progress,
        S=S,
        S_max=S_max
    )

    print()
    print(f"Done! Total graphs: {registry.total_graphs():,}")

    # Check conjecture
    if args.check:
        print("\nChecking conjecture...")
        results = check_conjecture_partition(registry, s_min=1, s_max=args.s_max, t_max=args.t_max)

        if results["all_connected"]:
            print("RESULT: All extremal graphs for s,t >= 2 are CONNECTED (product)")
        else:
            print("RESULT: Found DISCONNECTED extremal graphs:")
            for exc in results["exceptions"]:
                print(f"  n={exc['n']}, K_{{{exc['s']},{exc['t']}}}: {exc['structure']}")

        print("\nComponent size patterns:")
        for key, st_result in results["by_st"].items():
            if st_result["component_sizes"]:
                sizes = st_result["component_sizes"]
                # Group by n
                by_n = {}
                for n, size in sizes:
                    if n not in by_n:
                        by_n[n] = set()
                    by_n[n].add(size)
                print(f"  {key}: ", end="")
                sample = [(n, sorted(s)) for n, s in sorted(by_n.items())[:5]]
                print(", ".join(f"n={n}→{sizes}" for n, sizes in sample), "...")

    return registry


def _load_registry_for_query(args) -> Registry:
    """Load registry for query commands."""
    if hasattr(args, 'cache') and args.cache:
        registry, _ = load_registry(path=args.cache)
        return registry

    # Find best available cache
    N = getattr(args, 'N', None) or 20
    T = getattr(args, 'T', None)

    best = find_best_cache(N, T)
    if best:
        registry, _ = load_registry(best)
        return registry

    # No cache, need to build
    print("No cache found. Building registry...")
    registry = build_up_to(N, T)
    return registry


def main():
    parser = argparse.ArgumentParser(
        description="Extremal K_{s,t}-free cographs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # build command
    build_parser = subparsers.add_parser("build", help="Build registry")
    build_parser.add_argument("--N", type=int, required=True, help="Maximum vertex count")
    build_parser.add_argument("--T", type=int, help="Prune K_{T,T} containing graphs")
    build_parser.add_argument("--save", action="store_true", default=True, help="Save to cache")
    build_parser.add_argument("--no-save", dest="save", action="store_false", help="Don't save")
    build_parser.add_argument("--fresh", action="store_true", help="Start fresh, ignore cache")

    # export command
    export_parser = subparsers.add_parser("export", help="Export extremal data")
    export_parser.add_argument("--s", type=int, help="Left side of K_{s,t}")
    export_parser.add_argument("--t", type=int, help="Right side of K_{s,t}")
    export_parser.add_argument("--n", type=int, help="Specific vertex count (for graph6)")
    export_parser.add_argument("--s-max", type=int, default=4, help="Max s for tables")
    export_parser.add_argument("--t-max", type=int, default=4, help="Max t for tables")
    export_parser.add_argument("--format", choices=["csv", "json", "graph6"], default="json")
    export_parser.add_argument("--output", "-o", help="Output directory")
    export_parser.add_argument("--all", action="store_true", help="Export all formats")
    export_parser.add_argument("--cache", help="Specific cache file to use")
    export_parser.add_argument("--N", type=int, default=20, help="Max N for auto-loading cache")
    export_parser.add_argument("--T", type=int, help="T threshold for auto-loading cache")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze extremal graphs")
    analyze_parser.add_argument("--n", type=int, help="Specific vertex count")
    analyze_parser.add_argument("--s", type=int, help="Left side of K_{s,t}")
    analyze_parser.add_argument("--t", type=int, help="Right side of K_{s,t}")
    analyze_parser.add_argument("--cache", help="Specific cache file to use")
    analyze_parser.add_argument("--N", type=int, default=20, help="Max N for auto-loading")
    analyze_parser.add_argument("--T", type=int, help="T threshold for auto-loading")

    # cache command
    cache_parser = subparsers.add_parser("cache", help="Manage cache")
    cache_parser.add_argument("--show", action="store_true", help="Show available caches")
    cache_parser.add_argument("--clear", action="store_true", help="Clear all caches")
    cache_parser.add_argument("--info", help="Show info for specific cache file")

    # run command (incremental build)
    run_parser = subparsers.add_parser("run", help="Incremental build with per-n saves")
    run_parser.add_argument("--name", required=True, help="Run name (used in folder)")
    run_parser.add_argument("--N", type=int, required=True, help="Maximum vertex count")
    run_parser.add_argument("--T", type=int, help="Prune K_{T,T} containing graphs")
    run_parser.add_argument("--s-max", type=int, default=7, help="Max s for exports")
    run_parser.add_argument("--t-max", type=int, default=7, help="Max t for exports")

    # check command (check conjecture)
    check_parser = subparsers.add_parser("check", help="Check conjecture on run")
    check_parser.add_argument("--name", required=True, help="Run name to analyze")
    check_parser.add_argument("--s-min", type=int, default=2, help="Min s to check")
    check_parser.add_argument("--s-max", type=int, default=7, help="Max s to check")
    check_parser.add_argument("--t-max", type=int, default=7, help="Max t to check")

    # runs command (list runs)
    runs_parser = subparsers.add_parser("runs", help="List all runs")

    # fast command (parallel build)
    fast_parser = subparsers.add_parser("fast", help="Fast parallel build")
    fast_parser.add_argument("--N", type=int, required=True, help="Maximum vertex count")
    fast_parser.add_argument("--T", type=int, help="Prune K_{T,T} containing graphs")
    fast_parser.add_argument("--workers", "-w", type=int, help="Number of workers (default: cpu_count)")
    fast_parser.add_argument("--checkpoint-dir", help="Directory for checkpoints")
    fast_parser.add_argument("--checkpoint-interval", type=int, default=5, help="Checkpoint every N values")
    fast_parser.add_argument("--export-dir", help="Export results to directory")
    fast_parser.add_argument("--s-max", type=int, default=7, help="Max s for exports")
    fast_parser.add_argument("--t-max", type=int, default=7, help="Max t for exports")
    fast_parser.add_argument("--profile-domination", action="store_true", help="Enable profile domination pruning (batch mode)")
    fast_parser.add_argument("--profile-domination-lattice", action="store_true", help="Enable lattice-based profile domination (pre-filter combinations)")
    fast_parser.add_argument("--depth-domination", action="store_true", help="Enable depth domination pruning")
    fast_parser.add_argument("--check", action="store_true", help="Check conjecture after build")

    # partition command (N-independent build)
    partition_parser = subparsers.add_parser("partition", help="Partition-based build (N-independent)")
    partition_parser.add_argument("--start-n", type=int, default=2, help="Starting vertex count")
    partition_parser.add_argument("--end-n", type=int, required=True, help="Ending vertex count")
    partition_parser.add_argument("--T", type=int, help="Prune K_{T,T} containing graphs (legacy)")
    partition_parser.add_argument("--S", type=int, help="Truncate profiles to index S and export K_{i,j} for i<=S, j<=n")
    partition_parser.add_argument("--S-max", type=int, help="Prune K_{S,S_max} and limit exports to j<=S_max")
    partition_parser.add_argument("--workers", "-w", type=int, help="Number of workers (default: cpu_count)")
    partition_parser.add_argument("--checkpoint-dir", help="Directory for checkpoints")
    partition_parser.add_argument("--checkpoint-interval", type=int, default=5, help="Checkpoint every N values")
    partition_parser.add_argument("--export-dir", help="Export results to directory")
    partition_parser.add_argument("--s-max", type=int, default=7, help="Max s for exports (overridden by --S)")
    partition_parser.add_argument("--t-max", type=int, default=7, help="Max t for exports (overridden by --S-max)")
    partition_parser.add_argument("--profile-domination", action="store_true", help="Enable profile domination pruning (batch mode)")
    partition_parser.add_argument("--profile-domination-lattice", action="store_true", help="Enable lattice-based profile domination (pre-filter combinations)")
    partition_parser.add_argument("--depth-domination", action="store_true", help="Enable depth domination pruning")
    partition_parser.add_argument("--check", action="store_true", help="Check conjecture after build")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "cache":
        cmd_cache(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "runs":
        cmd_runs(args)
    elif args.command == "fast":
        cmd_fast(args)
    elif args.command == "partition":
        cmd_partition(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
