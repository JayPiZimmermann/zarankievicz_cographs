#!/usr/bin/env python3
"""
Add depth field to export JSON files that are missing it.

This script:
1. Checks each JSON file in specified export directories
2. If depth field is missing, calculates it from the structure string
3. Updates the JSON file with the depth field added
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_conjectures import parse_structure
from generate_tables import get_tree_height


def add_depth_to_file(json_path: Path, dry_run: bool = False) -> tuple[bool, int]:
    """
    Add depth field to structures in a JSON file if missing.

    Args:
        json_path: Path to JSON file
        dry_run: If True, don't modify file, just report

    Returns:
        (modified, count) - whether file was modified and count of structures updated
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    modified = False
    count = 0

    for n_key, n_data in data.get('extremal_by_n', {}).items():
        for struct in n_data.get('structures', []):
            if 'depth' not in struct:
                # Calculate depth from structure string
                try:
                    structure_str = struct.get('structure', '')
                    node = parse_structure(structure_str)
                    depth = get_tree_height(node)
                    struct['depth'] = depth
                    modified = True
                    count += 1
                except Exception as e:
                    print(f"  Error parsing structure in {json_path.name}, n={n_key}: {e}")
                    continue

    if modified and not dry_run:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    return modified, count


def process_directory(dir_path: Path, dry_run: bool = False):
    """Process all JSON files in a directory."""
    json_files = sorted(dir_path.glob('extremal_K*.json'))

    if not json_files:
        print(f"{dir_path.name}: No JSON files found")
        return

    total_files = len(json_files)
    modified_files = 0
    total_structures = 0

    print(f"\n{dir_path.name}: Processing {total_files} files...")

    for json_file in json_files:
        was_modified, count = add_depth_to_file(json_file, dry_run=dry_run)
        if was_modified:
            modified_files += 1
            total_structures += count
            if count > 0:
                print(f"  {json_file.name}: added depth to {count} structures")

    if modified_files == 0:
        print(f"  ✓ All files already have depth field")
    else:
        action = "Would modify" if dry_run else "Modified"
        print(f"  {action} {modified_files}/{total_files} files ({total_structures} structures)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Add depth field to export JSON files")
    parser.add_argument('directories', nargs='*', default=['exports_star', 'exports_lattice_12_2', 'exports_lattice_5_2'],
                        help='Export directories to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without modifying files')

    args = parser.parse_args()

    print("Adding depth field to export JSON files...")
    if args.dry_run:
        print("(DRY RUN - no files will be modified)")

    for dir_name in args.directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            print(f"{dir_name}: Directory not found")
            continue

        process_directory(dir_path, dry_run=args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
