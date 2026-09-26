import csv
import os
import sys

# Force UTF-8 output so international characters don't crash the terminal
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def inspect_tsv(filepath: str, label: str, max_rows: int = 5):
    """Inspect a TSV file using only Python built-ins. No pandas needed."""
    print("=" * 70)
    print(f"Dataset: {label}")
    print(f"File   : {filepath}")
    print("=" * 70)

    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}\n")
        return

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        headers = next(reader, [])
        rows = []
        for row in reader:
            rows.append(row)

    total = len(rows)
    print(f"Shape   : {total} rows  x  {len(headers)} columns")
    print(f"Columns : {headers}\n")

    # Missing value counts per column
    missing = [0] * len(headers)
    for row in rows:
        for i, val in enumerate(row):
            if i < len(missing) and (val == "" or val.strip().lower() == "nan"):
                missing[i] += 1

    print("Missing values per column:")
    for col, m in zip(headers, missing):
        pct = (m / total * 100) if total > 0 else 0
        print(f"  {col:<30} : {m:>6}  ({pct:.1f}%)")

    print(f"\nSample rows (first {min(max_rows, total)}):")
    col_widths = [min(len(h), 25) for h in headers]
    header_line = " | ".join(h[:w].ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows[:max_rows]:
        row_padded = row + [""] * (len(headers) - len(row))
        line = " | ".join(str(v)[:w].ljust(w) for v, w in zip(row_padded, col_widths))
        print(line)
    print()


# Run checks on all dataset files
FILES = [
    ("dataset/train_source1.tsv",      "Source 1"),
    ("dataset/train_source2.tsv",      "Source 2"),
    ("dataset/train_source3.tsv",      "Source 3"),
    ("dataset/train_ground_truth.tsv", "Ground Truth"),
]

for path, label in FILES:
    inspect_tsv(path, label)