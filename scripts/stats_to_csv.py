#!/usr/bin/env python3
"""Convert stats_*.json files to CSV tables for LaTeX."""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from typing import Dict, List

DEFAULT_ANSWER_TYPES = [
    "fact",
    "inference",
    "list",
    "numeric",
    "comparison",
    "procedural",
    "other",
    "unknown",
]


def load_stats(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_title(label: str) -> str:
    return label.replace("_", " ").title()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CSV summaries from stats JSON files.")
    parser.add_argument(
        "stats",
        nargs="+",
        help="Mappings of the form split=path/to/stats.json",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write CSV outputs",
    )
    parser.add_argument(
        "--counts-csv",
        default="figure_train_test_stats.csv",
        help="Filename for the split totals CSV",
    )
    parser.add_argument(
        "--answers-csv",
        default="figure_answer_type_distribution.csv",
        help="Filename for the answer-type counts CSV",
    )
    parser.add_argument(
        "--answers-long-csv",
        default="figure_answer_type_distribution_long.csv",
        help="Filename for the long-form answer-type CSV",
    )
    return parser.parse_args()


def parse_mappings(mappings: List[str]) -> OrderedDict:
    out: OrderedDict[str, str] = OrderedDict()
    for item in mappings:
        if "=" not in item:
            raise SystemExit(f"Invalid mapping '{item}'. Use split=path form.")
        split, path = item.split("=", 1)
        split = split.strip()
        path = path.strip()
        if not split or not path:
            raise SystemExit(f"Invalid mapping '{item}'.")
        out[split] = path
    return out


def determine_answer_types(stats_list: List[Dict[str, int]]) -> List[str]:
    types = OrderedDict.fromkeys(DEFAULT_ANSWER_TYPES)
    for stats in stats_list:
        counts = stats.get("answer_type_positive_counts") or {}
        for key in counts.keys():
            if key not in types:
                types[key] = None
    return list(types.keys())


def write_counts_csv(out_path: str, splits: OrderedDict[str, str], stats_map: Dict[str, Dict[str, int]]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Split,Total,Positive,Negative\n")
        for split, _ in splits.items():
            stats = stats_map[split]
            total = stats.get("total_examples", 0)
            positives = stats.get("positives", 0)
            negatives = stats.get("negatives", 0)
            f.write(f"{to_title(split)},{total},{positives},{negatives}\n")


def write_answers_csv(
    out_path: str,
    splits: OrderedDict[str, str],
    stats_map: Dict[str, Dict[str, int]],
    answer_types: List[str],
) -> None:
    header = ["Split"] + [to_title(t) for t in answer_types]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for split, _ in splits.items():
            stats = stats_map[split]
            counts = stats.get("answer_type_positive_counts") or {}
            row = [to_title(split)]
            for t in answer_types:
                row.append(str(counts.get(t, 0)))
            f.write(",".join(row) + "\n")


def write_answers_long_csv(
    out_path: str,
    splits: OrderedDict[str, str],
    stats_map: Dict[str, Dict[str, int]],
    answer_types: List[str],
) -> None:
    header = ["AnswerType"] + [to_title(split) for split in splits.keys()]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for answer_type in answer_types:
            row = [to_title(answer_type)]
            for split in splits.keys():
                stats = stats_map[split]
                counts = stats.get("answer_type_positive_counts") or {}
                row.append(str(counts.get(answer_type, 0)))
            f.write(",".join(row) + "\n")


def main() -> None:
    args = parse_args()
    mappings = parse_mappings(args.stats)
    stats_map: Dict[str, Dict[str, int]] = {}
    for split, path in mappings.items():
        resolved = os.path.abspath(path)
        if not os.path.exists(resolved):
            raise SystemExit(f"Stats file not found: {resolved}")
        stats_map[split] = load_stats(resolved)

    os.makedirs(args.out_dir, exist_ok=True)
    counts_csv = os.path.join(args.out_dir, args.counts_csv)
    answers_csv = os.path.join(args.out_dir, args.answers_csv)
    answers_long_csv = os.path.join(args.out_dir, args.answers_long_csv)

    answer_types = determine_answer_types(list(stats_map.values()))

    write_counts_csv(counts_csv, mappings, stats_map)
    write_answers_csv(answers_csv, mappings, stats_map, answer_types)
    write_answers_long_csv(answers_long_csv, mappings, stats_map, answer_types)

    print(f"Wrote {counts_csv}")
    print(f"Wrote {answers_csv}")
    print(f"Wrote {answers_long_csv}")


if __name__ == "__main__":
    main()
