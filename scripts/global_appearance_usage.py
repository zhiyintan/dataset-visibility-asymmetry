#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple


def load_meta(meta_path: str, max_citations: Optional[int]) -> Dict[str, Dict[str, str]]:
    by_id: Dict[str, Dict[str, str]] = {}
    with open(meta_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rid = (row.get("ID") or "").strip()
            if not rid:
                continue
            if max_citations is not None:
                try:
                    c = int(row.get("citationCount") or 0)
                except ValueError:
                    c = 0
                if c > max_citations:
                    continue
            by_id[rid] = row
    return by_id


def parse_year_from_citing(citing: str) -> Optional[int]:
    if not citing:
        return None
    m = re.search(r"\((\d{4})\)", citing)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def run(checked_tsv: str, meta_tsv: str, out_tsv: str, out_png: str, max_citations: Optional[int], year_min: Optional[int], year_max: Optional[int]) -> None:
    meta = load_meta(meta_tsv, max_citations)

    ts_appearance: Dict[int, int] = defaultdict(int)
    ts_usage: Dict[int, int] = defaultdict(int)
    seen_appearance: Set[Tuple[str, int]] = set()
    seen_usage: Set[Tuple[str, int]] = set()

    with open(checked_tsv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rid = (row.get("ID") or "").strip()
            m = meta.get(rid)
            if not m:
                continue
            pid = (m.get("s2_paperId") or m.get("normalized_id") or "").strip()
            if not pid:
                continue
            # appearance by metadata year
            try:
                y_inv = int(m.get("year") or 0)
            except ValueError:
                y_inv = 0
            if y_inv and (pid, y_inv) not in seen_appearance:
                seen_appearance.add((pid, y_inv))
                ts_appearance[y_inv] += 1
            # usage by citing year
            cy = parse_year_from_citing(row.get("Citing Article") or "")
            if cy and (pid, cy) not in seen_usage:
                seen_usage.add((pid, cy))
                ts_usage[cy] += 1

    # Determine year range
    all_years = sorted(set(ts_appearance.keys()) | set(ts_usage.keys()))
    if not all_years:
        years = []
    else:
        y0 = year_min if year_min is not None else all_years[0]
        y1 = year_max if year_max is not None else all_years[-1]
        years = list(range(y0, y1 + 1))

    # Write TSV (filled missing years with 0)
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["year", "appearance", "usage"])
        for y in years:
            w.writerow([y, ts_appearance.get(y, 0), ts_usage.get(y, 0)])

    # Plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        app = [ts_appearance.get(y, 0) for y in years]
        use = [ts_usage.get(y, 0) for y in years]
        x = list(range(len(years)))

        sns.set_theme(style="whitegrid")
        sns.set_context("talk", font_scale=1.1)
        palette = sns.color_palette("tab10")
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=x, y=app, color=palette[0], linewidth=2.6, label="emergence")
        sns.lineplot(x=x, y=use, color=palette[1], linewidth=2.6, label="use")
        if len(years) > 0:
            step = max(1, len(years) // 12)
            tick_idx = list(range(0, len(years), step))
            tick_labels = [years[i] for i in tick_idx]
            plt.xticks(tick_idx, tick_labels, rotation=45, ha="right", fontsize=12)
        plt.xlabel("Year", fontsize=13)
        plt.ylabel("Count", fontsize=13)
        plt.legend(frameon=False, fontsize=20)
        plt.tight_layout()
        plt.savefig(out_png, dpi=220)
        plt.close()
    except Exception:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="Global appearance vs usage from checked.tsv")
    p.add_argument("--checked", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--out-tsv", required=True)
    p.add_argument("--out-png", required=True)
    p.add_argument("--max-citations", type=int, default=10000)
    p.add_argument("--year-min", type=int, default=None)
    p.add_argument("--year-max", type=int, default=None)
    a = p.parse_args()
    run(a.checked, a.meta, a.out_tsv, a.out_png, a.max_citations, a.year_min, a.year_max)


if __name__ == "__main__":
    main()
