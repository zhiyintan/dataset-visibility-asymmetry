import argparse
import os
from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def categorize_rdi(value: float) -> str:
    if pd.isna(value):
        return None
    if value == 0:
        return "0"
    if 0 < value <= 0.1:
        return "0–0.1"
    if 0.1 < value <= 0.2:
        return "0.1–0.2"
    if 0.2 < value <= 0.3:
        return "0.2–0.3"
    if 0.3 < value <= 0.4:
        return "0.3–0.4"
    if 0.4 < value <= 0.5:
        return "0.4–0.5"
    if 0.5 < value <= 0.6:
        return "0.5–0.6"
    if 0.6 < value <= 0.7:
        return "0.6–0.7"
    if 0.7 < value <= 0.8:
        return "0.7–0.8"
    if 0.8 < value <= 0.9:
        return "0.8–0.9"
    if 0.9 < value <= 1.0:
        return "0.9–1.0"
    if value > 1.0:
        return ">1.0"
    return None


def get_order() -> List[str]:
    return [
        "0",
        "0–0.1",
        "0.1–0.2",
        "0.2–0.3",
        "0.3–0.4",
        "0.4–0.5",
        "0.5–0.6",
        "0.6–0.7",
        "0.7–0.8",
        "0.8–0.9",
        "0.9–1.0",
        ">1.0",
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the RDI interval distribution (paper Figure 1). "
            "Input is a TSV with one row per language and a column of average "
            "LRE+LDC RDI values (datasets per million speakers). Population "
            "data are not redistributed here; obtain them from Ethnologue "
            "(https://www.ethnologue.com/insights/ethnologue200/) and join "
            "with tables/rdi_counts.tsv to compute the RDI column."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a TSV with a column of RDI values per language",
    )
    parser.add_argument(
        "--column",
        default="Avg. LRE & LDC (Index)",
        help="Column containing RDI values",
    )
    parser.add_argument(
        "--output",
        default="plots/rdi_bins.png",
        help="Output image path (png/pdf/svg)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t", dtype=str)
    if args.column not in df.columns:
        raise SystemExit(f"Column '{args.column}' not found in {args.input}. Columns: {list(df.columns)}")

    rdi = pd.to_numeric(df[args.column], errors="coerce")
    cats = rdi.map(categorize_rdi)

    order = get_order()
    cat_type = pd.CategoricalDtype(categories=order, ordered=True)
    cats = cats.astype(cat_type)
    counts = cats.value_counts().sort_index().reindex(order).fillna(0).astype(int)

    plot_df = counts.reset_index()
    plot_df.columns = ["RDI interval", "Language count"]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=plot_df, y="RDI interval", x="Language count", color="#4C72B0")
    ax.set_xlabel("Language count", fontsize=12)
    ax.set_ylabel("RDI interval", fontsize=12)

    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(
                f"{int(width)}",
                (width, p.get_y() + p.get_height() / 2.0),
                ha="left",
                va="center",
                fontsize=9,
                xytext=(3, 0),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    plt.close()


if __name__ == "__main__":
    main()


