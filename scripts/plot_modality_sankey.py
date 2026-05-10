import argparse
import os
from typing import Dict, List, Tuple, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle


def clean_language_name(raw_name: str) -> str:
    if not isinstance(raw_name, str):
        return ""
    name = raw_name.strip()
    if name.endswith(" Corpus"):
        name = name[: -len(" Corpus")]
    if name.endswith(" Language"):
        name = name[: -len(" Language")]
    return name


def compute_links(
    df: pd.DataFrame,
    top_languages: int,
    top_uses: int,
    top_modalities: int,
    exclude_languages: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    df = df.copy()
    df["language_clean"] = df["language"].map(clean_language_name)
    df = df[(df["use"].astype(str).str.strip() != "") & (df["modality"].astype(str).str.strip() != "") & (df["language_clean"].astype(str).str.strip() != "")]

    # Exclude specified languages from the visualization
    if exclude_languages:
        excl = {name.strip() for name in exclude_languages if str(name).strip()}
        if excl:
            df = df[~df["language_clean"].isin(excl)]

    # Collapse infrequent uses into 'Other uses' if requested
    if top_uses and top_uses > 0:
        use_totals_all = df.groupby("use").size().sort_values(ascending=False)
        keep_uses = set(use_totals_all.head(top_uses).index)
    else:
        keep_uses = set(df["use"].unique())

    df["use2"] = df["use"].where(df["use"].isin(keep_uses), other="Other uses")

    # Collapse infrequent modalities into 'Other modalities' if requested
    if top_modalities and top_modalities > 0:
        mod_totals_all = df.groupby("modality").size().sort_values(ascending=False)
        keep_modalities = set(mod_totals_all.head(top_modalities).index)
    else:
        keep_modalities = set(df["modality"].unique())

    df["modality2"] = df["modality"].where(df["modality"].isin(keep_modalities), other="Other modalities")

    use_to_modality = df.groupby(["use2", "modality2"]).size().reset_index(name="count").rename(columns={"use2": "use", "modality2": "modality"})
    modality_to_language = df.groupby(["modality2", "language_clean"]).size().reset_index(name="count").rename(columns={"modality2": "modality"})

    language_totals = modality_to_language.groupby("language_clean")["count"].sum().sort_values(ascending=False)
    top_language_set = set(language_totals.head(top_languages).index)

    modality_to_language.loc[~modality_to_language["language_clean"].isin(top_language_set), "language_clean"] = "Other languages"
    modality_to_language = modality_to_language.groupby(["modality", "language_clean"]).agg(count=("count", "sum")).reset_index()

    use_nodes = use_to_modality.groupby("use")["count"].sum().sort_values(ascending=False).index.tolist()
    modality_nodes = sorted(use_to_modality["modality"].dropna().unique().tolist())
    language_nodes = sorted(modality_to_language["language_clean"].dropna().unique().tolist())

    return use_to_modality, modality_to_language, use_nodes, modality_nodes, language_nodes


def _stack_positions(
    items: List[Tuple[str, float]],
    total: float,
    start_y: float = 0.0,
    gap: float = 0.0,
) -> Dict[str, Tuple[float, float]]:
    positions: Dict[str, Tuple[float, float]] = {}
    y = start_y
    # scale values so that sum(heights) + total_gaps == 1.0
    num_items = len(items)
    usable_height = max(0.0, 1.0 - gap * max(0, num_items - 1))
    scale = 0.0 if total == 0 else usable_height / total
    for idx, (key, value) in enumerate(items):
        height = value * scale
        positions[key] = (y, y + height)
        y += height
        # add gap after each box except the last one
        if idx < len(items) - 1:
            y += gap
    return positions


def _ribbon(ax, x0: float, x1: float, y0a: float, y0b: float, y1a: float, y1b: float, color: str, alpha: float = 0.35) -> None:
    ctrl_dx = (x1 - x0) * 0.35
    verts = [
        (x0, y0a),
        (x0 + ctrl_dx, y0a),
        (x1 - ctrl_dx, y1a),
        (x1, y1a),
        (x1, y1b),
        (x1 - ctrl_dx, y1b),
        (x0 + ctrl_dx, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    patch = PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(patch)


def _lighten_color(color: Tuple[float, float, float], amount: float = 0.15) -> Tuple[float, float, float]:
    # Move color towards white by proportion `amount`
    r, g, b = color
    return (
        min(1.0, r + (1.0 - r) * amount),
        min(1.0, g + (1.0 - g) * amount),
        min(1.0, b + (1.0 - b) * amount),
    )


def _neon_boost(color: Tuple[float, float, float], amount: float = 0.25) -> Tuple[float, float, float]:
    # Boost saturation and brightness for a more neon-like look
    import colorsys

    r, g, b = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = min(1.0, s + amount)
    l = min(1.0, l + amount * 0.15)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2)


def draw_alluvial(use_to_modality: pd.DataFrame, modality_to_language: pd.DataFrame, use_nodes: List[str], modality_nodes: List[str], language_nodes: List[str], language_label_map: Dict[str, str], figsize: Tuple[float, float], output_path: str, dpi: int, gap: float, brighten: float, neon: float, link_alpha: float) -> None:
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=figsize)

    column_x = {"use": 0.0, "modality": 1.0, "language": 2.0}
    box_width = 0.10

    use_totals = use_to_modality.groupby("use")["count"].sum()
    modality_totals = pd.concat([
        use_to_modality.groupby("modality")["count"].sum(),
        modality_to_language.groupby("modality")["count"].sum(),
    ], axis=1).max(axis=1)
    language_totals = modality_to_language.groupby("language_clean")["count"].sum()

    grand_total = float(max(use_totals.sum(), modality_totals.sum(), language_totals.sum()))

    use_positions = _stack_positions([(k, use_totals.get(k, 0.0)) for k in use_nodes], total=grand_total, gap=gap)
    modality_positions = _stack_positions([(k, modality_totals.get(k, 0.0)) for k in modality_nodes], total=grand_total, gap=gap)
    language_positions = _stack_positions([(k, language_totals.get(k, 0.0)) for k in language_nodes], total=grand_total, gap=gap)

    # vivid categorical palette for modalities; uses will inherit dominant modality hue
    palette_mod = sns.color_palette("tab10", max(3, len(modality_nodes)))
    lang_color_base = (0.65, 0.65, 0.65)

    mod_color_map = {name: palette_mod[i % len(palette_mod)] for i, name in enumerate(modality_nodes)}

    # Override specific modality hues to avoid overly aggressive colors
    written_blue = (0.298, 0.471, 0.659)  # #4C78A8
    modality_color_overrides: Dict[str, Tuple[float, float, float]] = {
        "Written": written_blue,
        "TEXT": written_blue,
    }
    for key, col in modality_color_overrides.items():
        if key in mod_color_map:
            mod_color_map[key] = col

    # Force multimodal family to red (case-insensitive common variants)
    red_strong = (0.894, 0.102, 0.110)  # ColorBrewer red
    multimodal_aliases = {
        "multimodal",
        "multimodal/multimedia",
        "multi-modal",
        "multimodel",
        "multi model",
        "multimedia",
        "multi media",
    }
    for k in list(mod_color_map.keys()):
        if str(k).strip().lower() in multimodal_aliases:
            mod_color_map[k] = red_strong

    # Brighten colors for better legibility
    if brighten and brighten > 0:
        mod_color_map = {k: _lighten_color(v, brighten) for k, v in mod_color_map.items()}
        lang_color = _lighten_color(lang_color_base, brighten)
    else:
        lang_color = lang_color_base

    # Apply neon boost if requested
    if neon and neon > 0:
        mod_color_map = {k: _neon_boost(v, neon) for k, v in mod_color_map.items()}

    # Determine dominant modality for each use (max count)
    dominant_modality_by_use: Dict[str, str] = {}
    for use_name, grp in use_to_modality.groupby("use"):
        dom_row = grp.sort_values("count", ascending=False).iloc[0]
        dominant_modality_by_use[use_name] = str(dom_row["modality"])  # modality name

    # Use colors inherit the dominant modality color; fall back to modality color mapping
    use_color_map: Dict[str, Tuple[float, float, float]] = {}
    for use_name in use_nodes:
        dom_mod = dominant_modality_by_use.get(use_name)
        use_color_map[use_name] = mod_color_map.get(dom_mod, (0.5, 0.5, 0.5))

    # Pre-compute per-column scale factors to adjust ribbon thickness to column height
    scale_use = max(0.0, 1.0 - gap * max(0, len(use_nodes) - 1))
    scale_mod = max(0.0, 1.0 - gap * max(0, len(modality_nodes) - 1))
    scale_lang = max(0.0, 1.0 - gap * max(0, len(language_nodes) - 1))

    # Draw ribbons use -> modality
    current_offsets_use: Dict[Tuple[str, str], float] = {}
    current_offsets_mod_in: Dict[Tuple[str, str], float] = {}

    for _, row in use_to_modality.sort_values(["use", "modality"]).iterrows():
        use_name = row["use"]
        mod_name = row["modality"]
        value_raw = float(row["count"]) / grand_total

        y0_start, y0_end = use_positions[use_name]
        used_height = sum(v for (u, _m), v in current_offsets_use.items() if u == use_name)
        y0a = y0_start + used_height
        val_use = value_raw * scale_use
        y0b = y0a + val_use
        current_offsets_use[(use_name, mod_name)] = val_use

        y1_start, y1_end = modality_positions[mod_name]
        used_height_mod = sum(v for (_m, l), v in current_offsets_mod_in.items() if _m == mod_name)
        y1a = y1_start + used_height_mod
        val_mod = value_raw * scale_mod
        y1b = y1a + val_mod
        current_offsets_mod_in[(mod_name, use_name)] = val_mod

        _ribbon(
            ax,
            column_x["use"] + box_width,
            column_x["modality"] - box_width,
            y0a,
            y0b,
            y1a,
            y1b,
            color=use_color_map[use_name],
            alpha=max(0.05, min(1.0, link_alpha)),
        )

    # Draw ribbons modality -> language
    current_offsets_mod_out: Dict[Tuple[str, str], float] = {}
    current_offsets_lang: Dict[Tuple[str, str], float] = {}
    for _, row in modality_to_language.sort_values(["modality", "language_clean"]).iterrows():
        mod_name = row["modality"]
        lang_name = row["language_clean"]
        value_raw = float(row["count"]) / grand_total

        y0_start, y0_end = modality_positions[mod_name]
        used_height_mod_out = sum(v for (m, _l), v in current_offsets_mod_out.items() if m == mod_name)
        y0a = y0_start + used_height_mod_out
        val_mod = value_raw * scale_mod
        y0b = y0a + val_mod
        current_offsets_mod_out[(mod_name, lang_name)] = val_mod

        y1_start, y1_end = language_positions[lang_name]
        used_height_lang = sum(v for (l, _), v in current_offsets_lang.items() if l == lang_name)
        y1a = y1_start + used_height_lang
        val_lang = value_raw * scale_lang
        y1b = y1a + val_lang
        current_offsets_lang[(lang_name, mod_name)] = val_lang

        _ribbon(
            ax,
            column_x["modality"] + box_width,
            column_x["language"] - box_width,
            y0a,
            y0b,
            y1a,
            y1b,
            color=mod_color_map[mod_name],
            alpha=max(0.05, min(1.0, link_alpha)),
        )

    # Draw boxes and labels
    def _draw_column(
        nodes: List[str],
        positions: Dict[str, Tuple[float, float]],
        x_center: float,
        color_map,
        edge_color: str = "#444",
        text_position: str = "outside_left",  # one of: outside_left, outside_right, inside_left, inside_right, inside_center
        label_positions_y: Optional[Dict[str, float]] = None,
    ) -> None:
        for name in nodes:
            y0, y1 = positions[name]
            height = max(0.0001, y1 - y0)
            rect = Rectangle((x_center - box_width, y0), 2 * box_width, height, facecolor=color_map.get(name, lang_color), edgecolor=edge_color, lw=1.0)
            ax.add_patch(rect)

            pad = 0.01
            if text_position == "inside_left":
                x_text = x_center - box_width + pad
                ha = "left"
            elif text_position == "inside_right":
                x_text = x_center + box_width - pad
                ha = "right"
            elif text_position == "inside_center":
                x_text = x_center
                ha = "center"
            elif text_position == "outside_right":
                x_text = x_center + box_width + 0.02
                ha = "left"
            else:  # outside_left
                x_text = x_center - box_width - 0.02
                ha = "right"

            y_text = label_positions_y[name] if label_positions_y and name in label_positions_y else (y0 + y1) / 2.0
            ax.text(x_text, y_text, name, va="center", ha=ha, fontsize=18)

    _draw_column(use_nodes, use_positions, column_x["use"], use_color_map, text_position="outside_right")

    # Compute non-overlapping label positions for modality labels placed to the right of the middle column
    def _resolve_label_collisions(names: List[str], positions: Dict[str, Tuple[float, float]], min_gap: float = 0.02) -> Dict[str, float]:
        centers = [(name, (positions[name][0] + positions[name][1]) / 2.0) for name in names]
        centers.sort(key=lambda x: x[1])
        adjusted: Dict[str, float] = {}
        last_y = -1.0
        margin = 0.01
        for name, y in centers:
            y_adj = max(y, last_y + min_gap) if last_y >= 0 else max(y, margin)
            adjusted[name] = y_adj
            last_y = y_adj
        # If the last label goes beyond 1 - margin, shift all down uniformly
        overflow = (last_y + 0.0) - (1.0 - margin)
        if overflow > 0:
            for k in adjusted:
                adjusted[k] = max(margin, adjusted[k] - overflow)
        return adjusted

    modality_label_y = _resolve_label_collisions(modality_nodes, modality_positions, min_gap=0.02)
    _draw_column(modality_nodes, modality_positions, column_x["modality"], mod_color_map, text_position="outside_right", label_positions_y=modality_label_y)
    # Languages use a single grey color for boxes
    lang_color_map = {name: lang_color for name in language_nodes}
    for name in language_nodes:
        y0, y1 = language_positions[name]
        height = max(0.0001, y1 - y0)
        rect = Rectangle((column_x["language"] - box_width, y0), 2 * box_width, height, facecolor=lang_color_map[name], edgecolor="#444", lw=1.0)
        ax.add_patch(rect)
        label = language_label_map.get(name, name)
        ax.text(column_x["language"] - box_width - 0.02, (y0 + y1) / 2.0, label, va="center", ha="right", fontsize=18)

    ax.set_xlim(-0.15, 2.15)
    # Estimate used vertical space; keep within [0, 1] by design of gaps
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alluvial chart (seaborn/matplotlib) for use → modality → language")
    parser.add_argument("--input", default="tables/checked_modality_use.tsv", help="Input TSV path containing columns: use, modality, language")
    parser.add_argument("--output", default="plots/modality_sankey_seaborn.pdf", help="Output figure path (pdf/png)")
    parser.add_argument("--top_languages", type=int, default=18, help="Top-N languages to show; others grouped as 'Other languages'")
    parser.add_argument("--top_uses", type=int, default=10, help="Top-N uses to keep; others grouped as 'Other uses' (0 to disable)")
    parser.add_argument("--top_modalities", type=int, default=3, help="Top-N modalities to keep; others grouped as 'Other modalities' (0 to disable)")
    parser.add_argument("--width", type=float, default=14.0, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=8.0, help="Figure height in inches")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    parser.add_argument("--gap", type=float, default=0.006, help="Vertical gap between boxes (in axis fraction)")
    parser.add_argument("--brighten", type=float, default=0.18, help="Lighten colors by this fraction (0-1)")
    parser.add_argument("--neon", type=float, default=0.25, help="Neon-like boost for modality hues (0-1)")
    parser.add_argument("--link_alpha", type=float, default=0.45, help="Ribbon transparency (0-1, lower is lighter)")
    parser.add_argument(
        "--exclude_languages",
        type=str,
        default="Levantine Arabic,Tunisian Arabic",
        help="Comma-separated language names to exclude",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t", dtype=str)
    required = {"use", "modality", "language"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Missing required columns: {sorted(required - set(df.columns))}")

    exclude_list = [s for s in (args.exclude_languages.split(",") if args.exclude_languages else [])]

    use_to_modality, modality_to_language, use_nodes, modality_nodes, language_nodes = compute_links(
        df,
        top_languages=args.top_languages,
        top_uses=args.top_uses,
        top_modalities=args.top_modalities,
        exclude_languages=exclude_list,
    )
    # ISO 639 code mapping (languages not listed remain as their original names)
    iso639_map: Dict[str, str] = {
        "Setswana": "tsn",
        "Bavarian": "bar",
        "Tatar": "tat",
        "Kyrgyz": "kir",
        "Nepali": "npi",
        "Southern Kurdish": "sdh",
        "Central Pashto": "pst",
        "Northern Sotho": "nso",
        "Dholuo": "luo",
        "Ghanaian Pidgin English": "gpe",
        "Sindhi": "snd",
        "Odia": "ory",
        "Turkmen": "tuk",
        "Southern Sotho": "sot",
        "Cameroon Pidgin": "wes",
        "Northern Pashto": "pbu",
        "Northern Thai": "nod",
        "Eastern Oromo": "hae",
        "Sicilian": "scn",
        "Southern Uzbek": "uzs",
        "Burmese": "mya",
        "Tiv": "tiv",
        "Nigerian Pidgin": "pcm",
        "Zarma": "dje",
        "Tigrigna": "tir",
        "Sylheti": "syl",
        "Northeastern Thai": "tts",
        "Bhojpuri": "bho",
        "Eastern Punjabi": "pan",
        "Xhosa": "xho",
        "Magahi": "mag",
        "Sadri": "sck",
        "Rundi": "run",
        "Chittagonian": "ctg",
        "Sudanese Arabic": "apd",
        "Chichewa": "nya",
        "Maithili": "mai",
        "Najdi Arabic": "ars",
        "Lingala": "lin",
        "Western Punjabi": "pnb",
        "Adamawa Fulfulde": "fub",
        "Awadhi": "awa",
        "Bajjika": "vjk",
        "Bamanankan": "bam",
        "Baoulé": "bci",
        "Borana-Arsi-Guji Oromo": "gax",
        "Bundeli": "bns",
        "Chadian Arabic": "shu",
        "Chhattisgarhi": "hne",
        "Congo Swahili": "swc",
        "Deccan": "dcc",
        "Éwé": "ewe",
        "Javanese": "jav",
        "Hakka Chinese": "hak",
        "Yoruba": "yor",
        "Assamese": "asm",
        "Swahili": "swh",
        "Wolof": "wol",
        "Kinyarwanda": "kin",
        "Uyghur": "uig",
        "Somali": "som",
        "Akan": "aka",
        "Northern Kurdish": "kmr",
        "Indonesian": "ind",
        "Cebuano": "ceb",
        "Marathi": "mar",
        "Hausa": "hau",
        "Min Nan Chinese": "nan",
        "Khmer": "khm",
        "Central Kurdish": "ckb",
        "Gujarati": "guj",
    }

    language_label_map: Dict[str, str] = {name: iso639_map.get(name, name) for name in language_nodes}
    # Render the aggregated bucket on two lines for compactness
    if "Other languages" in language_label_map:
        language_label_map["Other languages"] = "Other\nlanguages"

    draw_alluvial(
        use_to_modality=use_to_modality,
        modality_to_language=modality_to_language,
        use_nodes=use_nodes,
        modality_nodes=modality_nodes,
        language_nodes=language_nodes,
        language_label_map=language_label_map,
        figsize=(args.width, args.height),
        output_path=args.output,
        dpi=args.dpi,
        gap=args.gap,
        brighten=args.brighten,
        neon=args.neon,
        link_alpha=args.link_alpha,
    )


if __name__ == "__main__":
    main()


