import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from bs4 import BeautifulSoup  # type: ignore
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore
from concurrent.futures import ProcessPoolExecutor


@dataclass
class ResultMetadata:
    language_dir: str
    source_file: str
    page: Optional[int]
    res_id: Optional[int]
    name: Optional[str]
    url: Optional[str]
    modality: Optional[str]
    resource_type: Optional[str]
    language_type: Optional[str]
    languages_text: Optional[str]
    availability: Optional[str]
    license: Optional[str]
    conference: Optional[str]
    size: Optional[str]
    production_status: Optional[str]
    use: Optional[str]
    paper_title: Optional[str]
    paper_track: Optional[str]
    paper_status: Optional[str]


def normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = " ".join(value.split())
    return value if value else None


def get_text_excluding_label(container, label_prefix: str) -> Optional[str]:
    if container is None:
        return None
    # Collect all text parts except the label
    parts: List[str] = []
    for s in container.stripped_strings:
        if s.strip().startswith(label_prefix):
            # Skip the label portion
            continue
        parts.append(s)
    return normalize_text(" ".join(parts))


def parse_result_div(result_div, language_dir: str, source_file: Path) -> ResultMetadata:
    # Basic fields
    res_id_text = None
    res_id_input = result_div.find("input", class_="res-id")
    if res_id_input is not None:
        res_id_text = res_id_input.get("value")
    try:
        res_id: Optional[int] = int(res_id_text) if res_id_text is not None else None
    except ValueError:
        res_id = None

    name = None
    name_el = result_div.find("span", class_="res-name")
    if name_el is not None:
        name = normalize_text(name_el.get_text(" "))

    # External URL (icon link)
    url = None
    link_icon = result_div.find("img", class_="link")
    if link_icon is not None and link_icon.parent and link_icon.parent.name == "a":
        url = link_icon.parent.get("href")
        if url:
            url = url.strip()

    modality = None
    modality_el = result_div.find("span", class_="modality")
    if modality_el is not None:
        modality = normalize_text(modality_el.get_text(" "))

    resource_type = None
    type_el = result_div.find("span", class_="res-type")
    if type_el is not None:
        resource_type = normalize_text(type_el.get_text(" ").rstrip(","))

    # Language Type and Languages
    language_type = None
    languages_text = None
    for lt in result_div.find_all("div", class_="lang-type"):
        label_el = lt.find("div", class_="divLabel")
        label = normalize_text(label_el.get_text(" ")) if label_el is not None else None
        text_value = None
        # Extract text of the lang-type div excluding the label
        text_value = get_text_excluding_label(lt, "Language Type:") if label == "Language Type:" else (
            get_text_excluding_label(lt, "Languages:") if label == "Languages:" else None
        )
        if label == "Language Type:" and text_value:
            language_type = text_value
        elif label == "Languages:" and text_value:
            languages_text = text_value

    availability = None
    avail_el = result_div.find("div", class_="avail")
    if avail_el is not None:
        availability = normalize_text(avail_el.get_text(" "))

    license_text = None
    license_container = result_div.find("div", class_="license")
    license_text = get_text_excluding_label(license_container, "License:")

    # Conference info
    conference = None
    conf_el = result_div.find("span", class_="conf")
    if conf_el is not None:
        conference = normalize_text(conf_el.get_text(" "))

    # Hidden details
    size = None
    size_el = result_div.find("div", class_="size")
    size = get_text_excluding_label(size_el, "Size:")

    production_status = None
    prod_el = result_div.find("span", class_="prod-status")
    if prod_el is not None:
        production_status = normalize_text(prod_el.get_text(" "))

    use_text = None
    use_el = result_div.find("span", class_="res-use")
    if use_el is not None:
        use_text = normalize_text(use_el.get_text(" "))

    documentation = None
    doc_el = result_div.find("div", class_="use", attrs={"class": "use readmore"})
    # The above may not match due to how BeautifulSoup handles class_ matching; use CSS selector
    if documentation is None:
        doc_container = result_div.select_one("div.use.readmore")
        documentation = get_text_excluding_label(doc_container, "Documentation:") if doc_container else None

    paper_title = None
    pt_el = result_div.find("span", class_="paper-title")
    if pt_el is not None:
        paper_title = normalize_text(pt_el.get_text(" "))

    paper_track = None
    # Paper track follows a divLabel 'Paper track:'; often next span
    track_label = result_div.find(lambda tag: tag.name == "div" and tag.get("class") == ["divLabel"] and tag.get_text(strip=True) == "Paper track:")
    if track_label is not None:
        sibling_span = track_label.find_next("span")
        if sibling_span is not None:
            paper_track = normalize_text(sibling_span.get_text(" "))

    paper_status = None
    status_label = result_div.find(lambda tag: tag.name == "div" and tag.get("class") == ["divLabel"] and tag.get_text(strip=True) == "Paper status:")
    if status_label is not None:
        sibling_span2 = status_label.find_next("span")
        if sibling_span2 is not None:
            paper_status = normalize_text(sibling_span2.get_text(" "))

    # Attempt to infer page number from filename
    page_num: Optional[int] = None
    try:
        if source_file.stem.startswith("page_"):
            page_num = int(source_file.stem.split("_", 1)[1])
    except Exception:
        page_num = None

    return ResultMetadata(
        language_dir=language_dir,
        source_file=str(source_file),
        page=page_num,
        res_id=res_id,
        name=name,
        url=url,
        modality=modality,
        resource_type=resource_type,
        language_type=language_type,
        languages_text=languages_text,
        availability=availability,
        license=license_text,
        conference=conference,
        size=size,
        production_status=production_status,
        use=use_text,
        paper_title=paper_title,
        paper_track=paper_track,
        paper_status=paper_status,
    )


def iter_html_files(input_dir: Path) -> Iterable[Path]:
    # Look for .../raw/**/page_*.html
    yield from input_dir.rglob("page_*.html")


def parse_file(path: Path) -> List[ResultMetadata]:
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    results: List[ResultMetadata] = []
    for res_div in soup.select("div.result"):
        language_dir = path.parent.name
        results.append(parse_result_div(res_div, language_dir=language_dir, source_file=path))
    return results


def write_jsonl(output_path: Path, items: Iterable[ResultMetadata]) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            # Convert dataclass to dict and ensure ascii disabled for UTF-8 output
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def parse_file_to_dicts(path_str: str) -> List[dict]:
    path = Path(path_str)
    items = parse_file(path)
    return [asdict(item) for item in items]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse LREMap HTML results into JSONL metadata.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "lremap" / "raw",
        help="Root directory containing language subfolders with page_*.html",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "lremap" / "metadata.jsonl",
        help="Path to write JSONL output",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of processes for parallel parsing (>=1)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Chunksize for process pool map",
    )
    args = parser.parse_args()

    files = sorted(iter_html_files(args.input_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_items = 0
    bar = tqdm(total=len(files), desc="Parsing files", unit="file") if tqdm is not None else None
    try:
        with args.output.open("w", encoding="utf-8") as f:
            if args.workers and args.workers > 1 and len(files) > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    for dict_list in ex.map(parse_file_to_dicts, [str(p) for p in files], chunksize=args.chunksize):
                        for d in dict_list:
                            f.write(json.dumps(d, ensure_ascii=False) + "\n")
                            total_items += 1
                        if bar is not None:
                            bar.update(1)
            else:
                for p in files:
                    items = parse_file(p)
                    for item in items:
                        f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
                        total_items += 1
                    if bar is not None:
                        bar.update(1)
    finally:
        if bar is not None:
            bar.close()

    print(f"Wrote {total_items} items to {args.output}")


if __name__ == "__main__":
    main()


