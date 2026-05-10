#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


SEMANTIC_SCHOLAR_BATCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/batch?fields=year,publicationVenue,journal,citationCount"
)


@dataclass(frozen=True)
class CitedItem:
    source_row_id: str
    raw_value: str
    normalized_id: str  # e.g., DOI:10.1000/xyz, URL:https://... or S2PaperId:...


def _strip_year_suffix(token: str) -> Tuple[str, Optional[str]]:
    """Remove trailing ' (YYYY)' and return (clean, year_if_present)."""
    m = re.search(r"\s*\((\d{4})\)\s*$", token)
    if m:
        year = m.group(1)
        return token[: m.start()].strip(), year
    return token.strip(), None


def _normalize_identifier_from_url(url_or_id: str) -> Optional[str]:
    s = url_or_id.strip()
    if not s:
        return None

    # If it's a pure DOI URL
    m = re.match(r"^(?:https?://)?doi\.org/(.+)$", s, flags=re.IGNORECASE)
    if m:
        doi = m.group(1).strip()
        # Avoid enclosing parentheses or trailing punctuation
        doi = doi.strip().strip(".,)")
        if doi:
            return f"DOI:{doi}"

    # If it's a Semantic Scholar URL or any other URL -> use URL: lookup
    if re.match(r"^https?://", s, flags=re.IGNORECASE):
        return f"URL:{s}"

    # If it looks like a bare DOI
    if re.match(r"^10\.\d{4,9}/\S+$", s, flags=re.IGNORECASE):
        return f"DOI:{s}"

    # If it looks like a corpus/paper id we can try passing through as-is
    if re.match(r"^(S2PaperId|CorpusId|ArXiv|PubMed|ACL):", s, flags=re.IGNORECASE):
        return s

    return None


def iter_cited_items(tsv_path: str) -> Iterable[CitedItem]:
    with open(tsv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header: List[str] = next(reader)
        # Identify columns
        try:
            id_idx = header.index("ID")
        except ValueError:
            raise RuntimeError("Missing 'ID' column in TSV header")
        try:
            cited_idx = header.index("Citied Article")
        except ValueError:
            raise RuntimeError("Missing 'Citied Article' column in TSV header")

        for row in reader:
            # Skip malformed rows
            if len(row) <= max(id_idx, cited_idx):
                continue
            row_id = row[id_idx].strip()
            cited_raw = (row[cited_idx] or "").strip()
            if not cited_raw:
                continue

            # split on commas; entries appear as comma-separated URLs with optional trailing (YYYY)
            parts = [p.strip() for p in cited_raw.split(",") if p.strip()]
            if not parts:
                continue
            # Only take the first cited article if multiple are present
            first = parts[0]
            clean, _maybe_year = _strip_year_suffix(first)
            norm = _normalize_identifier_from_url(clean)
            if norm is None:
                continue
            yield CitedItem(source_row_id=row_id, raw_value=clean, normalized_id=norm)


def batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def _http_post_json(url: str, payload: dict, api_key: Optional[str], timeout: float = 30.0) -> Tuple[int, dict, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("x-api-key", api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            status = resp.getcode()
            headers = dict(resp.headers.items())
            return status, json.loads(body.decode("utf-8")), headers
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            parsed = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            parsed = {"error": body.decode("utf-8", errors="replace")}
        headers = dict(e.headers.items()) if e.headers else {}
        return e.code, parsed, headers
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e}")


def fetch_metadata_for_ids(unique_ids: List[str], api_key: Optional[str], max_retries: int = 5) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    backoff = 1.0
    for chunk in batched(unique_ids, 100):
        retries = 0
        while True:
            status, body, headers = _http_post_json(
                SEMANTIC_SCHOLAR_BATCH_URL, {"ids": chunk}, api_key
            )
            if status == 200 and isinstance(body, list):
                # Response is a list aligned to input chunk
                for idx, entry in enumerate(body):
                    # entry may be an error object or dict with requested fields
                    key = chunk[idx]
                    results[key] = entry
                break

            # Handle rate limit or server errors
            if status in (429, 500, 502, 503, 504):
                retries += 1
                if retries > max_retries:
                    # Record failures for this chunk and continue
                    for key in chunk:
                        results[key] = {"_error": f"HTTP {status}", "_body": body}
                    break
                retry_after = headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else backoff
                except ValueError:
                    delay = backoff
                time.sleep(delay)
                backoff = min(backoff * 2, 60.0)
                continue

            # Other non-success
            for key in chunk:
                results[key] = {"_error": f"HTTP {status}", "_body": body}
            break

    return results


def flatten_field(entry: Optional[dict], path: List[str]) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    cur = entry
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if cur is None:
        return None
    return str(cur)


def write_output(
    out_path: str,
    items: List[CitedItem],
    id_to_entry: Dict[str, dict],
) -> None:
    # We write per item (not just unique id), so that original row mapping is preserved.
    # Columns: ID, Citied Article, normalized_id, year, publicationVenue, journal, citationCount
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "ID",
                "Citied Article",
                "normalized_id",
                "s2_paperId",
                "year",
                "publicationVenue",
                "journal",
                "citationCount",
                "_error",
            ]
        )
        for it in items:
            entry = id_to_entry.get(it.normalized_id)
            s2_id = None
            year = None
            pub_venue = None
            journal = None
            citation_count = None
            err = None

            if isinstance(entry, dict):
                if "_error" in entry:
                    err = entry.get("_error")
                else:
                    s2_id = entry.get("paperId")
                    # Fields per API contract
                    year = entry.get("year")
                    citation_count = entry.get("citationCount")
                    pub_venue = flatten_field(entry, ["publicationVenue", "name"]) or entry.get("venue")
                    journal = flatten_field(entry, ["journal", "name"])
            else:
                err = "no_entry"

            # Prefer Semantic Scholar paperId in normalized_id output when available
            normalized_id_out = s2_id if s2_id else it.normalized_id

            writer.writerow(
                [
                    it.source_row_id,
                    it.raw_value,
                    normalized_id_out,
                    s2_id if s2_id is not None else "",
                    year if year is not None else "",
                    pub_venue if pub_venue is not None else "",
                    journal if journal is not None else "",
                    citation_count if citation_count is not None else "",
                    err if err is not None else "",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Semantic Scholar metadata for cited articles in TSV")
    parser.add_argument("--input", required=True, help="Path to input TSV")
    parser.add_argument("--output", required=True, help="Path to output TSV")
    parser.add_argument(
        "--dry-run",
        type=int,
        default=0,
        help="If > 0, limit to the first N unique IDs and print a sample of the API response",
    )
    args = parser.parse_args()

    api_key = os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

    # Collect items
    items: List[CitedItem] = list(iter_cited_items(args.input))
    if not items:
        print("No cited items found.", file=sys.stderr)
        sys.exit(1)

    # Unique id order preservation
    seen: Dict[str, None] = {}
    unique_ids: List[str] = []
    for it in items:
        if it.normalized_id not in seen:
            seen[it.normalized_id] = None
            unique_ids.append(it.normalized_id)

    if args.dry_run and args.dry_run > 0:
        unique_ids = unique_ids[: args.dry_run]
        # Filter items to only those in the dry-run set to keep mapping consistent
        items = [it for it in items if it.normalized_id in set(unique_ids)]

    id_to_entry = fetch_metadata_for_ids(unique_ids, api_key)

    # If dry-run, print a small sample of entries to stdout for verification
    if args.dry_run and args.dry_run > 0:
        sample_keys = unique_ids[: min(3, len(unique_ids))]
        sample = {k: id_to_entry.get(k) for k in sample_keys}
        print(json.dumps(sample, indent=2, ensure_ascii=False))

    # Write output TSV
    write_output(args.output, items, id_to_entry)


if __name__ == "__main__":
    main()


