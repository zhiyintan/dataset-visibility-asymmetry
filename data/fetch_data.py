import asyncio
import argparse
import math
import re
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import aiohttp
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

BASE_URL = "https://lremap.elra.info/"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "lremap" / "raw"
DEFAULT_MAX_RETRIES = 4
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_CONCURRENCY_LIMIT = 6
DEFAULT_PER_PAGE = 10


def safe_dirname(value: str) -> str:
    """Convert a language string to a filesystem-safe name while preserving case and '+' sign.

    Keeps letters, digits, spaces, '+', '-', '_', and parentheses. Replaces others with '_'.
    """
    value = value.strip()
    # Replace disallowed chars with underscore, keep case and '+'
    return re.sub(r"[^A-Za-z0-9 \+_\-\(\)]", "_", value)


async def fetch_page_html(
    session: aiohttp.ClientSession,
    page: int,
    language: str,
    request_timeout_seconds: int,
) -> str:
    params = {"page": str(page), "languages": language}
    async with session.get(
        BASE_URL,
        params=params,
        timeout=aiohttp.ClientTimeout(total=request_timeout_seconds),
    ) as response:
        response.raise_for_status()
        return await response.text()


async def fetch_with_retries(
    session: aiohttp.ClientSession,
    page: int,
    out_dir: Path,
    language: str,
    max_retries: int,
    request_timeout_seconds: int,
    resume: bool,
) -> Optional[Path]:
    out_file = out_dir / f"page_{page}.html"
    if resume and out_file.exists() and out_file.stat().st_size > 0:
        print(f"Skip existing {out_file}")
        return out_file
    backoff_seconds = 1.5
    for attempt in range(1, max_retries + 1):
        try:
            html = await fetch_page_html(
                session,
                page,
                language,
                request_timeout_seconds,
            )
            tmp_file = out_dir / f".page_{page}.html.tmp"
            tmp_file.write_text(html, encoding="utf-8")
            tmp_file.replace(out_file)
            print(f"Saved {out_file}")
            return out_file
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt >= max_retries:
                print(f"Failed page {page} after {attempt} attempts: {exc}")
                return None
            await asyncio.sleep(backoff_seconds)
            backoff_seconds *= 2


class Progress:
    def __init__(self, total: int, desc: str) -> None:
        self.total = total
        self.desc = desc
        self.completed = 0
        self._bar = tqdm(total=total, desc=desc, unit="page") if tqdm else None
        if self._bar is None:
            print(f"{desc}: 0/{total}")

    def update(self, n: int = 1) -> None:
        self.completed += n
        if self._bar is not None:
            self._bar.update(n)
        else:
            # Print occasionally in fallback mode
            if self.completed == self.total or self.completed % 10 == 0:
                print(f"{self.desc}: {self.completed}/{self.total}")

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


async def run_single(
    language: str,
    start_page: int,
    end_page: int,
    base_output_dir: Path,
    concurrency_limit: int,
    max_retries: int,
    request_timeout_seconds: int,
    resume: bool,
) -> None:
    language_dir = base_output_dir / safe_dirname(language)
    language_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; lremap-async-fetch/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    connector = aiohttp.TCPConnector(limit=concurrency_limit, ssl=False)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        pages = list(range(start_page, end_page + 1))
        if resume:
            pages = [
                p
                for p in pages
                if not ((language_dir / f"page_{p}.html").exists() and (language_dir / f"page_{p}.html").stat().st_size > 0)
            ]
        if not pages:
            print("Nothing to do (all pages present).")
            return
        tasks = [
            fetch_with_retries(
                session,
                p,
                language_dir,
                language,
                max_retries,
                request_timeout_seconds,
                resume,
            )
            for p in pages
        ]
        progress = Progress(total=len(tasks), desc=f"{language}")
        for fut in asyncio.as_completed(tasks):
            await fut
            progress.update()
        progress.close()


def parse_languages_counts(file_path: Path) -> List[Tuple[str, int]]:
    pattern = re.compile(r"^\s*(.+?)\s*\((\d+)\)\s*$")
    results: List[Tuple[str, int]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            # Skip lines that don't match "Name (count)"
            continue
        name = m.group(1)
        count = int(m.group(2))
        results.append((name, count))
    return results


def compute_total_pages(total_count: int, per_page: int) -> int:
    if per_page <= 0:
        return 0
    return max(1, math.ceil(total_count / per_page))


async def run_multi(
    languages_counts: Sequence[Tuple[str, int]],
    base_output_dir: Path,
    concurrency_limit: int,
    max_retries: int,
    request_timeout_seconds: int,
    per_page: int,
    max_pages_per_language: Optional[int] = None,
    resume: bool = False,
) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; lremap-async-fetch/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    connector = aiohttp.TCPConnector(limit=concurrency_limit, ssl=False)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks: List[asyncio.Task] = []
        seen_by_name: dict[str, int] = {}
        for language, count in languages_counts:
            # Determine query value and directory name, with special case for second 'English' => '+English'
            seen = seen_by_name.get(language, 0)
            if language == "English" and seen >= 1:
                query_language = "+English"
                dir_name = "+English"
            else:
                query_language = language
                dir_name = language
            seen_by_name[language] = seen + 1

            language_dir = base_output_dir / safe_dirname(dir_name)
            language_dir.mkdir(parents=True, exist_ok=True)
            total_pages = compute_total_pages(count, per_page)
            if max_pages_per_language is not None:
                total_pages = min(total_pages, max_pages_per_language)
            for p in range(1, total_pages + 1):
                # Skip pre-existing when resuming for accurate totals
                if resume:
                    out_file = language_dir / f"page_{p}.html"
                    if out_file.exists() and out_file.stat().st_size > 0:
                        continue
                tasks.append(
                    asyncio.create_task(
                        fetch_with_retries(
                            session,
                            p,
                            language_dir,
                            query_language,
                            max_retries,
                            request_timeout_seconds,
                            resume,
                        )
                    )
                )
        if not tasks:
            print("Nothing to do (all pages present).")
            return
        progress = Progress(total=len(tasks), desc="All languages")
        for fut in asyncio.as_completed(tasks):
            await fut
            progress.update()
        progress.close()


def is_page_present(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def count_missing_single(language: str, start_page: int, end_page: int, base_output_dir: Path) -> int:
    language_dir = base_output_dir / safe_dirname(language)
    missing = 0
    for p in range(start_page, end_page + 1):
        if not is_page_present(language_dir / f"page_{p}.html"):
            missing += 1
    return missing


def count_missing_multi(
    languages_counts: Sequence[Tuple[str, int]],
    base_output_dir: Path,
    per_page: int,
    max_pages_per_language: Optional[int] = None,
) -> int:
    missing = 0
    seen_by_name: dict[str, int] = {}
    for language, count in languages_counts:
        seen = seen_by_name.get(language, 0)
        if language == "English" and seen >= 1:
            dir_name = "+English"
        else:
            dir_name = language
        seen_by_name[language] = seen + 1

        language_dir = base_output_dir / safe_dirname(dir_name)
        total_pages = compute_total_pages(count, per_page)
        if max_pages_per_language is not None:
            total_pages = min(total_pages, max_pages_per_language)
        for p in range(1, total_pages + 1):
            if not is_page_present(language_dir / f"page_{p}.html"):
                missing += 1
    return missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch LREMap search result pages concurrently.")
    parser.add_argument(
        "--language",
        type=str,
        default="Indonesian",
        help="Language filter as it appears on the site (e.g., 'Mandarin Chinese').",
    )
    parser.add_argument("--start", type=int, default=1, help="Start page (inclusive).")
    parser.add_argument("--end", type=int, default=3, help="End page (inclusive).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory; a language subfolder will be created inside.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY_LIMIT,
        help="Max concurrent requests.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Max retries per page.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--languages-file",
        type=Path,
        help="Path to a file with lines like: Name (count). If set, fetch all.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=DEFAULT_PER_PAGE,
        help="Items per page used to compute total pages for each language.",
    )
    parser.add_argument(
        "--max-pages-per-lang",
        type=int,
        help="Optional cap on pages per language (useful for testing).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-downloaded pages and continue",
    )
    parser.add_argument(
        "--until-complete",
        action="store_true",
        help="After a run, sleep and rerun until no pages are missing",
    )
    parser.add_argument(
        "--loop-sleep-seconds",
        type=int,
        default=120,
        help="Seconds to sleep between iterations when --until-complete is set",
    )

    args = parser.parse_args()
    if args.languages_file:
        langs = parse_languages_counts(args.languages_file)
        if args.until_complete:
            while True:
                asyncio.run(
                    run_multi(
                        languages_counts=langs,
                        base_output_dir=args.out_dir,
                        concurrency_limit=args.concurrency,
                        max_retries=args.retries,
                        request_timeout_seconds=args.timeout,
                        per_page=args.per_page,
                        max_pages_per_language=args.max_pages_per_lang,
                        resume=True,
                    )
                )
                remaining = count_missing_multi(
                    languages_counts=langs,
                    base_output_dir=args.out_dir,
                    per_page=args.per_page,
                    max_pages_per_language=args.max_pages_per_lang,
                )
                if remaining <= 0:
                    print("All pages fetched for all languages.")
                    break
                print(f"{remaining} pages still missing. Sleeping {args.loop_sleep_seconds}s before retrying...")
                time.sleep(args.loop_sleep_seconds)
        else:
            asyncio.run(
                run_multi(
                    languages_counts=langs,
                    base_output_dir=args.out_dir,
                    concurrency_limit=args.concurrency,
                    max_retries=args.retries,
                    request_timeout_seconds=args.timeout,
                    per_page=args.per_page,
                    max_pages_per_language=args.max_pages_per_lang,
                    resume=args.resume,
                )
            )
    else:
        if args.until_complete:
            while True:
                asyncio.run(
                    run_single(
                        language=args.language,
                        start_page=args.start,
                        end_page=args.end,
                        base_output_dir=args.out_dir,
                        concurrency_limit=args.concurrency,
                        max_retries=args.retries,
                        request_timeout_seconds=args.timeout,
                        resume=True,
                    )
                )
                remaining = count_missing_single(
                    language=args.language,
                    start_page=args.start,
                    end_page=args.end,
                    base_output_dir=args.out_dir,
                )
                if remaining <= 0:
                    print("All pages fetched for the language.")
                    break
                print(f"{remaining} pages still missing. Sleeping {args.loop_sleep_seconds}s before retrying...")
                time.sleep(args.loop_sleep_seconds)
        else:
            asyncio.run(
                run_single(
                    language=args.language,
                    start_page=args.start,
                    end_page=args.end,
                    base_output_dir=args.out_dir,
                    concurrency_limit=args.concurrency,
                    max_retries=args.retries,
                    request_timeout_seconds=args.timeout,
                    resume=args.resume,
                )
            )
