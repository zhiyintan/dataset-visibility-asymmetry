#!/usr/bin/env python3
import asyncio
import aiohttp
import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, Optional


def build_system_prompt() -> str:
	return (
		"You are a data curation assistant producing LREMap-style metadata. "
		"Given a short description of how a dataset/resource is used (from a 'Features' column), "
		"infer two fields with concise values: 'modality' and 'use'.\n\n"
		"Rules:\n"
		"- 'modality': one short label describing the input modality. Examples: 'Written', 'Speech', 'Multimodal/Multimedia'.\n"
		"- 'use': one short label for primary use. Examples: 'Parsing and Tagging', 'Machine Translation', 'Question Answering', 'Sentiment Analysis', 'Information Retrieval', 'Speech Recognition/Understanding', 'Lexicon/Dictionary'.\n"
		"- If unclear or not applicable, set the field to null.\n"
		"- Output strictly a compact JSON object with exactly keys: modality, use."
	)


def build_user_prompt(features_text: str) -> str:
	return (
		"From the following Features text, infer 'modality' and 'use' in the LREMap style.\n\n"
		f"Features:\n{features_text.strip()}\n\n"
		"Return ONLY JSON with keys 'modality' and 'use'."
	)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
	# Try strict parse first
	try:
		obj = json.loads(text)
		if isinstance(obj, dict):
			return obj
	except Exception:
		pass
	# Fallback: find first balanced {...}
	start = text.find("{")
	if start == -1:
		return None
	depth = 0
	for i in range(start, len(text)):
		c = text[i]
		if c == "{":
			depth += 1
		elif c == "}":
			depth -= 1
			if depth == 0:
				candidate = text[start : i + 1]
				try:
					obj = json.loads(candidate)
					if isinstance(obj, dict):
						return obj
				except Exception:
					return None
	return None


async def classify_features(
	session: aiohttp.ClientSession,
	base_url: str,
	model: str,
	features_text: str,
	timeout_s: float = 60.0,
) -> Dict[str, Any]:
	if not features_text or not features_text.strip():
		return {"modality": None, "use": None}

	body = {
		"model": model,
		"messages": [
			{"role": "system", "content": build_system_prompt()},
			{"role": "user", "content": build_user_prompt(features_text)},
		],
		"temperature": 0,
		"max_tokens": 64,
	}

	url = base_url.rstrip("/") + "/chat/completions"
	for attempt in range(3):
		try:
			async with session.post(url, json=body, timeout=timeout_s) as resp:
				if resp.status != 200:
					txt = await resp.text()
					raise RuntimeError(f"Bad status {resp.status}: {txt[:200]}")
				data = await resp.json()
				choice = (data.get("choices") or [{}])[0]
				message = (choice.get("message") or {})
				content = message.get("content") or ""
				obj = extract_json_object(content)
				if not obj:
					return {"modality": None, "use": None}
				modality = obj.get("modality")
				use = obj.get("use")
				# Normalize to short strings or None
				modality = modality if isinstance(modality, str) and modality.strip() else None
				use = use if isinstance(use, str) and use.strip() else None
				return {"modality": modality, "use": use}
		except Exception:
			if attempt == 2:
				return {"modality": None, "use": None}
			await asyncio.sleep(0.5 * (attempt + 1))


async def worker(
	name: str,
	sem: asyncio.Semaphore,
	session: aiohttp.ClientSession,
	base_url: str,
	model: str,
	row: Dict[str, str],
	features_col: str,
	writer_queue: "asyncio.Queue[Dict[str, Any]]",
	idx: int,
) -> None:
	features_text = row.get(features_col, "")
	result = await classify_features(session, base_url, model, features_text)
	full_row = dict(row)
	full_row["modality"] = result.get("modality") or ""
	full_row["use"] = result.get("use") or ""
	await writer_queue.put({"_idx": idx, "_row": full_row})


async def writer_task(path: str, headers: list[str], queue: "asyncio.Queue[Dict[str, Any]]", total: int) -> None:
	written = 0
	next_idx = 0
	buffer: Dict[int, Dict[str, Any]] = {}
	with open(path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
		writer.writeheader()
		while written < total:
			item = await queue.get()
			idx = int(item["_idx"])  # position in input
			row = item["_row"]
			buffer[idx] = row
			while next_idx in buffer:
				writer.writerow(buffer.pop(next_idx))
				next_idx += 1
				written += 1


async def run_async(
	input_path: str,
	output_path: str,
	base_url: str,
	model: str,
	concurrency: int,
	max_rows: Optional[int],
) -> None:
	rows: list[Dict[str, str]] = []
	headers: list[str] = []
	with open(input_path, "r", encoding="utf-8") as rf:
		reader = csv.DictReader(rf, delimiter="\t")
		if "Features" not in (reader.fieldnames or []):
			raise SystemExit(f"Features column not found in {input_path}")
		headers = list(reader.fieldnames or [])
		if "modality" not in headers:
			headers.append("modality")
		if "use" not in headers:
			headers.append("use")
		for i, row in enumerate(reader):
			rows.append(row)
			if max_rows is not None and len(rows) >= max_rows:
				break

	if not rows:
		# Create empty file
		open(output_path, "w", encoding="utf-8").close()
		return

	conn = aiohttp.TCPConnector(limit=concurrency)
	queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
	sem = asyncio.Semaphore(concurrency)
	async with aiohttp.ClientSession(connector=conn, headers={
		"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
		"Content-Type": "application/json",
	}) as session:
		writer = asyncio.create_task(writer_task(output_path, headers, queue, total=len(rows)))
		tasks = []
		for idx, row in enumerate(rows):
			# Small gate to keep overall concurrency bounded
			async def run_one(r=row, i=idx):
				async with sem:
					await worker(
						name=f"job-{r.get('ID', str(id(r)))}",
						sem=sem,
						session=session,
						base_url=base_url,
						model=model,
						row=r,
						features_col="Features",
						writer_queue=queue,
						idx=i,
					)
			tasks.append(asyncio.create_task(run_one()))
			if (idx + 1) % (concurrency * 4) == 0:
				await asyncio.sleep(0)  # yield

		await asyncio.gather(*tasks)
		await writer


def main(argv: Optional[list[str]] = None) -> None:
	parser = argparse.ArgumentParser(description="Extract modality/use via local vLLM from Features column")
	repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	parser.add_argument(
		"--input",
		type=str,
		default=os.path.join(repo_root, "tables", "checked.tsv"),
		help="Path to input TSV with a Features column (default: tables/checked.tsv)",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=os.path.join(repo_root, "tables", "checked_modality_use.tsv"),
		help="Path to output TSV (default: tables/checked_modality_use.tsv)",
	)
	parser.add_argument(
		"--base-url",
		type=str,
		default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
		help="OpenAI-compatible base URL of vLLM server",
	)
	parser.add_argument(
		"--model",
		type=str,
		default=os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
		help="Model name served by vLLM",
	)
	# Determine default concurrency (env override takes precedence)
	default_concurrency_str = os.environ.get("VLLM_CONCURRENCY")
	if default_concurrency_str and default_concurrency_str.isdigit():
		default_concurrency = max(1, int(default_concurrency_str))
	else:
		default_concurrency = 64
	parser.add_argument(
		"--concurrency",
		type=int,
		default=default_concurrency,
		help="Number of concurrent requests",
	)
	parser.add_argument(
		"--max-rows",
		type=int,
		default=None,
		help="Process only the first N rows (for quick tests)",
	)
	args = parser.parse_args(argv)

	# Ensure absolute paths
	input_path = os.path.abspath(args.input)
	output_path = os.path.abspath(args.output)

	try:
		asyncio.run(
			run_async(
				input_path=input_path,
				output_path=output_path,
				base_url=args.base_url,
				model=args.model,
				concurrency=int(args.concurrency),
				max_rows=args.max_rows,
			)
		)
	except KeyboardInterrupt:
		print("Interrupted", file=sys.stderr)
		sys.exit(130)


if __name__ == "__main__":
	main()


