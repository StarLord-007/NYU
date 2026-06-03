from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import APIConnectionError, OpenAI, OpenAIError, RateLimitError
from pydantic import BaseModel, Field, ValidationError


class DbRow(BaseModel):
	article_mla: Optional[str] = Field(None)
	authors: Optional[str] = Field(None)
	doi: Optional[str] = Field(None)

	geometry_of_sample: Optional[str] = Field(None)
	dimensions_of_sample: Optional[str] = Field(None)
	material_of_sample: Optional[str] = Field(None)

	oxygen_concentration: Optional[str] = Field(None)
	pressure: Optional[str] = Field(None)
	flow_velocity: Optional[str] = Field(None)
	gravity_gearth: Optional[float] = Field(None)
	experimental_facility: Optional[str] = Field(None)

	ignition_method: Optional[str] = Field(None)
	ignition_power: Optional[str] = Field(None)
	ignition_time: Optional[str] = Field(None)
	chamber_diameter: Optional[str] = Field(None)

	ignition_extinction: Optional[str] = Field(None)
	fsr: Optional[str] = Field(None)
	hrr: Optional[str] = Field(None)
	smoke_aerosols: Optional[str] = Field(None)

	experiment_label: Optional[str] = Field(None)
	notes: Optional[str] = Field(None)


class ExtractionResult(BaseModel):
	rows: List[DbRow]


PLACEHOLDER_VALUES = {
	"",
	"n/a",
	"na",
	"none",
	"null",
	"unknown",
	"not provided",
	"not available",
	"not stated",
	"...",
}

GEOMETRY_CANONICAL = {
	"wire": "wire",
	"cylindrical": "cylindrical",
	"cylinder": "cylindrical",
	"rod": "cylindrical",
	"flat": "flat",
	"sheet": "flat",
	"plate": "flat",
}

FACILITY_CANONICAL = {
	"parabolic": "Parabolic Aircraft",
	"drop tower": "Drop Tower",
	"iss": "Spacecraft",
	"spacecraft": "Spacecraft",
	"rocket": "Sounding Rocket",
}


def load_markdown_context(markdown_path: Path) -> str:
	with markdown_path.open("r", encoding="utf-8") as f:
		return f.read()


def truncate_markdown_context(context_md: str, max_chars: int) -> str:
	full = context_md.strip()
	if len(full) > max_chars:
		full = full[:max_chars] + "\n\n[TRUNCATED]\n"
	return full


def build_prompt_markdown(context_md: str, paper_label: str) -> str:
	return f"""You are helping to extract rows for a microgravity flammability database from the paper:

{paper_label}

The final database has one row per DISTINCT experimental condition (fuel + geometry + ambient
conditions + gravity level + facility + ignition configuration). Each row repeats the citation
fields and then fills as many experimental fields as possible.

IMPORTANT: For this task you MUST ignore any values that are only visible in the PLOTS.
Only use what is explicitly stated in the text and figure captions.

TASK:
1. Identify each DISTINCT experimental condition suitable for a database row.
2. For each row, extract the following fields (use null when the paper does not clearly state it):

Citation metadata (repeat for each row of this paper):
- article_mla
- authors
- doi

Sample:
- geometry_of_sample: one of 'flat', 'wire', or 'cylindrical' when possible.
- dimensions_of_sample
- material_of_sample

Ambient / flow:
- oxygen_concentration
- pressure
- flow_velocity
- gravity_gearth
- experimental_facility: one of ['Parabolic Aircraft', 'Drop Tower', 'Spacecraft', 'Sounding Rocket'] when possible.

Ignition:
- ignition_method
- ignition_power
- ignition_time
- chamber_diameter

Outcomes (ONLY from text, not from plots):
- ignition_extinction
- fsr
- hrr
- smoke_aerosols

Additional:
- experiment_label
- notes

3. Do NOT invent values; only use what is in the text. If unsure, use null.

OUTPUT FORMAT:
Return a JSON object with exactly one key: "rows".

FULL MARKDOWN SOURCE:

```markdown
{context_md}
```"""


def create_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
	key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
	if not key:
		raise RuntimeError("Please set GROQ_API_KEY/OPENAI_API_KEY or pass --api-key.")
	url = base_url or "https://api.groq.com/openai/v1"
	return OpenAI(api_key=key, base_url=url)


def parse_json_like_content(content: str) -> Any:
	text = content.strip()
	if not text:
		raise json.JSONDecodeError("Empty model response", text, 0)

	try:
		return json.loads(text)
	except json.JSONDecodeError:
		pass

	fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
	if fence_match:
		fenced = fence_match.group(1).strip()
		try:
			return json.loads(fenced)
		except json.JSONDecodeError:
			pass

	start = text.find("{")
	end = text.rfind("}")
	if start != -1 and end != -1 and end > start:
		return json.loads(text[start : end + 1])

	raise json.JSONDecodeError("Could not locate JSON object in model response", text, 0)


def _normalize_optional_text(value: Any) -> Optional[str]:
	if value is None:
		return None
	if not isinstance(value, str):
		return str(value)
	cleaned = re.sub(r"\s+", " ", value).strip()
	if cleaned.lower() in PLACEHOLDER_VALUES:
		return None
	return cleaned


def _normalize_yes_no(value: Any) -> Optional[str]:
	text = _normalize_optional_text(value)
	if text is None:
		return None
	lowered = text.lower()
	if lowered in {"yes", "y", "true", "1", "ignited", "extinction"}:
		return "Yes"
	if lowered in {"no", "n", "false", "0", "did not ignite"}:
		return "No"
	return None


def _normalize_geometry(value: Any) -> Optional[str]:
	text = _normalize_optional_text(value)
	if text is None:
		return None
	lowered = text.lower()
	for key, canonical in GEOMETRY_CANONICAL.items():
		if key in lowered:
			return canonical
	return text


def _normalize_facility(value: Any) -> Optional[str]:
	text = _normalize_optional_text(value)
	if text is None:
		return None
	lowered = text.lower()
	for key, canonical in FACILITY_CANONICAL.items():
		if key in lowered:
			return canonical
	if text in {"Parabolic Aircraft", "Drop Tower", "Spacecraft", "Sounding Rocket"}:
		return text
	return None


def _normalize_doi(value: Any) -> Optional[str]:
	text = _normalize_optional_text(value)
	if text is None:
		return None
	if text.startswith("https://doi.org/"):
		return text
	if text.startswith("http://doi.org/"):
		return "https://" + text[len("http://") :]
	if text.lower().startswith("doi:"):
		text = text[4:].strip()
	if re.match(r"^10\.\d{4,9}/\S+$", text):
		return f"https://doi.org/{text}"
	return text


def _normalize_gravity(value: Any) -> Optional[float]:
	if value is None:
		return None
	if isinstance(value, (int, float)):
		return float(value)
	text = _normalize_optional_text(value)
	if text is None:
		return None
	m = re.search(r"-?\d+(?:\.\d+)?", text)
	if not m:
		return None
	try:
		return float(m.group(0))
	except ValueError:
		return None


def _row_dedupe_key(row: DbRow) -> tuple[Any, ...]:
	return (
		row.geometry_of_sample,
		row.dimensions_of_sample,
		row.material_of_sample,
		row.oxygen_concentration,
		row.pressure,
		row.flow_velocity,
		row.gravity_gearth,
		row.experimental_facility,
		row.ignition_method,
		row.ignition_power,
		row.ignition_time,
		row.chamber_diameter,
	)


def sanitize_raw_result_payload(data: Any) -> Any:
	"""
	Coerce common loose model outputs into schema-friendly values before validation.
	"""
	if not isinstance(data, dict):
		return data

	rows = data.get("rows")
	if not isinstance(rows, list):
		return data

	sanitized_rows: List[Any] = []
	for row in rows:
		if not isinstance(row, dict):
			sanitized_rows.append(row)
			continue

		row_copy = dict(row)
		row_copy["gravity_gearth"] = _normalize_gravity(row_copy.get("gravity_gearth"))
		sanitized_rows.append(row_copy)

	out = dict(data)
	out["rows"] = sanitized_rows
	return out


def normalize_extraction_result(result: ExtractionResult) -> ExtractionResult:
	normalized_rows: List[DbRow] = []
	seen_keys: set[tuple[Any, ...]] = set()

	for row in result.rows:
		payload = row.model_dump()
		for key, value in list(payload.items()):
			if key == "gravity_gearth":
				continue
			payload[key] = _normalize_optional_text(value)

		payload["geometry_of_sample"] = _normalize_geometry(payload.get("geometry_of_sample"))
		payload["experimental_facility"] = _normalize_facility(payload.get("experimental_facility"))
		payload["doi"] = _normalize_doi(payload.get("doi"))
		payload["gravity_gearth"] = _normalize_gravity(payload.get("gravity_gearth"))
		payload["ignition_extinction"] = _normalize_yes_no(payload.get("ignition_extinction"))
		payload["smoke_aerosols"] = _normalize_yes_no(payload.get("smoke_aerosols"))

		normalized = DbRow.model_validate(payload)
		key = _row_dedupe_key(normalized)
		if key in seen_keys:
			continue
		seen_keys.add(key)
		normalized_rows.append(normalized)

	return ExtractionResult(rows=normalized_rows)


def call_llm_json(
	client: OpenAI,
	prompt: str,
	model: str,
	max_retries: int = 5,
	initial_delay: float = 1.0,
) -> ExtractionResult:
	delay = initial_delay
	for attempt in range(max_retries):
		try:
			completion = client.chat.completions.create(
				model=model,
				messages=[
					{
						"role": "system",
						"content": (
							"You are an expert in combustion and microgravity "
							"flammability experiments. Always return ONLY valid JSON "
							"matching the requested schema, with no extra commentary."
						),
					},
					{"role": "user", "content": prompt},
				],
				temperature=0.1,
				response_format={"type": "json_object"},
			)

			content = completion.choices[0].message.content
			if not content:
				raise RuntimeError("Model returned empty content.")

			data = parse_json_like_content(content)
			data = sanitize_raw_result_payload(data)
			return ExtractionResult.model_validate(data)

		except (RateLimitError, APIConnectionError) as e:
			if attempt < max_retries - 1:
				print(
					f"[WARN] Transient API error ({type(e).__name__}): {e}. "
					f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...",
					file=sys.stderr,
				)
				time.sleep(delay)
				delay = min(delay * 2, 60.0)
				continue
			raise RuntimeError(f"API request failed after {max_retries} attempts: {e}") from e
		except OpenAIError as e:
			raise RuntimeError(f"API error: {e}") from e
		except json.JSONDecodeError as e:
			raise RuntimeError(f"Failed to parse model output as JSON: {e}") from e
		except ValidationError as e:
			raise RuntimeError(f"Pydantic validation failed on model output: {e}") from e
		except Exception as e:
			raise RuntimeError(f"Unexpected error during LLM call: {e}") from e

	raise RuntimeError("LLM call failed unexpectedly.")


def process_single_markdown_file(
	client: OpenAI,
	markdown_path: Path,
	model: str,
	max_chars: int,
	max_retries: int,
) -> ExtractionResult:
	context_md_raw = load_markdown_context(markdown_path)
	if not context_md_raw.strip():
		raise RuntimeError(f"No context found in markdown: {markdown_path}")

	context_md = truncate_markdown_context(context_md_raw, max_chars=max_chars)
	paper_label = markdown_path.stem
	prompt = build_prompt_markdown(context_md, paper_label=paper_label)

	result = call_llm_json(client, prompt=prompt, model=model, max_retries=max_retries)
	return normalize_extraction_result(result)


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"LLM extraction of database-shaped rows directly from original markdown files.\n\n"
			"Example:\n"
			"  python llm_v2.py src/paper.md --out src/paper.row.json\n"
		),
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument(
		"markdown",
		nargs="+",
		help="One or more paths or glob patterns to source *.md files.",
	)
	parser.add_argument(
		"--out",
		type=str,
		default=None,
		help=(
			"Output JSON path. If multiple markdown files are provided, --out is required "
			"and a combined dict is written."
		),
	)
	parser.add_argument(
		"--model",
		type=str,
		default="llama-3.3-70b-versatile",
		help="Model name (default: llama-3.3-70b-versatile for Groq).",
	)
	parser.add_argument(
		"--api-key",
		type=str,
		default=None,
		help="API key (otherwise uses GROQ_API_KEY or OPENAI_API_KEY).",
	)
	parser.add_argument(
		"--base-url",
		type=str,
		default=None,
		help="OpenAI-compatible base URL (default: https://api.groq.com/openai/v1).",
	)
	parser.add_argument(
		"--max-chars",
		type=int,
		default=30000,
		help="Max markdown characters sent to the model.",
	)
	parser.add_argument(
		"--max-retries",
		type=int,
		default=5,
		help="Max retries for transient API errors.",
	)

	args = parser.parse_args()

	markdown_files: List[Path] = []
	for pattern in args.markdown:
		matches = list(Path(".").glob(pattern))
		if not matches:
			print(f"[WARN] No files matched pattern: {pattern}", file=sys.stderr)
		else:
			markdown_files.extend(matches)

	if not markdown_files:
		raise SystemExit("No markdown files found for the given patterns.")

	client = create_client(api_key=args.api_key, base_url=args.base_url)
	all_results: Dict[str, Any] = {}

	for markdown_path in sorted(markdown_files):
		if not markdown_path.is_file():
			print(f"[WARN] Skipping non-file: {markdown_path}", file=sys.stderr)
			continue
		if markdown_path.suffix.lower() != ".md":
			print(f"[WARN] Skipping non-markdown file: {markdown_path}", file=sys.stderr)
			continue

		print(f"[INFO] Processing {markdown_path} ...", file=sys.stderr)
		try:
			result = process_single_markdown_file(
				client=client,
				markdown_path=markdown_path,
				model=args.model,
				max_chars=args.max_chars,
				max_retries=args.max_retries,
			)
		except Exception as exc:
			print(f"[ERROR] Failed on {markdown_path}: {exc}", file=sys.stderr)
			continue

		all_results[str(markdown_path)] = result.model_dump()

	if not all_results:
		raise SystemExit("No successful extractions; exiting.")

	if args.out:
		out_path = Path(args.out)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		with out_path.open("w", encoding="utf-8") as f:
			json.dump(all_results, f, ensure_ascii=False, indent=2)
		print(f"[INFO] Wrote extraction results to {out_path}", file=sys.stderr)
	else:
		if len(all_results) == 1:
			single_input = Path(next(iter(all_results.keys())))
			out_path = single_input.with_suffix(".row.json")
			out_path.parent.mkdir(parents=True, exist_ok=True)
			with out_path.open("w", encoding="utf-8") as f:
				json.dump(all_results, f, ensure_ascii=False, indent=2)
			print(f"[INFO] Wrote extraction results to {out_path}", file=sys.stderr)
		else:
			json.dump(all_results, sys.stdout, ensure_ascii=False, indent=2)
			print()


if __name__ == "__main__":
	main()
