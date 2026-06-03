from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import OpenAI, OpenAIError, RateLimitError, APIConnectionError
from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------

class DbRow(BaseModel):
    # Citation metadata (repeat per row for that paper)
    article_mla: Optional[str] = Field(
        None,
        description=(
            "Full reference similar to: \"Konno, Yusuke, et al. 'Experimental study on downward/opposed "
            "flame spread and extinction over electric wires in partial gravity environments.' Proceedings "
            "of the Combustion Institute 39.3 (2023): 3785-3794.\""
        )
    )
    authors: Optional[str] = Field(
        None,
        description="Comma-separated list of authors, matching the paper."
    )
    doi: Optional[str] = Field(
        None,
        description="DOI URL, e.g. 'https://doi.org/10.1016/j.proci.2018.06.022'."
    )

    # Sample description
    geometry_of_sample: Optional[str] = Field(
        None,
        description="Geometry of sample: 'Rectangle', 'Wire', or 'Cylindrical' (or closest match)."
    )
    dimensions_of_sample: Optional[str] = Field(
        None,
        description=(
            "Compact dimensions. Examples: wire '0.5mm core; 1.1 mm outer'; rectangle "
            "'20mm x 40mm x 1.5mm'; cylindrical '1.1 mm inner; 0.75 mm outer'."
        )
    )
    material_of_sample: Optional[str] = Field(
        None,
        description=(
            "Material name. Examples: 'black PMMA', 'LDPE', 'NiCr core; LDPE outer', "
            "'Cu core; LDPE outer', 'SS inner; LDPE outer', 'Cellulosic tissue', "
            "'Nomex HT90-40', 'Mylar G'."
        )
    )

    # Ambient / flow conditions
    oxygen_concentration: Optional[str] = Field(
        None,
        description="Oxygen concentration, ideally as a single value or clear range, e.g. '21%' or '16–27%'."
    )
    pressure: Optional[str] = Field(
        None,
        description="Pressure (or range) with units, e.g. '101.3 kPa', '30–100 kPa', '1 atm'."
    )
    flow_velocity: Optional[str] = Field(
        None,
        description="Flow velocity with sign convention if given (co-flow positive, counter-flow negative), "
                    "e.g. '0–350 mm/s' or '−150 mm/s'."
    )
    gravity_gearth: Optional[float] = Field(
        None,
        description="Gravity level as a numeric multiple of Earth gravity, e.g. 1.0 for normal gravity, "
                    "0.0 for microgravity, or a positive value like 0.16 if specified."
    )
    experimental_facility: Optional[str] = Field(
        None,
        description="Closest category from: 'Parabolic', 'Drop Tower', 'ISS', 'Sounding Rocket'."
    )

    # Ignition details
    ignition_method: Optional[str] = Field(
        None,
        description=(
            "Map to one of 'Wire', 'Open flame', 'Radiative Heater', and keep the specific "
            "description if useful (e.g. 'methane diffusion flame', 'hot-wire igniter')."
        )
    )
    ignition_power: Optional[str] = Field(
        None,
        description="Ignition power if given, with units, e.g. '20 W', '2.5 kW/m^2'. "
                    "Otherwise null."
    )
    ignition_time: Optional[str] = Field(
        None,
        description="Ignition duration if explicitly given, with units (e.g. '8 s'); otherwise null."
    )
    chamber_diameter: Optional[str] = Field(
        None,
        description="Diameter of combustion chamber or duct if reported, with units."
    )

    # Outcomes (we expect these to be mostly null because they are often only in plots)
    ignition_extinction: Optional[str] = Field(
        None,
        description="Use 'Yes' if the text clearly states ignition or extinction under this condition, "
                    "'No' if clearly not, otherwise null."
    )
    fsr: Optional[str] = Field(
        None,
        description="Flame spread rate (FSR) if a specific numeric value is explicitly given in text "
                    "for this condition (not just in plots). Otherwise null."
    )
    hrr: Optional[str] = Field(
        None,
        description="Heat release rate (HRR) if explicitly given in text for this condition. Otherwise null."
    )
    smoke_aerosols: Optional[str] = Field(
        None,
        description="Use 'Yes' if the text clearly indicates significant smoke/aerosol release for this "
                    "condition, 'No' if clearly absent, otherwise null."
    )

    # Extra fields for book-keeping / debugging
    experiment_label: Optional[str] = Field(
        None,
        description="Short label for this condition (free-form), for your own use."
    )
    notes: Optional[str] = Field(
        None,
        description="Any extra experimental notes that help interpret this row."
    )


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
    "wire": "Wire",
    "cylindrical": "Cylindrical",
    "cylinder": "Cylindrical",
    "rod": "Cylindrical",
    "flat": "Rectangle",
    "sheet": "Rectangle",
    "plate": "Rectangle",
}

FACILITY_CANONICAL = {
    "parabolic": "Parabolic Aircraft",
    "drop tower": "Drop Tower",
    "iss": "Spacecraft",
    "spacecraft": "Spacecraft",
    "rocket": "Sounding Rocket",
}


# ---------------------------------------------------------------------
# Utility: load extracted markdown context
# ---------------------------------------------------------------------

def load_extracted_markdown(markdown_path: Path) -> str:
    """
    Load markdown created by ext_exp.py.
    """
    with markdown_path.open("r", encoding="utf-8") as f:
        return f.read()


def truncate_markdown_context(context_md: str, max_chars: int) -> str:
    """
    Truncate markdown context to max_chars to avoid overly long prompts.
    """
    full = context_md.strip()

    if len(full) > max_chars:
        full = full[:max_chars] + "\n\n[TRUNCATED]\n"
    return full


def build_prompt_markdown(context_md: str, paper_label: str) -> str:
    """
    Build the full user prompt sent to the LLM, including instructions
    and the Markdown context.
    """
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
- article_mla: full reference similar to 'Konno, Yusuke, et al. "Experimental study on downward/opposed flame spread and extinction over electric wires in partial gravity environments." Proceedings of the Combustion Institute 39.3 (2023): 3785-3794.'
- authors: comma-separated list of authors.
- doi: DOI URL, e.g. 'https://doi.org/10.1016/j.proci.2018.06.022'.

Sample:
- geometry_of_sample: 'Rectangle', 'Wire', or 'cylindrical' (or closest match).
- dimensions_of_sample: compact description of sample dimensions consistent with the paper. (e.g. if wire: 0.5mm core; 1.1 mm outer or if rectangle: 20mm x 40mm x 1.5mm or if cylindrical: 1.1 mm inner; 0.75 mm outer )
- material_of_sample: material name (e.g. 'black PMMA', 'LDPE', 'NiCr core; LDPE outer', 'Cu core; LDPE outer' or 'SS inner; LDPE outer', 'Cellulosic tissue', 'Nomex HT90-40' or 'Mylar G' ).

Ambient / flow:
- oxygen_concentration: single value or range with units, e.g. '21%' or '16–27%'.
- pressure: single value or range with units, e.g. '101.3 kPa', '30–100 kPa', '1 atm'.
- flow_velocity: velocity information with sign convention if stated (co-flow positive, counter-flow negative).
- gravity_gearth: numeric factor of Earth gravity (1.0 for normal gravity, 0.0 for microgravity; use a positive value if a specific reduced-g is given).
- experimental_facility: one of ['Parabolic', 'Drop Tower', 'ISS', 'Sounding Rocket'], choosing the closest category based on the text.

Ignition:
- ignition_method: map the textual description to one of ['Wire', 'Open flame', 'Radiative Heater'] AND include the more specific description if helpful.
- ignition_power: numeric power or flux if explicitly given, with units.
- ignition_time: ignition duration if explicitly given, with units (e.g. '8 s').
- chamber_diameter: diameter of the combustion chamber or duct if it is given.

Outcomes (ONLY from text, not from plots):
- ignition_extinction: 'Yes' if the text clearly states ignition/extinction for this condition, 'No' if clearly not, otherwise null.
- fsr: flame spread rate as text if a specific numeric value is stated for this condition in the text; otherwise null.
- hrr: heat release rate as text if explicitly stated; otherwise null.
- smoke_aerosols: 'Yes' if the text clearly describes smoke/aerosol production for this condition, 'No' if clearly absent, otherwise null.

Additional:
- experiment_label: short free-form label for this condition (for your own use).
- notes: any extra description that helps interpret this row.

3. Do NOT invent values; only use what is in the excerpts. If you're unsure about a field,
   set it to null.

OUTPUT FORMAT:
Return a JSON object with a single key "rows", whose value is a list of row objects
following exactly the schema described above.

EXCERPTS (Markdown):

```markdown
{context_md}
```"""


# ---------------------------------------------------------------------
# OpenAI-compatible client (Groq) with basic retry logic
# ---------------------------------------------------------------------

def create_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """
    Create an OpenAI-compatible client. For Groq, use base_url
    'https://api.groq.com/openai/v1' and GROQ_API_KEY or --api-key.
    """
    key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Please set GROQ_API_KEY/OPENAI_API_KEY or pass --api-key.")

    url = base_url or "https://api.groq.com/openai/v1"
    return OpenAI(api_key=key, base_url=url)


def parse_json_like_content(content: str) -> Any:
    """
    Parse model output that may include fenced JSON or extra prose.
    """
    text = content.strip()
    if not text:
        raise json.JSONDecodeError("Empty model response", text, 0)

    # Best case: raw JSON object.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Common case: markdown code fence containing JSON.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        fenced = fence_match.group(1).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass

    # Last resort: extract first JSON object-like span.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

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
    text = text.strip()
    if text.startswith("https://doi.org/"):
        return text
    if text.startswith("http://doi.org/"):
        return "https://" + text[len("http://"):]
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
    """
    Call an OpenAI-compatible chat model that returns JSON in message.content.
    Then parse and validate it with Pydantic.
    """
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
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = completion.choices[0].message.content
            if not content:
                raise RuntimeError("Model returned empty content.")

            data = parse_json_like_content(content)
            result = ExtractionResult.model_validate(data)
            return result

        except (RateLimitError, APIConnectionError) as e:
            if attempt < max_retries - 1:
                print(
                    f"[WARN] Transient API error ({type(e).__name__}): {e}. "
                    f"Retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})...",
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


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def process_single_chunks_file(
    client: OpenAI,
    markdown_path: Path,
    model: str,
    max_chars: int,
    max_retries: int,
) -> ExtractionResult:
    context_md_raw = load_extracted_markdown(markdown_path)
    if not context_md_raw.strip():
        raise RuntimeError(f"No context found in extracted markdown: {markdown_path}")

    context_md = truncate_markdown_context(context_md_raw, max_chars=max_chars)
    paper_label = markdown_path.stem
    prompt = build_prompt_markdown(context_md, paper_label=paper_label)

    result = call_llm_json(client, prompt=prompt, model=model, max_retries=max_retries)
    return normalize_extraction_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-based extraction of database-shaped rows from extracted *.md context files "
            "using an OpenAI-compatible endpoint (e.g. Groq llama-3.3-70b-versatile).\n\n"
            "Example:\n"
            "  export GROQ_API_KEY=your_groq_key\n"
            "  python llm_extract.py test_v2.exp.md "
            "--out huang_2019_dbrows.json "
            "--model llama-3.3-70b-versatile\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "markdown",
        nargs="+",
        help="One or more paths or glob patterns to extracted *.md files (from ext_exp.py).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output JSON path. If multiple input files are provided and --out is set, "
            "a single JSON file containing a dict {chunks_file: rows[]} is written. "
            "If --out is not set, results are printed to stdout."
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
        help="Base URL for OpenAI-compatible endpoint (default: Groq's https://api.groq.com/openai/v1).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=20000,
        help="Max characters of concatenated context sent to the model.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for transient API errors.",
    )

    args = parser.parse_args()

    # Resolve glob patterns
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
            result = process_single_chunks_file(
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
        print(f"[INFO] Wrote combined extraction results to {out_path}", file=sys.stderr)
    else:
        json.dump(all_results, sys.stdout, ensure_ascii=False, indent=2)
        print()


if __name__ == "__main__":
    main()