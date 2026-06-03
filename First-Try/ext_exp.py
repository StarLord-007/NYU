from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    type: str
    source_markdown: str
    section_heading: Optional[str]
    line_start: int  # 1-based
    line_end: int    # 1-based
    text: str
    meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Heuristics 
# ---------------------------------------------------------------------------

# Section headings that usually contain experimental conditions
SETUP_SECTION_PATTERNS = [
    r"\bexperimental\b",
    r"\bmethods?\b",
    r"\bmaterials?\b",
    r"\bapparatus\b",
    r"\bsetup\b",
    r"\btest(s)?\b",
    r"\bfacilit(y|ies)\b",
    r"\bmicrogravity\b",
    r"\bparabolic flight\b",
    r"\bdrop tower\b",
]

# Section headings that we explicitly do NOT want to treat as setup
NON_SETUP_SECTION_PREFIXES = [
    "nomenclature",
    "references",
    "acknowledgements",
    "acknowledgments",
    "conclusions",
    "conclusion",
    "numerical model",
    "modeling",
    "model development",
    "theory",
    "theoretical",
    "results",
    "discussion",
    "results and discussion",
]

# Keywords that tend to appear in actual experimental descriptions
SETUP_KEYWORDS = [
    # Materials / fuels
    "pmma",
    "poly(methyl methacrylate)",
    "poly(methyl-methacrylate)",
    "ldpe",
    "low density polyethylene",
    "polyethylene",
    "etfe",
    "tefzel",
    "insulated wire",
    "insulation",
    "copper core",
    "nickel chromium",
    "nicr",
    # Geometry
    "rod",
    "rods",
    "cylinder",
    "cylinders",
    "cylindrical",
    "wire",
    "wires",
    "cable",
    "cables",
    "sheet",
    "strip",
    "sample",
    "samples",
    "diameter",
    "radius",
    "length",
    "thickness",
    "mm",
    "cm",
    "m ",
    # Environment / facility
    "microgravity",
    "normal gravity",
    "1 g",
    "1g",
    "reduced gravity",
    "parabolic flight",
    "drop tower",
    "iss",
    "international space station",
    "bass-ii",
    "msg",
    "spacecraft",
    "ventilation",
    "duct",
    "tunnel",
    "chamber",
    "pressure",
    "kpa",
    "pa",
    # Flow / atmosphere
    "opposed flow",
    "upward flow",
    "flow velocity",
    "velocity",
    "cm/s",
    "mm/s",
    "m/s",
    "flow rate",
    "oxidizer",
    "air flow",
    "oxygen",
    "o2",
    "x o2",
    "mole fraction",
    # Heating / radiation
    "radiant flux",
    "external radiation",
    "heat flux",
    "kw/m2",
    "kw m-2",
    "kW/m",
    # Ignition / hardware
    "ignition",
    "igniter",
    "hot-wire",
    "methane flame",
    "ceramic heater",
    "halogen lamp",
    # Misc experimental words
    "experiment",
    "experiments",
    "tested",
    "under the same conditions",
]

FIGURE_PREFIXES = [
    "fig. ",
    "figure ",
    "fig ",
]

MEASUREMENT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|kpa|pa|atm|mm|cm|m|mm/s|cm/s|m/s|w|kw|kw/m2|kw m-2|s)\b",
    flags=re.IGNORECASE,
)

BACKGROUND_PHRASES = [
    "fire safety is a concern",
    "important concern",
    "future spacecraft",
    "insufficient knowledge",
    "has been studied",
    "in this paper",
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    s = unescape(s)
    s = re.sub(r"\s+", " ", s)
    return s


def _is_heading(line: str) -> Optional[str]:
    """
    Return heading text if this line is a Markdown heading (## ...), else None.
    We ignore level-1 (#) headings; level >=2 are treated as section headings.
    """
    m = re.match(r"^(#+)\s+(.*)$", line.strip())
    if not m:
        return None
    level = len(m.group(1))
    text = m.group(2).strip()
    if level >= 2 and text:
        return text
    return None


def _is_figure_caption(line: str) -> bool:
    line_norm = line.strip().lower()
    # Also treat Markdown image captions that start with "![Figure" as captions
    if line_norm.startswith("![figure") or line_norm.startswith("![fig."):
        return True
    for prefix in FIGURE_PREFIXES:
        if line_norm.startswith(prefix):
            return True
    return False


def _section_is_non_setup(section_heading: Optional[str]) -> bool:
    if not section_heading:
        return False
    h = section_heading.strip().lower()
    for prefix in NON_SETUP_SECTION_PREFIXES:
        if h.startswith(prefix):
            return True
    return False


def _looks_like_setup_paragraph(text: str, section_heading: Optional[str]) -> bool:
    """
    Heuristic: paragraph mentions experimental conditions and is not obviously
    references / pure theory / nomenclature.
    """
    if not text:
        return False

    lowered = text.lower().strip()

    # Quick skips: reference lists, equations-only lines, etc.
    if lowered.startswith("references") or lowered.startswith("## references"):
        return False
    if re.match(r"^\[\d+\]\s", lowered):
        return False

    # Skip if the section is clearly non-setup
    if _section_is_non_setup(section_heading):
        return False

    # Section-based hint: if heading looks like experimental section, be lenient
    section_boost = False
    if section_heading:
        sh = section_heading.lower()
        if any(re.search(pattern, sh) for pattern in SETUP_SECTION_PATTERNS):
            section_boost = True

    keyword_hits = sum(1 for kw in SETUP_KEYWORDS if kw in lowered)
    measurement_hits = len(MEASUREMENT_PATTERN.findall(lowered))
    has_condition_terms = any(
        token in lowered for token in ("condition", "conditions", "under", "ranging", "varied")
    )
    has_outcome_terms = any(
        token in lowered for token in ("ignition", "extinction", "flame spread", "fsr")
    )

    looks_like_background = any(phrase in lowered for phrase in BACKGROUND_PHRASES)
    if looks_like_background and measurement_hits == 0 and keyword_hits < 4:
        return False

    score = 0
    score += min(keyword_hits, 6)
    score += min(measurement_hits, 3)
    if section_boost:
        score += 2
    if has_condition_terms:
        score += 1
    if has_outcome_terms:
        score += 1

    if section_boost:
        return score >= 3
    return score >= 5 and (measurement_hits > 0 or has_outcome_terms)


def _read_markdown_lines(md_path: Path) -> List[str]:
    with md_path.open("r", encoding="utf-8") as f:
        return f.readlines()


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def extract_chunks_from_markdown(md_path: Path) -> List[Chunk]:
    lines = _read_markdown_lines(md_path)
    chunks: List[Chunk] = []

    current_section: Optional[str] = None
    buffer: List[str] = []
    buffer_start_line: Optional[int] = None

    def flush_paragraph(end_line_idx: int) -> None:
        nonlocal buffer, buffer_start_line, current_section

        if not buffer or buffer_start_line is None:
            buffer = []
            buffer_start_line = None
            return

        raw_text = "".join(buffer)
        text = _normalize(raw_text)
        if not text:
            buffer = []
            buffer_start_line = None
            return

        first_line = buffer[0].strip()

        if _is_figure_caption(first_line):
            chunk_type = "caption_context"
            meta = {
                "is_caption": True,
                "section_heading": current_section,
            }
        elif _looks_like_setup_paragraph(text, current_section):
            chunk_type = "setup_paragraph"
            meta = {
                "contains_setup_keywords": True,
                "section_heading": current_section,
            }
        else:
            chunk_type = "other_paragraph"
            meta = {
                "section_heading": current_section,
            }

        chunks.append(
            Chunk(
                type=chunk_type,
                source_markdown=str(md_path),
                section_heading=current_section,
                line_start=buffer_start_line + 1,  # 1-based
                line_end=end_line_idx + 1,
                text=text,
                meta=meta,
            )
        )

        buffer = []
        buffer_start_line = None

    for idx, raw_line in enumerate(lines):
        line = raw_line.rstrip("\n")

        # Skip pure Docling image markers if present
        if line.strip().startswith("<!-- image"):
            continue

        # Heading → flush paragraph and update section
        heading_text = _is_heading(line)
        if heading_text is not None:
            flush_paragraph(idx - 1)
            current_section = heading_text
            continue

        # Blank line → paragraph boundary
        if not line.strip():
            flush_paragraph(idx - 1)
            continue

        # Accumulate into current paragraph
        if buffer_start_line is None:
            buffer_start_line = idx
        buffer.append(line + "\n")

    # Flush trailing paragraph
    flush_paragraph(len(lines) - 1)

    # Deduplicate by (type, section, text)
    unique: Dict[str, Chunk] = {}
    for ch in chunks:
        key = f"{ch.type}|{ch.section_heading or ''}|{ch.text}"
        if key not in unique:
            unique[key] = ch

    return list(unique.values())


def render_extracted_markdown(chunks: List[Chunk], source_markdown: Path, max_excerpts: int) -> str:
    """
    Render selected setup/caption chunks to a compact markdown file consumed by llm_extract.py.
    """
    selected = [
        ch for ch in chunks if ch.type in ("setup_paragraph", "caption_context")
    ][:max_excerpts]

    lines: List[str] = []
    lines.append("# Extracted Experimental Context")
    lines.append("")
    lines.append(f"Source markdown: {source_markdown}")
    lines.append(f"Total selected excerpts: {len(selected)}")
    lines.append("")

    for idx, ch in enumerate(selected, start=1):
        lines.append(f"## Excerpt {idx}")
        lines.append("")
        lines.append(f"- Type: {ch.type}")
        lines.append(f"- Section: {ch.section_heading or 'N/A'}")
        lines.append(f"- Lines: {ch.line_start}-{ch.line_end}")
        lines.append("")
        lines.append(ch.text)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def save_extracted_markdown(
    chunks: List[Chunk],
    out_path: Path,
    source_markdown: Path,
    max_excerpts: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_extracted_markdown(
        chunks,
        source_markdown=source_markdown,
        max_excerpts=max_excerpts,
    )
    with out_path.open("w", encoding="utf-8") as f:
        f.write(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract experimental setup and caption-context excerpts into Markdown.\n\n"
            "Example:\n"
            "  python ext_exp.py test_v2.markdown\n"
            "  python ext_exp.py test_v2.markdown --out test_v2.exp.md\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "markdown",
        type=str,
        help="Path to the Markdown file produced by parse_v2.py / Docling.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output Markdown path (default: same path with .exp.md).",
    )
    parser.add_argument(
        "--max-excerpts",
        type=int,
        default=80,
        help="Maximum number of selected setup/caption excerpts to write.",
    )

    args = parser.parse_args()
    md_path = Path(args.markdown)
    if not md_path.exists():
        raise SystemExit(f"Markdown not found: {md_path}")

    out_path = Path(args.out) if args.out else md_path.with_suffix(".exp.md")

    try:
        chunks = extract_chunks_from_markdown(md_path)
    except Exception as exc:
        raise SystemExit(f"Chunk extraction failed: {exc}") from exc

    save_extracted_markdown(
        chunks,
        out_path,
        md_path,
        max_excerpts=max(1, args.max_excerpts),
    )
    selected_count = sum(1 for ch in chunks if ch.type in ("setup_paragraph", "caption_context"))
    written_count = min(selected_count, max(1, args.max_excerpts))
    print(f"Extracted {written_count} setup/caption excerpts to {out_path} ({selected_count} selected total)")


if __name__ == "__main__":
    main()