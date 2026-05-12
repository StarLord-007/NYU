from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

BASE_PDF_DIR = Path("data") / "raw_pdfs"


def _clean_markdown(markdown: str) -> str:
    """
    Normalize markdown text so downstream heuristics and prompts are more stable.
    """
    text = markdown.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Keep line structure but avoid noisy spacing.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _with_pipeline_header(markdown: str, source_pdf: str) -> str:
    """
    Prefix markdown with provenance metadata useful for debugging and traceability.
    """
    ts = datetime.now(timezone.utc).isoformat()
    header = (
        "# Parsed PDF Context\n\n"
        f"- Source PDF: {source_pdf}\n"
        f"- Parsed UTC: {ts}\n\n"
    )
    return header + markdown


def _contains_glob(pattern: str) -> bool:
    return any(char in pattern for char in ("*", "?", "[", "]"))


def _resolve_unique_pdf(matches: list[Path], source: str) -> Path:
    pdf_matches = [match for match in matches if match.is_file() and match.suffix.lower() == ".pdf"]
    if not pdf_matches:
        raise FileNotFoundError(f"No PDF found for source: {source}")
    if len(pdf_matches) > 1:
        sample = ", ".join(str(path) for path in pdf_matches[:3])
        raise ValueError(
            "Source matched multiple PDFs; pass one explicit file. "
            f"Examples: {sample}"
        )
    return pdf_matches[0]


def _resolve_pdf_source(source: str, base_dir: Path = BASE_PDF_DIR) -> str:
    """Resolve one local PDF from data/raw_pdfs using filename, subpath, or glob."""
    source = source.strip()
    if not source:
        raise ValueError("Input PDF source cannot be empty.")

    base_dir = base_dir.resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Expected base directory does not exist: {base_dir}")

    if _contains_glob(source):
        if "/" in source or "\\" in source:
            # Treat explicit relative patterns as rooted at workspace.
            matches = sorted(Path(".").glob(source))
        else:
            # Treat bare filename patterns as recursive under data/raw_pdfs.
            matches = sorted(base_dir.rglob(source))
        resolved = _resolve_unique_pdf(matches, source)
    else:
        candidate = Path(source)
        if not candidate.suffix:
            candidate = candidate.with_suffix(".pdf")

        if candidate.is_absolute():
            resolved = candidate
        elif str(candidate).replace("\\", "/").startswith("data/raw_pdfs/"):
            resolved = candidate
        else:
            resolved = base_dir / candidate

        if not resolved.exists():
            fallback_matches = sorted(base_dir.rglob(candidate.name))
            resolved = _resolve_unique_pdf(fallback_matches, source)

    resolved = resolved.resolve()
    if base_dir not in resolved.parents:
        raise ValueError(f"PDF must be inside {base_dir}. Got: {resolved}")
    if resolved.is_dir():
        raise IsADirectoryError(f"Input is a directory, expected a PDF file: {resolved}")
    if resolved.suffix.lower() != ".pdf":
        raise ValueError(f"Input is not a PDF file: {resolved}")

    return str(resolved)


def _extract_markdown(result: object) -> str:
    """Extract markdown text from either newer or older Docling result objects."""
    if hasattr(result, "render_as_markdown"):
        return str(result.render_as_markdown())

    document = getattr(result, "document", None)
    if document is not None and hasattr(document, "export_to_markdown"):
        return str(document.export_to_markdown())

    raise RuntimeError(
        "Could not extract Markdown from Docling result. "
        "Expected render_as_markdown() or document.export_to_markdown()."
    )


def _build_converter(lightweight: bool = False) -> DocumentConverter:
    if not lightweight:
        return DocumentConverter()

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    pipeline_options.do_code_enrichment = False
    pipeline_options.do_formula_enrichment = False
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.generate_table_images = False

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def convert_pdf_to_markdown(
    source: str,
    out_path: Optional[Path] = None,
    use_convert_single: bool = True,
    lightweight: bool = False,
    max_pages: Optional[int] = None,
    page_range: Optional[Tuple[int, int]] = None,
    clean_markdown: bool = True,
    add_pipeline_header: bool = True,
) -> Path:
    """
    Convert a single local PDF to Markdown using Docling.

    Parameters
    ----------
    source : str
        PDF name, subpath, or glob that resolves inside data/raw_pdfs.
    out_path : Optional[Path]
        Where to write the Markdown file. If None, use <source>.md.
    use_convert_single : bool
        If True, use converter.convert_single(); otherwise use converter.convert().
        Both are supported by Docling; convert_single() is preferred in newer docs.

    Returns
    -------
    Path
        Path to the written Markdown file.
    """
    source = _resolve_pdf_source(source)

    # Resolve output path
    if out_path is None:
        src_path = Path(source)
        if src_path.suffix.lower() == ".pdf" and src_path.exists():
            out_path = src_path.with_suffix(".md")
        else:
            out_path = Path("output.md")

    out_path = out_path.resolve()

    # Create converter and run Docling pipeline
    converter = _build_converter(lightweight=lightweight)

    convert_kwargs = {}
    if max_pages is not None:
        convert_kwargs["max_num_pages"] = max_pages
    if page_range is not None:
        convert_kwargs["page_range"] = page_range

    if use_convert_single and hasattr(converter, "convert_single") and not convert_kwargs:
        # Newer Docling examples use convert_single() → result.render_as_markdown()
        # See: PyPI / docling docs.
        result = converter.convert_single(source)
        markdown = _extract_markdown(result)
    else:
        # Older examples and current stable builds use convert().
        result = converter.convert(source, **convert_kwargs)
        markdown = _extract_markdown(result)

    if clean_markdown:
        markdown = _clean_markdown(markdown)
    if add_pipeline_header:
        markdown = _with_pipeline_header(markdown, source_pdf=source)

    # Write to disk
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(markdown)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a PDF to Markdown using Docling.\n\n"
            "Example:\n"
            "  python parse_v2.py \"UC Berkeley/2019_Huang_ISS_PROCI.pdf\"\n"
            "  python parse_v2.py \"2019_Huang*.pdf\" --out test_v2.markdown --lightweight\n"
            "  python parse_v2.py \"UC Berkeley/2019_Huang_ISS_PROCI.pdf\" --no-lightweight --max-pages 8\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "pdf",
        type=str,
        help=(
            "Local PDF selector under data/raw_pdfs: filename, relative subpath, "
            "or glob pattern."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output Markdown path (default: same input path with .md).",
    )
    parser.add_argument(
        "--use-convert",
        action="store_true",
        help=(
            "Use converter.convert(...) + .document.export_to_markdown() "
            "instead of converter.convert_single(...). "
            "Mostly for compatibility with older Docling examples."
        ),
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        dest="lightweight",
        help="Use lighter PDF pipeline settings (disable OCR/table/code/formula enrichments).",
    )
    parser.add_argument(
        "--no-lightweight",
        action="store_false",
        dest="lightweight",
        help="Use full Docling pipeline settings.",
    )
    parser.set_defaults(lightweight=False)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional max number of pages to process.",
    )
    parser.add_argument(
        "--page-start",
        type=int,
        default=None,
        help="Optional 1-based start page (must be used with --page-end).",
    )
    parser.add_argument(
        "--page-end",
        type=int,
        default=None,
        help="Optional 1-based end page (must be used with --page-start).",
    )
    parser.add_argument(
        "--clean-markdown",
        action="store_true",
        dest="clean_markdown",
        help="Normalize markdown whitespace and line endings before writing output.",
    )
    parser.add_argument(
        "--no-clean-markdown",
        action="store_false",
        dest="clean_markdown",
        help="Write markdown exactly as returned by Docling.",
    )
    parser.set_defaults(clean_markdown=True)
    parser.add_argument(
        "--add-header",
        action="store_true",
        dest="add_header",
        help="Prefix markdown with source/timestamp metadata for pipeline traceability.",
    )
    parser.add_argument(
        "--no-add-header",
        action="store_false",
        dest="add_header",
        help="Do not add pipeline metadata header to markdown output.",
    )
    parser.set_defaults(add_header=True)

    args = parser.parse_args()

    pdf_source = args.pdf
    out_path = Path(args.out) if args.out is not None else None
    page_range: Optional[Tuple[int, int]] = None

    if (args.page_start is None) != (args.page_end is None):
        raise SystemExit("Both --page-start and --page-end must be set together.")
    if args.page_start is not None and args.page_end is not None:
        if args.page_start < 1 or args.page_end < args.page_start:
            raise SystemExit("Invalid page range. Expected 1 <= page-start <= page-end.")
        page_range = (args.page_start, args.page_end)

    try:
        md_path = convert_pdf_to_markdown(
            source=pdf_source,
            out_path=out_path,
            use_convert_single=not args.use_convert,
            lightweight=args.lightweight,
            max_pages=args.max_pages,
            page_range=page_range,
            clean_markdown=args.clean_markdown,
            add_pipeline_header=args.add_header,
        )
    except Exception as exc:
        raise SystemExit(f"Conversion failed: {exc}") from exc

    print(f"Converted '{pdf_source}' to Markdown at: {md_path}")


if __name__ == "__main__":
    main()