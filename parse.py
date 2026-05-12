from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

BBox = Tuple[float, float, float, float]

CAPTION_RE = re.compile(r"^(fig(?:ure)?|table|tab\.?)[\s.:\-]*\d*", re.IGNORECASE)
EQUATION_NUMBER_RE = re.compile(r"\(\s*\d+\s*\)$")
SECTION_HEADER_RE = re.compile(
    r"^(abstract|keywords?|introduction|conclusion(?:s)?|acknowledg(?:e)?ments?|references?)\b",
    re.IGNORECASE,
)
NUMBERED_SECTION_RE = re.compile(r"^\d+(?:\.\d+)*\.\s*\S+")
PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")

PAGE_HEADER_KEYWORDS = (
    "www.",
    "proceedings of the",
    "journal of",
    "elsevier",
    "sciencedirect",
    "vol.",
    "volume",
    "issue",
)

FOOTNOTE_KEYWORDS = (
    "doi",
    "copyright",
    "all rights reserved",
    "e-mail",
    "email",
    "corresponding author",
    "received",
    "accepted",
    "available online",
)

FIGURE_KEYWORDS = (
    "figure",
    "fig.",
    "diagram",
    "plot",
    "graph",
    "chart",
    "image",
    "picture",
    "schematic",
    "photo",
    "micrograph",
)

TABLE_KEYWORDS = ("table", "tab.")

METADATA_KEYWORDS = (
    "university",
    "department",
    "institute",
    "center",
    "centre",
    "laboratory",
    "school",
    "college",
    "nasa",
    "author",
    "corresponding",
)

HEADER_MARGIN_RATIO = 0.14
FOOTER_MARGIN_RATIO = 0.15
WIDE_BLOCK_RATIO = 0.65


@dataclass
class LayoutRegion:
    label: str
    score: float
    bbox: BBox
    source: str = "model"


@dataclass
class TextBlock:
    bbox: BBox
    text: str
    reading_order: Optional[int] = None
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    is_bold: Optional[bool] = None
    is_italic: Optional[bool] = None
    is_caption_like: bool = False
    is_table_candidate: bool = False
    is_figure_candidate: bool = False
    region_label: Optional[str] = None
    region_score: Optional[float] = None
    assignment_method: Optional[str] = None


@dataclass
class ParsedPage:
    page_num: int
    width: float
    height: float
    layout_regions: List[LayoutRegion]
    text_blocks: List[TextBlock]


@dataclass
class ParsedDocument:
    path: str
    model_name: str
    threshold: float
    dpi: int
    pages: List[ParsedPage]


def _render_page_to_pil(page: fitz.Page, dpi: int) -> Image.Image:
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _normalize_text(text: str) -> str:
    return " ".join(text.replace("\u00ad", "").split())


def _bbox_width(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def _bbox_height(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])


def _bbox_area(bbox: BBox) -> float:
    return _bbox_width(bbox) * _bbox_height(bbox)


def _bbox_center(bbox: BBox) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _point_in_bbox(point: Tuple[float, float], bbox: BBox) -> bool:
    x, y = point
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _bbox_intersection_area(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)

    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    return (inter_x1 - inter_x0) * (inter_y1 - inter_y0)


def _bbox_iou(a: BBox, b: BBox) -> float:
    inter_area = _bbox_intersection_area(a, b)
    if inter_area <= 0:
        return 0.0

    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    denom = area_a + area_b - inter_area

    if denom <= 0:
        return 0.0
    return inter_area / denom


def _bbox_overlap_ratio(a: BBox, b: BBox) -> float:
    area_a = _bbox_area(a)
    if area_a <= 0:
        return 0.0
    return _bbox_intersection_area(a, b) / area_a


def _bbox_center_distance(a: BBox, b: BBox, page_size: Tuple[float, float]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    page_w, page_h = page_size
    diagonal = (page_w * page_w + page_h * page_h) ** 0.5
    if diagonal <= 0:
        return 1.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / diagonal


def _most_common_nonempty(values: List[str]) -> Optional[str]:
    cleaned = [value for value in values if value]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def _span_is_bold(span: Dict[str, Any]) -> bool:
    font_name = str(span.get("font", "")).lower()
    flags = int(span.get("flags", 0) or 0)
    return "bold" in font_name or bool(flags & 16)


def _span_is_italic(span: Dict[str, Any]) -> bool:
    font_name = str(span.get("font", "")).lower()
    flags = int(span.get("flags", 0) or 0)
    return "italic" in font_name or "oblique" in font_name or bool(flags & 2)


def _is_caption_like(text: str) -> bool:
    normalized = _normalize_text(text)
    return bool(CAPTION_RE.match(normalized))


def _looks_like_table_candidate(text: str) -> bool:
    normalized = _normalize_text(text)
    lowered = normalized.lower()
    if any(keyword in lowered for keyword in TABLE_KEYWORDS):
        return True

    line_count = sum(1 for line in text.splitlines() if line.strip())
    if line_count < 2:
        return False

    digit_count = sum(character.isdigit() for character in normalized)
    has_delimiter = "|" in normalized or "\t" in normalized or normalized.count(";") >= 2
    return digit_count >= 10 and has_delimiter and len(normalized) < 500


def _looks_like_figure_candidate(text: str) -> bool:
    normalized = _normalize_text(text)
    lowered = normalized.lower()
    if _is_caption_like(normalized):
        return True
    return any(keyword in lowered for keyword in FIGURE_KEYWORDS) and len(normalized.split()) < 80


def _has_equation_number(text: str) -> bool:
    return bool(EQUATION_NUMBER_RE.search(_normalize_text(text)))


def _looks_like_section_header(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return bool(SECTION_HEADER_RE.match(normalized.lower()) or NUMBERED_SECTION_RE.match(normalized))


def _is_page_header_like(text: str, bbox: BBox, page_size: Tuple[float, float]) -> bool:
    normalized = _normalize_text(text)
    lowered = normalized.lower()
    page_width, page_height = page_size
    top_ratio = bbox[1] / page_height if page_height > 0 else 0.0

    if top_ratio > HEADER_MARGIN_RATIO:
        return False

    if PAGE_NUMBER_RE.match(normalized):
        return True

    if any(keyword in lowered for keyword in PAGE_HEADER_KEYWORDS):
        return True

    return len(normalized.split()) <= 14 and _bbox_width(bbox) >= page_width * 0.25


def _is_footnote_like(text: str, bbox: BBox, page_size: Tuple[float, float]) -> bool:
    normalized = _normalize_text(text)
    lowered = normalized.lower()
    _, page_height = page_size
    bottom_ratio = bbox[3] / page_height if page_height > 0 else 1.0

    if any(keyword in lowered for keyword in FOOTNOTE_KEYWORDS):
        return True

    return bottom_ratio >= 1.0 - FOOTER_MARGIN_RATIO and len(normalized.split()) <= 40


def _looks_like_author_block(text: str, bbox: BBox, page_size: Tuple[float, float]) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    page_width, page_height = page_size
    if bbox[1] > page_height * 0.40 or _bbox_width(bbox) < page_width * 0.45:
        return False

    lowered = normalized.lower()
    if any(keyword in lowered for keyword in METADATA_KEYWORDS):
        return False

    words = normalized.replace("∗", " ").replace("†", " ").split()
    if len(words) < 6 or len(words) > 40:
        return False

    if normalized.count(",") < 2:
        return False

    capitalized_tokens = sum(1 for word in words if word[:1].isupper())
    return capitalized_tokens >= 4


def _infer_fallback_label(text_block: TextBlock, page_size: Tuple[float, float]) -> Optional[Tuple[str, float, str]]:
    text = _normalize_text(text_block.text)
    if not text:
        return None

    if _is_caption_like(text):
        return "Caption", 0.9, "heuristic-caption"

    if _is_page_header_like(text, text_block.bbox, page_size):
        return "Page-header", 0.8, "heuristic-header"

    if _is_footnote_like(text, text_block.bbox, page_size):
        return "Footnote", 0.78, "heuristic-footnote"

    if _looks_like_author_block(text, text_block.bbox, page_size):
        return "Text", 0.56, "heuristic-author"

    if _looks_like_section_header(text):
        return "Section-header", 0.75, "heuristic-section-header"

    if text_block.bbox[1] <= page_size[1] * 0.42 and any(
        keyword in text.lower() for keyword in METADATA_KEYWORDS
    ):
        return "Text", 0.58, "heuristic-metadata"

    if _looks_like_table_candidate(text):
        return "Text", 0.45, "heuristic-table-candidate"

    if _looks_like_figure_candidate(text):
        return "Text", 0.4, "heuristic-figure-candidate"

    return None


def _sort_text_blocks_for_reading_order(text_blocks: List[TextBlock], page_size: Tuple[float, float]) -> List[TextBlock]:
    page_width, page_height = page_size
    top_band_threshold = page_height * 0.25
    row_tolerance = max(4.0, min(8.0, page_width * 0.012))

    def sort_key(block: TextBlock) -> Tuple[float, int, float, float]:
        x0, y0, x1, _ = block.bbox
        block_width = x1 - x0

        if y0 < top_band_threshold:
            return (0, int(y0 // row_tolerance), x0, y0)

        if block_width >= page_width * WIDE_BLOCK_RATIO:
            return (1, 0, y0, x0)

        center_x = (x0 + x1) / 2.0
        column_rank = 0 if center_x <= page_width / 2.0 else 1
        return (2, column_rank, y0, x0)

    return sorted(text_blocks, key=sort_key)


def _merge_formula_blocks(text_blocks: List[TextBlock], page_width: float) -> List[TextBlock]:
    if not text_blocks:
        return text_blocks

    merged_blocks: List[TextBlock] = []
    merge_gap = max(6.0, page_width * 0.015)

    for text_block in text_blocks:
        if not merged_blocks:
            merged_blocks.append(text_block)
            continue

        previous_block = merged_blocks[-1]
        if (
            previous_block.region_label == "Formula"
            and text_block.region_label == "Formula"
            and not _has_equation_number(previous_block.text)
            and _has_equation_number(text_block.text)
        ):
            overlap_x0 = max(previous_block.bbox[0], text_block.bbox[0])
            overlap_x1 = min(previous_block.bbox[2], text_block.bbox[2])
            overlap_width = max(0.0, overlap_x1 - overlap_x0)
            min_width = min(_bbox_width(previous_block.bbox), _bbox_width(text_block.bbox))
            vertical_gap = text_block.bbox[1] - previous_block.bbox[3]

            if min_width > 0 and overlap_width / min_width >= 0.15 and vertical_gap <= merge_gap:
                merged_blocks[-1] = TextBlock(
                    bbox=(
                        min(previous_block.bbox[0], text_block.bbox[0]),
                        min(previous_block.bbox[1], text_block.bbox[1]),
                        max(previous_block.bbox[2], text_block.bbox[2]),
                        max(previous_block.bbox[3], text_block.bbox[3]),
                    ),
                    text=f"{previous_block.text}\n{text_block.text}",
                    reading_order=previous_block.reading_order,
                    font_name=_most_common_nonempty([previous_block.font_name or "", text_block.font_name or ""]),
                    font_size=(previous_block.font_size + text_block.font_size) / 2.0
                    if previous_block.font_size is not None and text_block.font_size is not None
                    else previous_block.font_size or text_block.font_size,
                    is_bold=bool(previous_block.is_bold or text_block.is_bold),
                    is_italic=bool(previous_block.is_italic or text_block.is_italic),
                    is_caption_like=bool(previous_block.is_caption_like or text_block.is_caption_like),
                    is_table_candidate=bool(previous_block.is_table_candidate or text_block.is_table_candidate),
                    is_figure_candidate=bool(previous_block.is_figure_candidate or text_block.is_figure_candidate),
                    region_label="Formula",
                    region_score=max(
                        previous_block.region_score or 0.0,
                        text_block.region_score or 0.0,
                    ),
                    assignment_method="layout-model-merged",
                )
                continue

        merged_blocks.append(text_block)

    return merged_blocks


def _scale_bbox_from_image_to_pdf(image_bbox: BBox, image_size: Tuple[int, int], pdf_size: Tuple[float, float]) -> BBox:
    img_w, img_h = image_size
    pdf_w, pdf_h = pdf_size
    sx = pdf_w / img_w
    sy = pdf_h / img_h

    x0, y0, x1, y1 = image_bbox
    return x0 * sx, y0 * sy, x1 * sx, y1 * sy


def _extract_text_blocks(page: fitz.Page) -> List[TextBlock]:
    blocks: List[TextBlock] = []
    text_dict = page.get_text("dict")

    for block in text_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue

        lines = block.get("lines", [])
        if not lines:
            continue

        line_texts: List[str] = []
        font_names: List[str] = []
        font_sizes: List[float] = []
        bold_votes = 0
        italic_votes = 0

        for line in lines:
            span_texts: List[str] = []
            for span in line.get("spans", []):
                span_text = str(span.get("text", ""))
                if not span_text:
                    continue

                span_texts.append(span_text)

                font_name = str(span.get("font", "")).strip()
                if font_name:
                    font_names.append(font_name)

                size = span.get("size")
                if isinstance(size, (int, float)):
                    font_sizes.append(float(size))

                if _span_is_bold(span):
                    bold_votes += 1
                if _span_is_italic(span):
                    italic_votes += 1

            if span_texts:
                line_text = "".join(span_texts).strip()
                if line_text:
                    line_texts.append(line_text)

        text = "\n".join(line_texts).strip()
        if not text:
            continue

        bbox = tuple(float(value) for value in block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        average_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else None

        blocks.append(
            TextBlock(
                bbox=bbox,
                text=text,
                font_name=_most_common_nonempty(font_names),
                font_size=average_font_size,
                is_bold=(bold_votes > 0) if font_names else None,
                is_italic=(italic_votes > 0) if font_names else None,
                is_caption_like=_is_caption_like(text),
                is_table_candidate=_looks_like_table_candidate(text),
                is_figure_candidate=_looks_like_figure_candidate(text),
            )
        )

    return blocks


def _assign_regions_to_text_blocks(
    text_blocks: List[TextBlock],
    regions: List[LayoutRegion],
    page_size: Tuple[float, float],
) -> None:
    for text_block in text_blocks:
        best_region: Optional[LayoutRegion] = None
        best_iou = 0.0
        best_overlap_ratio = 0.0
        best_center_inside = False

        for region in regions:
            iou = _bbox_iou(text_block.bbox, region.bbox)
            overlap_ratio = _bbox_overlap_ratio(text_block.bbox, region.bbox)
            center_inside = _point_in_bbox(_bbox_center(text_block.bbox), region.bbox)

            if (
                iou > best_iou
                or (iou == best_iou and overlap_ratio > best_overlap_ratio)
                or (iou == best_iou and overlap_ratio == best_overlap_ratio and center_inside and not best_center_inside)
            ):
                best_region = region
                best_iou = iou
                best_overlap_ratio = overlap_ratio
                best_center_inside = center_inside

        if best_region is not None and (
            best_iou >= 0.02 or best_overlap_ratio >= 0.20 or best_center_inside
        ):
            text_block.region_label = best_region.label
            text_block.region_score = best_region.score
            text_block.assignment_method = "layout-model"

    for text_block in text_blocks:
        if text_block.region_label is not None:
            continue

        fallback = _infer_fallback_label(text_block, page_size)
        if fallback is None:
            continue

        label, confidence, method = fallback
        text_block.region_label = label
        text_block.region_score = confidence
        text_block.assignment_method = method
        regions.append(
            LayoutRegion(
                label=label,
                score=confidence,
                bbox=text_block.bbox,
                source="heuristic",
            )
        )


def parse_pdf_with_layout_model(
    pdf_path: str | Path,
    model_name: str = "Aryn/deformable-detr-DocLayNet",
    threshold: float = 0.6,
    dpi: int = 144,
    max_pages: Optional[int] = None,
    device: Optional[str] = None,
) -> ParsedDocument:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name)
    model.to(device)
    model.eval()

    pages: List[ParsedPage] = []

    with fitz.open(pdf_path) as doc:
        page_count = len(doc) if max_pages is None else min(len(doc), max_pages)

        for page_idx in range(page_count):
            page = doc.load_page(page_idx)
            pdf_width, pdf_height = page.rect.width, page.rect.height

            image = _render_page_to_pil(page, dpi=dpi)
            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]], device=device)
            predictions = processor.post_process_object_detection(
                outputs,
                threshold=threshold,
                target_sizes=target_sizes,
            )[0]

            regions: List[LayoutRegion] = []
            for score, label_id, bbox_tensor in zip(
                predictions["scores"], predictions["labels"], predictions["boxes"]
            ):
                label = model.config.id2label[int(label_id)]
                x0, y0, x1, y1 = [float(value) for value in bbox_tensor.tolist()]
                pdf_bbox = _scale_bbox_from_image_to_pdf(
                    (x0, y0, x1, y1), image.size, (pdf_width, pdf_height)
                )
                regions.append(
                    LayoutRegion(
                        label=label,
                        score=float(score),
                        bbox=pdf_bbox,
                    )
                )

            text_blocks = _extract_text_blocks(page)
            text_blocks = _sort_text_blocks_for_reading_order(text_blocks, (float(pdf_width), float(pdf_height)))
            text_blocks = _merge_formula_blocks(text_blocks, float(pdf_width))
            for reading_order, text_block in enumerate(text_blocks, start=1):
                text_block.reading_order = reading_order

            _assign_regions_to_text_blocks(text_blocks, regions, (float(pdf_width), float(pdf_height)))

            pages.append(
                ParsedPage(
                    page_num=page_idx + 1,
                    width=float(pdf_width),
                    height=float(pdf_height),
                    layout_regions=regions,
                    text_blocks=text_blocks,
                )
            )

    return ParsedDocument(
        path=str(pdf_path),
        model_name=model_name,
        threshold=threshold,
        dpi=dpi,
        pages=pages,
    )


def save_parsed_document(doc: ParsedDocument, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "path": doc.path,
        "model_name": doc.model_name,
        "threshold": doc.threshold,
        "dpi": doc.dpi,
        "pages": [asdict(page) for page in doc.pages],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a PDF with a vision-layout model and heuristic post-processing."
    )
    parser.add_argument("pdf_path", type=str, help="Path to input PDF")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: <pdf_name>.layout.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Aryn/deformable-detr-DocLayNet",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Detection threshold in [0, 1]",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=144,
        help="Render DPI for PDF pages",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page limit for faster runs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Force device ("cpu" or "cuda"). Default auto-detect.',
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    out_path = Path(args.out) if args.out else pdf_path.with_suffix(".layout.json")

    parsed = parse_pdf_with_layout_model(
        pdf_path=pdf_path,
        model_name=args.model,
        threshold=args.threshold,
        dpi=args.dpi,
        max_pages=args.max_pages,
        device=args.device,
    )
    save_parsed_document(parsed, out_path)
    print(f"Saved layout JSON to {out_path}")


if __name__ == "__main__":
    main()
