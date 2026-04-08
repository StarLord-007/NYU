# parse_pdf.py

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import fitz  # PyMuPDF


BBox = Tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class TextBlock:
    page_num: int
    bbox: BBox
    text: str
    is_caption_like: bool = False
    is_table_candidate: bool = False
    is_figure_candidate: bool = False


@dataclass
class PageLayout:
    page_num: int
    width: float
    height: float
    text_blocks: List[TextBlock]


@dataclass
class ParsedPDF:
    path: str
    pages: List[PageLayout]


def _is_caption_like(text: str) -> bool:
    """Heuristique simple pour légendes de figures/tables."""
    t = text.strip().lower()
    return (
        t.startswith("fig.") or t.startswith("fig ")
        or t.startswith("figure ")
        or t.startswith("table ") or t.startswith("tab.")
    )


def _classify_block(text: str) -> Tuple[bool, bool]:
    """
    Heuristique très simple pour distinguer blocs 'table-like' / 'figure-like'.
    À améliorer plus tard, ou à remplacer par un vrai modèle de layout.
    """
    t = text.strip()

    # Table candidate: beaucoup de chiffres, séparateurs, petits mots
    num_chars = sum(ch.isdigit() for ch in t)
    sep_chars = t.count(" ") + t.count("\t") + t.count(";") + t.count(",")
    table_like = (num_chars > 10 and sep_chars > 10) or "nomenclature" in t.lower()

    # Figure candidate: contient souvent "flame", "temperature", "velocity" etc. + courte légende
    fig_keywords = ["flame", "temperature", "velocity", "rate", "ignition", "spread"]
    fig_like = any(k in t.lower() for k in fig_keywords) and len(t.split()) < 40

    return table_like, fig_like


def parse_pdf(path: str | Path) -> ParsedPDF:
    """
    Parse un PDF scientifique et retourne une structure PageLayout / TextBlock.

    - Utilise PyMuPDF pour récupérer les blocs de texte et leurs bounding boxes.
    - Ne fait PAS encore de vraie détection d’images ou tables par vision,
      mais prépare une structure sur laquelle tu pourras brancher des modèles plus avancés.
    """
    path = str(path)
    doc = fitz.open(path)

    pages: List[PageLayout] = []

    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        width, height = page.rect.width, page.rect.height

        # PyMuPDF: "blocks" = liste [x0, y0, x1, y1, text, block_no, block_type, ...]
        raw_blocks = page.get_text("blocks")

        text_blocks: List[TextBlock] = []
        for block in raw_blocks:
            x0, y0, x1, y1, text, *_ = block
            if not text or not text.strip():
                continue

            bbox: BBox = (x0, y0, x1, y1)

            is_caption = _is_caption_like(text)
            is_table_cand, is_fig_cand = _classify_block(text)

            tb = TextBlock(
                page_num=page_idx + 1,
                bbox=bbox,
                text=text,
                is_caption_like=is_caption,
                is_table_candidate=is_table_cand,
                is_figure_candidate=is_fig_cand,
            )
            text_blocks.append(tb)

        pages.append(
            PageLayout(
                page_num=page_idx + 1,
                width=width,
                height=height,
                text_blocks=text_blocks,
            )
        )

    doc.close()
    return ParsedPDF(path=path, pages=pages)


def parsed_pdf_to_dict(parsed: ParsedPDF) -> Dict[str, Any]:
    """Convertit ParsedPDF en dict JSON‑sérialisable."""
    return {
        "path": parsed.path,
        "pages": [
            {
                "page_num": p.page_num,
                "width": p.width,
                "height": p.height,
                "text_blocks": [asdict(tb) for tb in p.text_blocks],
            }
            for p in parsed.pages
        ],
    }


def save_parsed_pdf(parsed: ParsedPDF, out_path: str | Path) -> None:
    """Sauvegarde le résultat dans un fichier JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = parsed_pdf_to_dict(parsed)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse un PDF de combustion microgravité.")
    parser.add_argument("pdf_path", type=str, help="Chemin vers le fichier PDF")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Chemin du JSON de sortie (par défaut: même nom, .json)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    parsed = parse_pdf(pdf_path)

    if args.out is None:
        out_path = pdf_path.with_suffix(".json")
    else:
        out_path = Path(args.out)

    save_parsed_pdf(parsed, out_path)
    print(f"Sauvé dans {out_path}")
  
