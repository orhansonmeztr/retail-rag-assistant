"""Turn raw catalog/FAQ files into retrievable `Document` chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Document

_TR_MAP = str.maketrans("çğıİöşüÇĞÖŞÜ", "cgiiosucgosu")


def slugify(text: str) -> str:
    text = text.translate(_TR_MAP).lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def load_products(path: Path) -> list[Document]:
    return parse_products(path.read_text(encoding="utf-8"))


def parse_products(text: str) -> list[Document]:
    """JSON Lines catalog -> one document per product."""
    docs: list[Document] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        content = (
            f"{p['name']} - {p['brand']}. Kategori: {p['category']}. Cinsiyet: {p['gender']}. "
            f"Fiyat: {p['price']:.0f} TL. Renk: {p['color']}. Malzeme: {p['material']}. "
            f"Bedenler: {', '.join(p['sizes'])}. {p['description']}"
        )
        docs.append(
            Document(
                id=p["id"],
                source="product",
                title=f"{p['brand']} {p['name']}",
                content=content,
                category=p["category"],
                gender=p["gender"],
                price=float(p["price"]),
                url=f"https://example.com/urun/{p['id']}",
            )
        )
    return docs


def chunk_text(text: str, max_chars: int = 1200, overlap_paragraphs: int = 1) -> list[str]:
    """Paragraph-aware chunking; keeps paragraphs intact and overlaps by N paragraphs."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        if current and len("\n\n".join([*current, para])) > max_chars:
            chunks.append("\n\n".join(current))
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
        current.append(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def load_faq(path: Path, max_chars: int = 1200) -> list[Document]:
    return parse_faq(path.read_text(encoding="utf-8"), max_chars)


def parse_faq(text: str, max_chars: int = 1200) -> list[Document]:
    """Markdown FAQ (one `## ` section per topic) -> chunked documents."""
    sections = re.split(r"^## +", text, flags=re.MULTILINE)[1:]  # drop the H1 preamble
    docs: list[Document] = []
    for section in sections:
        heading, _, body = section.partition("\n")
        heading = heading.strip()
        for i, chunk in enumerate(chunk_text(body, max_chars=max_chars)):
            docs.append(
                Document(
                    id=f"faq-{slugify(heading)}-{i}",
                    source="faq",
                    title=heading,
                    content=f"{heading}\n\n{chunk}",
                    category="sss",
                    url=f"https://example.com/sss#{slugify(heading)}",
                )
            )
    return docs


def load_corpus(data_dir: Path) -> list[Document]:
    return load_products(data_dir / "products.jsonl") + load_faq(data_dir / "faq.md")
