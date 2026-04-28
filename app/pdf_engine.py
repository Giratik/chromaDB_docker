"""
pdf_engine.py
─────────────
Extraction de texte depuis des PDFs (natif + OCR fallback)
et découpage en chunks avec overlap.

Dépendances :
  - pdfplumber        : extraction texte natif
  - Pillow            : manipulation image pour OCR
  - pytesseract       : OCR (optionnel, nécessite tesseract-ocr installé)
"""

import re
import io
from datetime import datetime
from typing import List, Dict, Any

import pdfplumber

# OCR optionnel
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# ─── PARAMÈTRES ────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500   # caractères par chunk
CHUNK_OVERLAP = 100   # overlap entre chunks
MIN_CHARS_NATIVE = 30 # seuil : en dessous → probablement page scannée


# ─── EXTRACTION ────────────────────────────────────────────────────────────────

def _extract_page_text(page) -> str:
    """
    Extrait le texte d'une page pdfplumber.
    Si le texte natif est insuffisant, tente l'OCR via pytesseract.
    """
    text = (page.extract_text() or "").strip()

    if len(text) < MIN_CHARS_NATIVE and OCR_AVAILABLE:
        try:
            img = page.to_image(resolution=200).original
            ocr_text = pytesseract.image_to_string(img, lang="fra+eng").strip()
            if len(ocr_text) > len(text):
                text = ocr_text
        except Exception:
            pass

    return text


def extract_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Retourne [{page_num, text}] pour chaque page non vide."""
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = re.sub(r'\n{3,}', '\n\n', _extract_page_text(page))
            text = re.sub(r' {2,}', ' ', text).strip()
            if text:
                pages.append({"page_num": i, "text": text})
    return pages


# ─── CHUNKING ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Découpe en chunks de ~chunk_size caractères avec overlap.
    Priorité de coupure : paragraphe > phrase > espace.
    """
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        if end >= length:
            chunks.append(text[start:].strip())
            break

        slice_ = text[start:end]
        cut = -1

        para = slice_.rfind('\n\n')
        if para > chunk_size // 2:
            cut = para + 2

        if cut == -1:
            for punct in ['. ', '! ', '? ', '.\n']:
                s = slice_.rfind(punct)
                if s > chunk_size // 2:
                    cut = s + len(punct)
                    break

        if cut == -1:
            sp = slice_.rfind(' ')
            cut = sp + 1 if sp > 0 else chunk_size

        chunk = text[start:start + cut].strip()
        if chunk:
            chunks.append(chunk)

        start = start + cut - overlap
        if start < 0:
            start = 0

    return [c for c in chunks if len(c) > 20]


# ─── PIPELINE ──────────────────────────────────────────────────────────────────

def pdf_to_chunks(pdf_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    PDF bytes → liste de chunks prêts pour ChromaDB.

    Chaque item :
    {
        "document": str,
        "metadata": {
            "source":      str,   # nom du fichier
            "page":        int,   # numéro de page
            "chunk_idx":   int,   # index dans la page
            "imported_at": str,   # datetime ISO UTC
        }
    }
    """
    pages = extract_pages(pdf_bytes)
    imported_at = datetime.utcnow().isoformat()
    all_chunks = []

    for page_data in pages:
        for idx, chunk in enumerate(_chunk_text(page_data["text"])):
            all_chunks.append({
                "document": chunk,
                "metadata": {
                    "source":      filename,
                    "page":        page_data["page_num"],
                    "chunk_idx":   idx,
                    "imported_at": imported_at,
                }
            })

    return all_chunks


def generate_ids(chunks: List[Dict], existing_ids: List[str]) -> List[str]:
    """
    Génère des IDs stables et uniques pour chaque chunk.
    Format : pdf__{slug}__{page}__c{chunk_idx}
    """
    ids = []
    existing_set = set(existing_ids)

    for c in chunks:
        meta = c["metadata"]
        slug = re.sub(r'[^a-zA-Z0-9]', '_', meta["source"])[:40]
        base = f"pdf__{slug}__p{meta['page']}__c{meta['chunk_idx']}"
        candidate = base
        suffix = 0
        while candidate in existing_set:
            suffix += 1
            candidate = f"{base}_{suffix}"
        ids.append(candidate)
        existing_set.add(candidate)

    return ids
