#/app/pdf_engine.py

"""
pdf_engine.py
─────────────
Extraction de texte depuis des PDFs :
  - Texte natif via pdfplumber
  - Pages scannées / mal orientées : PaddleOCR (GPU CUDA) avec correction
    automatique d'orientation (use_angle_cls=True)
  - Fallback : pytesseract si PaddleOCR indisponible
  - Chunking avec overlap, coupure aux frontières naturelles
"""

import re
import io
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

import pdfplumber
from PIL import Image

# ─── OCR : PaddleOCR (prioritaire) ────────────────────────────────────────────
try:
    from paddleocr import PaddleOCR
    _paddle = PaddleOCR(
        use_angle_cls=True,   # détecte et corrige l'orientation (0/90/180/270°)
        lang="fr",
        use_gpu=True,
        show_log=False,
    )
    PADDLE_AVAILABLE = True
except Exception:
    _paddle = None
    PADDLE_AVAILABLE = False

# ─── OCR : Tesseract (fallback) ───────────────────────────────────────────────
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

OCR_AVAILABLE = PADDLE_AVAILABLE or TESSERACT_AVAILABLE

# ─── PARAMÈTRES ────────────────────────────────────────────────────────────────
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 100
MIN_NATIVE_CHARS = 30   # seuil texte natif : en dessous → OCR


# ─── EXTRACTION ────────────────────────────────────────────────────────────────

def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convertit une image PIL en numpy array RGB."""
    return np.array(img.convert("RGB"))


def _ocr_paddle(img: Image.Image) -> str:
    """
    OCR via PaddleOCR.
    use_angle_cls=True corrige automatiquement les pages paysage/verticale.
    Retourne le texte concaténé avec scores de confiance filtrés.
    """
    arr = _pil_to_numpy(img)
    result = _paddle.ocr(arr, cls=True)
    lines = []
    if result and result[0]:
        for line in result[0]:
            text, confidence = line[1]
            if confidence >= 0.5:          # ignorer les détections douteuses
                lines.append(text)
    return "\n".join(lines)


def _ocr_tesseract(img: Image.Image) -> str:
    """
    OCR via tesseract avec détection d'orientation (OSD).
    Corrige la rotation avant la reconnaissance.
    """
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        angle = osd.get("rotate", 0)
        if angle != 0:
            img = img.rotate(-angle, expand=True)
    except Exception:
        pass   # OSD peut échouer sur les pages trop vides
    return pytesseract.image_to_string(img, lang="fra+eng").strip()


def _extract_page_text(page) -> str:
    """
    Extrait le texte d'une page pdfplumber.
    Si le résultat est insuffisant → OCR (PaddleOCR puis tesseract).
    """
    text = (page.extract_text() or "").strip()

    if len(text) >= MIN_NATIVE_CHARS:
        return text

    # Rasteriser la page pour l'OCR
    img: Optional[Image.Image] = None
    try:
        img = page.to_image(resolution=200).original
    except Exception:
        return text   # impossible de rasteriser, on garde le texte partiel

    if PADDLE_AVAILABLE:
        try:
            ocr_text = _ocr_paddle(img)
            if len(ocr_text) > len(text):
                return ocr_text
        except Exception:
            pass

    if TESSERACT_AVAILABLE:
        try:
            ocr_text = _ocr_tesseract(img)
            if len(ocr_text) > len(text):
                return ocr_text
        except Exception:
            pass

    return text


def extract_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Retourne [{page_num, text}] pour chaque page non vide du PDF."""
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = _extract_page_text(page)
            # Nettoyage
            text = re.sub(r'\n{3,}', '\n\n', raw)
            text = re.sub(r' {2,}', ' ', text).strip()
            if text:
                pages.append({"page_num": i, "text": text})
    return pages


# ─── CHUNKING ──────────────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Découpe en chunks de ~chunk_size caractères avec overlap.
    Priorité de coupure : paragraphe > phrase > espace.
    """
    chunks = []
    start  = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        if end >= length:
            chunks.append(text[start:].strip())
            break

        slice_ = text[start:end]
        cut    = -1

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

def pdf_to_chunks(
    pdf_bytes: bytes,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    PDF bytes → liste de chunks prêts pour ChromaDB.

    Chaque item :
    {
        "document": str,
        "metadata": {
            "source":      str,
            "page":        int,
            "chunk_idx":   int,
            "imported_at": str,   # ISO UTC
        }
    }
    """
    pages       = extract_pages(pdf_bytes)
    imported_at = datetime.utcnow().isoformat()
    all_chunks  = []

    for page_data in pages:
        for idx, chunk in enumerate(_chunk_text(page_data["text"], chunk_size, overlap)):
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
    """IDs stables et uniques : pdf__{slug}__p{page}__c{chunk_idx}"""
    ids         = []
    existing_set = set(existing_ids)

    for c in chunks:
        meta   = c["metadata"]
        slug   = re.sub(r'[^a-zA-Z0-9]', '_', meta["source"])[:40]
        base   = f"pdf__{slug}__p{meta['page']}__c{meta['chunk_idx']}"
        candidate = base
        suffix = 0
        while candidate in existing_set:
            suffix   += 1
            candidate = f"{base}_{suffix}"
        ids.append(candidate)
        existing_set.add(candidate)

    return ids
