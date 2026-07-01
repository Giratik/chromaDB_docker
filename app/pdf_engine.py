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
  - Re-ranking par fraîcheur post-retrieval
  - Formatage des chunks avec signal temporel visible pour le LLM
"""

import re
import io
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber
from PIL import Image

# ─── OCR : PaddleOCR (prioritaire) ────────────────────────────────────────────
try:
    from paddleocr import PaddleOCR
    _paddle = PaddleOCR(
    use_angle_cls=True,
    lang="fr",
    use_gpu=True,
    show_log=False,  # Désactive les logs pour éviter le spam


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










# pré traitement

def _preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Prétraite une image PIL pour améliorer l'OCR :
    - Conversion en niveaux de gris
    - Binarisation adaptative (seuillage)
    - Déskew (correction de rotation)
    - Nettoyage (dilation/érosion pour fusionner les caractères)
    """
    import cv2
    import numpy as np

    # Convertir en numpy array (niveaux de gris)
    img_np = np.array(img.convert("L"))

    # Binarisation adaptative (meilleure pour les documents scannés)
    binary = cv2.adaptiveThreshold(
        img_np, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Déskew (correction de rotation)
    coords = np.column_stack(np.where(binary > 0))
    if coords.size > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = binary.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        binary = cv2.warpAffine(
            binary, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    # Nettoyage : dilation pour fusionner les caractères proches
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Retourner en PIL Image
    return Image.fromarray(cv2.bitwise_not(binary))  # Inverser pour fond blanc



# ─── EXTRACTION ────────────────────────────────────────────────────────────────

def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convertit une image PIL en numpy array RGB."""
    return np.array(img.convert("RGB"))


def _ocr_paddle(img: Image.Image) -> str:
    """
    OCR via PaddleOCR avec prétraitement et post-traitement.
    - Corrige l'orientation (use_angle_cls=True)
    - Fusionne les lignes segmentées
    - Nettoie les erreurs de caractères
    """
    # 1. Prétraitement
    #processed_img = _preprocess_image_for_ocr(img)
    arr = _pil_to_numpy(img)

    # 2. OCR avec Paddle
    result = _paddle.ocr(arr, cls=True)
    lines = []

    if result and result[0]:
        # 3. Extraction et filtrage par confiance
        for line in result[0]:
            text, confidence = line[1]
            if confidence >= 0.5:  # Seuil de confiance
                lines.append(text)

    # 4. Post-traitement
    full_text = "\n".join(lines)

    # Corriger les erreurs courantes de PaddleOCR
    full_text = (
        full_text
        .replace("ä", "à")
        .replace("á", "à")
        .replace("xuvre", "œuvre")
        .replace("I'", "l'")
        .replace("critére", "critère")
        .replace("hirarchique", "hiérarchique")
        .replace("täches", "tâches")
        .replace(" á ", " à ")
        .replace(" ä ", " à ")
        .replace("|", "I")
    )

    # Fusionner les lignes trop proches (éviter les coupures)
    full_text = re.sub(r'\n([a-z])', r' \1', full_text)  # "mot\nsuivant" → "mot suivant"
    full_text = re.sub(r'(\w)\n(\w)', r'\1\2', full_text)  # "mot\nsuivant" → "motsuivant" (à affiner)

    # Nettoyage final
    full_text = re.sub(r' {2,}', ' ', full_text).strip()

    return full_text


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

    # Si le texte natif est suffisant ET contient des mots valides (pas juste des métadonnées)
    if len(text) >= MIN_NATIVE_CHARS and re.search(r'[a-zA-Zàâäéèêëîïôöùûüÿæœç]{10,}', text):
        return text

    # Rasteriser la page pour l'OCR
    img: Optional[Image.Image] = None
    try:
        img = page.to_image(resolution=150).original
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
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                raw = _extract_page_text(page)
                text = re.sub(r'\n{3,}', '\n\n', raw)
                text = re.sub(r' {2,}', ' ', text).strip()
                if text:
                    pages.append({"page_num": i, "text": text})
            except Exception as e:
                print(f"⚠️ Erreur sur la page {i}: {e}")
                continue  # Passe à la page suivante
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
            for punct in ['. ', '! ', '? ', '.\n', ', ']:
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

def _date_str_to_ts(date_str: str) -> float:
    """
    Convertit une date ISO (YYYY-MM-DD ou YYYY-MM-DDTHH:MM:SS) en timestamp UTC.
    Retourne 0.0 si la chaîne est vide ou invalide.
    """
    if not date_str:
        return 0.0
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            dt = datetime.strptime(date_str[:len(fmt) + 2].strip(), fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return 0.0



def pdf_to_chunks(
    pdf_bytes: bytes,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    doc_date: str = "",
    source_url: str = "", # ⬅️ NOUVEAU PARAMÈTRE
) -> List[Dict[str, Any]]:
    """
    PDF bytes → liste de chunks prêts pour ChromaDB.

    Chaque item :
    {
        "document": str,
        "metadata": {
            "source":       str,
            "page":         int,
            "chunk_idx":    int,
            "imported_at":  str,    # ISO UTC
            "doc_date":     str,    # YYYY-MM-DD fourni à l'import
            "doc_date_ts":  float,  # timestamp unix de doc_date (0 si absent)
        }
    }
    """
    pages        = extract_pages(pdf_bytes)
    imported_at  = datetime.now(timezone.utc).isoformat()
    doc_date_ts  = _date_str_to_ts(doc_date)
    all_chunks   = []

    for page_data in pages:
        for idx, chunk in enumerate(_chunk_text(page_data["text"], chunk_size, overlap)):
            all_chunks.append({
                "document": chunk,
                "metadata": {
                    "source":      filename,
                    "page":        page_data["page_num"],
                    "chunk_idx":   idx,
                    "imported_at": imported_at,
                    "doc_date":    doc_date,
                    "doc_date_ts": doc_date_ts,
                    "source_url":  source_url, # ⬅️ NOUVELLE MÉTADONNÉE
                }
            })

    return all_chunks


def generate_ids(chunks: List[Dict], existing_ids: List[str]) -> List[str]:
    """IDs stables et uniques : pdf__{slug}__p{page}__c{chunk_idx}__{timestamp}"""
    ids = []
    existing_set = set(existing_ids)
    # Timestamp court pour garantir l'unicité lors d'une réindexation
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    for c in chunks:
        meta = c["metadata"]
        slug = re.sub(r'[^a-zA-Z0-9]', '_', meta["source"])[:40]
        base = f"pdf__{slug}__p{meta['page']}__c{meta['chunk_idx']}__{ts}"
        candidate = base
        suffix = 0
        while candidate in existing_set:
            suffix += 1
            candidate = f"{base}_{suffix}"
        ids.append(candidate)
        existing_set.add(candidate)

    return ids


# ─── POST-RETRIEVAL : RE-RANKING & FORMATAGE ───────────────────────────────────

def rerank_by_freshness(
    chunks: List[Dict[str, Any]],
    distances: Optional[List[float]] = None,
    similarity_weight: float = 0.65,
    freshness_weight: float = 0.35,
    max_age_days: int = 365 * 5,
) -> List[Dict[str, Any]]:
    """
    Re-classe les chunks ChromaDB par score combiné similarité + fraîcheur.

    chunks    : liste de dicts {"document", "metadata", optionnel "distance"}
    distances : distances cosinus retournées par ChromaDB (même ordre que chunks)
                Si None, on utilise chunk.get("distance", 0.5)
    Retourne la liste triée du plus pertinent au moins pertinent.
    """
    if not chunks:
        return chunks

    now_ts      = datetime.now(timezone.utc).timestamp()
    max_age_sec = max_age_days * 86400

    for i, c in enumerate(chunks):
        # ── Score similarité ────────────────────────────────────────────────
        dist = distances[i] if distances else c.get("distance", 0.5)
        # ChromaDB cosine distance ∈ [0, 2] ; 0 = identique
        sim_score = max(0.0, 1.0 - dist / 2.0)

        # ── Score fraîcheur ─────────────────────────────────────────────────
        meta          = c.get("metadata", {})
        doc_ts        = float(meta.get("doc_date_ts") or 0)
        imported_ts   = _date_str_to_ts(meta.get("imported_at", ""))
        # Priorité : date du document > date d'import
        ref_ts        = doc_ts if doc_ts > 0 else imported_ts
        age_sec       = max(0.0, now_ts - ref_ts)
        freshness     = max(0.0, 1.0 - age_sec / max_age_sec)

        c["_sim_score"]       = round(sim_score, 4)
        c["_freshness_score"] = round(freshness, 4)
        c["_rerank_score"]    = round(
            similarity_weight * sim_score + freshness_weight * freshness, 4
        )

    return sorted(chunks, key=lambda x: x["_rerank_score"], reverse=True)


def format_chunk_for_llm(chunk: Dict[str, Any], rank: int = 0) -> str:
    """
    Formate un chunk pour qu'il soit injecté dans le contexte LLM.
    La date et la source sont visibles dans le texte — le LLM peut
    s'en servir explicitement pour arbitrer les contradictions.

    Exemple de sortie :
    ╔ [1] rapport_tarifaire_2024.pdf · p.3 · 2024-06-01 ══════════════
    Le taux de TVA applicable est de 22 % à compter du 1er juin 2024.
    ═══════════════════════════════════════════════════════════════════
    """
    meta    = chunk.get("metadata", {})
    source  = meta.get("source", "inconnu")
    page    = meta.get("page", "?")
    date    = meta.get("doc_date") or meta.get("imported_at", "")[:10] or "date inconnue"
    text    = chunk.get("document", "").strip()
    label   = f"[{rank + 1}] {source} · p.{page} · {date}"
    bar     = "═" * max(0, 68 - len(label))

    return f"╔ {label} {bar}\n{text}\n"


def format_chunks_for_llm(chunks: List[Dict[str, Any]]) -> str:
    """
    Assemble tous les chunks formatés + avertissement si plusieurs dates détectées.
    """
    if not chunks:
        return ""

    parts = [format_chunk_for_llm(c, i) for i, c in enumerate(chunks)]

    # Avertissement de diversité temporelle
    dates = sorted(set(
        c.get("metadata", {}).get("doc_date", "")
        for c in chunks
        if c.get("metadata", {}).get("doc_date", "")
    ))
    if len(dates) > 1:
        warning = (
            "\n⚠️  CES EXTRAITS PROVIENNENT DE DOCUMENTS DE DATES DIFFÉRENTES "
            f"({', '.join(dates)}). "
            "En cas de contradiction, la source la plus récente fait foi.\n"
        )
        parts.insert(0, warning)

    return "\n".join(parts)