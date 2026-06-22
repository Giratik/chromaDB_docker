# ─────────────────────────────────────────────────────────────────────────────
# Base CUDA 11.8 + cuDNN 8 — compatible PaddlePaddle-GPU 2.6.x
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── Dépendances système ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-distutils \
    build-essential \
    curl \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    # tesseract (fallback OCR)
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# ── PaddlePaddle GPU (wheel CUDA 11.8) ───────────────────────────────────────
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir paddlepaddle-gpu==2.6.2 -f https://www.paddlepaddle.org.cn/whl/cu118/stable.html
RUN pip install --no-cache-dir paddleocr

# ── Reste des dépendances ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='fr', use_angle_cls=True, use_gpu=False)"

COPY main.py .
COPY pdf_engine.py .
COPY ee.pdf .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
