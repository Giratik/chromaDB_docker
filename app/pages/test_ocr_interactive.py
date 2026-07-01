import streamlit as st
import io
import cv2
import numpy as np
from PIL import Image
import pdfplumber

st.set_page_config(page_title="OCR Tuner", page_icon="🎛️", layout="wide")

# Mise en cache du modèle pour éviter de le recharger à chaque mouvement de slider
@st.cache_resource
def load_paddle():
    from paddleocr import PaddleOCR
    # Utilisation du GPU localement pour une mise à jour fluide des sliders
    return PaddleOCR(use_angle_cls=True, lang="fr", use_gpu=True, show_log=False)

try:
    paddle_model = load_paddle()
except Exception as e:
    st.error(f"Erreur lors du chargement de PaddleOCR : {e}")
    st.stop()

st.title("🎛️ Débogueur OCR Interactif")
st.markdown("Modifiez les paramètres dans la barre latérale. L'extraction est relancée en temps réel.")

# --- BARRE LATÉRALE ---
st.sidebar.header("⚙️ Paramètres")

st.sidebar.subheader("1. Rastérisation (PDF -> Image)")
# Changement des valeurs par défaut suite à vos tests
dpi_resolution = st.sidebar.slider("Résolution (DPI)", 50, 500, 150, step=10)
force_ocr = st.sidebar.checkbox("Forcer l'OCR (Ignorer texte natif)", value=True)

st.sidebar.subheader("2. Prétraitement OpenCV")
# Désactivé par défaut suite à vos tests
use_preprocess = st.sidebar.checkbox("Activer le prétraitement", value=False)
block_size = st.sidebar.slider("Binarisation: Block Size", 3, 31, 11, step=2, help="Doit être impair.")
c_val = st.sidebar.slider("Binarisation: C", 0, 15, 2)
dilate_iter = st.sidebar.slider("Dilatation (Itérations)", 0, 5, 1, help="0 désactive l'épaississement.")

st.sidebar.subheader("3. Filtrage OCR")
conf_threshold = st.sidebar.slider("Seuil de confiance minimum", 0.0, 1.0, 0.50, step=0.01)

# --- FONCTION LOCALE DE PRÉTRAITEMENT ---
def debug_preprocess(img):
    img_np = np.array(img.convert("L"))
    
    # Binarisation adaptative
    binary = cv2.adaptiveThreshold(
        img_np, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, c_val
    )
    
    # Déskew
    coords = np.column_stack(np.where(binary > 0))
    if coords.size > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = -(90 + angle)
        else: angle = -angle
        (h, w) = binary.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
    # Dilatation
    if dilate_iter > 0:
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=dilate_iter)
        
    return Image.fromarray(cv2.bitwise_not(binary))

uploaded_file = st.file_uploader("Chargez le document problématique", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        num_pages = len(pdf.pages)
        
        st.sidebar.subheader("📄 Sélection de la page")
        page_number = st.sidebar.number_input(
            f"Page à analyser (sur {num_pages})", 
            min_value=1, 
            max_value=num_pages, 
            value=1,
            step=1
        )
        
        # On sélectionne la page demandée
        page = pdf.pages[page_number - 1]
        
        col1, col2 = st.columns([1, 1])
        
        native_text = (page.extract_text() or "").strip()
        
        with st.spinner("Rastérisation..."):
            img = page.to_image(resolution=dpi_resolution).original
        
        if use_preprocess:
            processed_img = debug_preprocess(img)
        else:
            processed_img = img
            
        with col1:
            st.subheader(f"Image analysée (Page {page_number})")
            st.image(processed_img, use_column_width=True)
            
        with st.spinner("Analyse PaddleOCR..."):
            arr = np.array(processed_img.convert("RGB"))
            result = paddle_model.ocr(arr, cls=True)
            
            accepted_lines = []
            rejected_lines = []
            
            if result and result[0]:
                for line in result[0]:
                    text, confidence = line[1]
                    if confidence >= conf_threshold:
                        accepted_lines.append(f"[{confidence:.2f}] {text}")
                    else:
                        rejected_lines.append(f"[{confidence:.2f}] {text}")
                        
        with col2:
            st.subheader("Texte Extrait")
            
            if not force_ocr and len(native_text) > 30:
                st.info("Le texte natif a été utilisé. Cochez 'Forcer l'OCR' à gauche pour bypasser.")
                
                # NOUVEAU: Hauteur dynamique basée sur le nombre de lignes
                native_lines = len(native_text.split('\n'))
                dynamic_height = max(150, native_lines * 24 + 40) # 24px par ligne + marge
                st.text_area("Texte Natif", value=native_text, height=dynamic_height)
                
            else:
                st.success(f"{len(accepted_lines)} lignes acceptées")
                clean_text = "\n".join([l.split("] ", 1)[1] for l in accepted_lines if "] " in l])
                
                # NOUVEAU: Hauteur dynamique basée sur le nombre de lignes (OCR)
                ocr_lines = len(clean_text.split('\n'))
                dynamic_height = max(150, ocr_lines * 24 + 40) # 24px par ligne + marge
                st.text_area("Texte Final (OCR)", value=clean_text, height=dynamic_height)
                
                if rejected_lines:
                    st.warning(f"⚠️ {len(rejected_lines)} ligne(s) rejetée(s) car confiance < {conf_threshold}")
                    for r in rejected_lines:
                        st.caption(f"❌ {r}")


