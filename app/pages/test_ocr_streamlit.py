import streamlit as st
import io
import time

# Import de votre moteur d'extraction local
try:
    from pdf_engine import extract_pages
except ImportError:
    st.error("Le fichier `pdf_engine.py` est introuvable. Assurez-vous de lancer ce script depuis le dossier `app/`.")
    st.stop()

st.set_page_config(page_title="PDF OCR Debugger", page_icon="🔍", layout="wide")

st.title("🔍 Débogueur OCR PDF (Test d'extraction)")

st.markdown("""
Cet outil utilise votre fichier `pdf_engine.py` local pour extraire le texte. 
Il est parfait pour tester l'impact de vos modifications (comme la désactivation du prétraitement OpenCV) sur la reconnaissance de PaddleOCR.
""")

uploaded_file = st.file_uploader("Chargez votre fichier PDF à tester", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    
    if st.button("Lancer l'extraction", type="primary"):
        start_time = time.time()
        
        with st.spinner("Extraction en cours (pdfplumber -> PaddleOCR)..."):
            try:
                # Appel direct à votre fonction modifiée
                pages = extract_pages(pdf_bytes)
                
                elapsed_time = time.time() - start_time
                st.success(f"Extraction terminée en {elapsed_time:.2f} secondes ! ({len(pages)} page(s) traitée(s))")
                
                # Affichage des résultats par page
                for p in pages:
                    with st.expander(f"📄 Page {p['page_num']}", expanded=True):
                        # On utilise un text_area pour faciliter le copier-coller et la lecture
                        st.text_area(
                            label=f"Texte brut extrait", 
                            value=p["text"], 
                            height=400,
                            key=f"page_{p['page_num']}",
                            label_visibility="collapsed"
                        )
                        
                        # Affichage de quelques métriques
                        st.caption(f"Nombre de caractères : {len(p['text'])}")
                        
            except Exception as e:
                st.error(f"Une erreur s'est produite lors de l'extraction : {e}")