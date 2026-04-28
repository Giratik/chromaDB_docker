import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import os
import json
import pandas as pd

try:
    from pdf_engine import pdf_to_chunks, generate_ids, OCR_AVAILABLE
except ImportError:
    OCR_AVAILABLE = False
    def pdf_to_chunks(*a, **k): return []
    def generate_ids(*a, **k): return []

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CHROMA_HOST         = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT         = int(os.environ.get("CHROMA_PORT", 8000))
OLLAMA_HOST         = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
COLLECTION_NAME     = "base_connaissances_globale_acronymes"
PDF_COLLECTION_NAME = "documents_pdf"

st.set_page_config(page_title="ChromaDB Manager", page_icon="🧠", layout="wide")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --surface: #e2e2e2;
    --surface2: #e2e2e282;
    --border: #2a2f40;
    --accent: #5b8dee;
    --accent2: #38e8b0;
    --danger: #e85b5b;
    --warn: #e8b05b;
    --text: #dde3f0;
    --text2: #7a84a0;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg)!important;color:var(--text)!important;}
.stApp{background-color:var(--bg);}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
h1,h2,h3{font-family:'Space Mono',monospace;}
h1{color:var(--accent2)!important;letter-spacing:-1px;}
h2{color:var(--accent)!important;font-size:1.1rem!important;}
.metric-card{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:1.2rem 1.5rem;text-align:center;}
.metric-val{font-family:'Space Mono',monospace;font-size:2.2rem;color:var(--accent2);}
.metric-label{color:var(--text2);font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;}
.stButton>button{background:var(--surface2)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:6px!important;font-family:'Space Mono',monospace!important;font-size:0.78rem!important;transition:all 0.15s;}
.stButton>button:hover{border-color:var(--accent)!important;color:var(--accent)!important;}
.stTextInput>div>div>input,.stTextArea>div>div>textarea,.stSelectbox>div>div{background:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:6px!important;}
.stDataFrame{border:1px solid var(--border)!important;border-radius:8px;}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.72rem;font-family:'Space Mono',monospace;}
.badge-ok{background:#1a3a2a;color:var(--accent2);border:1px solid #2a5a3a;}
.badge-err{background:#3a1a1a;color:var(--danger);border:1px solid #5a2a2a;}
hr{border-color:var(--border)!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{color:var(--text2)!important;}
.stTabs [aria-selected="true"]{color:var(--accent2)!important;}
</style>
""", unsafe_allow_html=True)

# ─── CLIENT ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

@st.cache_resource
def get_embedding_fn():
    return embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_HOST + "/api/embeddings",
        model_name="nomic-embed-text"
    )

def get_collection(client, name):
    return client.get_or_create_collection(name=name, embedding_function=get_embedding_fn())

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def fetch_all_lexique(collection):
    result = collection.get(include=["documents", "metadatas"])
    rows = []
    for i, doc_id in enumerate(result["ids"]):
        meta = result["metadatas"][i] if result["metadatas"] else {}
        rows.append({
            "id": doc_id,
            "acronyme": meta.get("acronyme", ""),
            "signification": meta.get("signification", ""),
            "document": result["documents"][i] if result["documents"] else "",
        })
    return rows

def fetch_all_pdf(collection):
    result = collection.get(include=["documents", "metadatas"])
    rows = []
    for i, doc_id in enumerate(result["ids"]):
        meta = result["metadatas"][i] if result["metadatas"] else {}
        rows.append({
            "id": doc_id,
            "source": meta.get("source", ""),
            "page": meta.get("page", ""),
            "chunk_idx": meta.get("chunk_idx", ""),
            "imported_at": meta.get("imported_at", ""),
            "extrait": (result["documents"][i][:120] + "…") if result["documents"] else "",
        })
    return rows

def build_document(acronyme, signification):
    return (f"{acronyme} {acronyme.lower()} : {signification}. "
            f"Également appelé {signification.lower()}.")

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 ChromaDB Manager")
    st.markdown(f"<span class='badge badge-ok'>● CHROMA {CHROMA_HOST}:{CHROMA_PORT}</span>", unsafe_allow_html=True)
    ocr_badge = "badge-ok" if OCR_AVAILABLE else "badge-err"
    ocr_label = "OCR actif" if OCR_AVAILABLE else "OCR inactif"
    st.markdown(f"<span class='badge {ocr_badge}'>● {ocr_label}</span>", unsafe_allow_html=True)
    st.markdown("---")

    section = st.radio("Section", ["📚 Lexique", "📄 Documents PDF"], label_visibility="collapsed")
    st.markdown("---")

    if section == "📚 Lexique":
        page = st.radio("Navigation — Lexique", [
            "📊 Dashboard", "📋 Parcourir", "➕ Ajouter",
            "✏️ Modifier", "🗑️ Supprimer", "📥 Import JSON", "📤 Export"
        ], label_visibility="collapsed")
        st.markdown(f"<div style='color:var(--text2);font-size:0.75rem;'>Collection<br><code style='color:var(--accent)'>{COLLECTION_NAME}</code></div>", unsafe_allow_html=True)
    else:
        page = st.radio("Navigation — PDF", [
            "📊 Dashboard PDF", "📋 Parcourir PDF",
            "📥 Importer PDF", "🗑️ Supprimer PDF"
        ], label_visibility="collapsed")
        st.markdown(f"<div style='color:var(--text2);font-size:0.75rem;'>Collection<br><code style='color:var(--accent)'>{PDF_COLLECTION_NAME}</code></div>", unsafe_allow_html=True)

# ─── CONNEXION ────────────────────────────────────────────────────────────────
try:
    client = get_client()
except Exception as e:
    st.error(f"❌ Connexion ChromaDB impossible : {e}")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION LEXIQUE
# ══════════════════════════════════════════════════════════════════════════════

if section == "📚 Lexique":
    collection = get_collection(client, COLLECTION_NAME)

    # ── Dashboard ─────────────────────────────────────────────────────────────
    if page == "📊 Dashboard":
        st.title("Dashboard — Lexique")
        total = collection.count()
        all_docs = fetch_all_lexique(collection)
        c1, c2, c3 = st.columns(3)
        for col, val, label in [
            (c1, total, "Entrées totales"),
            (c2, len(set(r["acronyme"] for r in all_docs)), "Acronymes uniques"),
            (c3, COLLECTION_NAME[:14] + "…", "Collection"),
        ]:
            col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("## Aperçu")
        if all_docs:
            st.dataframe(pd.DataFrame(all_docs)[["acronyme","signification"]].head(20), use_container_width=True, hide_index=True)
        else:
            st.info("Base vide.")

    # ── Parcourir ─────────────────────────────────────────────────────────────
    elif page == "📋 Parcourir":
        st.title("Parcourir le lexique")
        all_docs = fetch_all_lexique(collection)
        search = st.text_input("🔍 Filtrer", placeholder="acronyme ou signification…")
        if search:
            s = search.lower()
            all_docs = [r for r in all_docs if s in r["acronyme"].lower() or s in r["signification"].lower()]
        st.caption(f"{len(all_docs)} entrée(s)")
        if all_docs:
            st.dataframe(pd.DataFrame(all_docs)[["id","acronyme","signification"]], use_container_width=True, hide_index=True)

    # ── Ajouter ───────────────────────────────────────────────────────────────
    elif page == "➕ Ajouter":
        st.title("Ajouter une entrée")
        with st.form("form_add"):
            acro = st.text_input("Acronyme *")
            sig  = st.text_input("Signification *")
            ok   = st.form_submit_button("➕ Ajouter")
        if ok:
            if not acro or not sig:
                st.warning("Les deux champs sont obligatoires.")
            else:
                existing = collection.get(where={"acronyme": acro.upper()})
                if existing["ids"]:
                    st.error(f"❌ {acro.upper()} existe déjà.")
                else:
                    all_ids = collection.get()["ids"]
                    new_id = str(max([int(i) for i in all_ids if i.isdigit()], default=-1) + 1)
                    collection.add(
                        documents=[build_document(acro.upper(), sig)],
                        metadatas=[{"acronyme": acro.upper(), "signification": sig}],
                        ids=[new_id]
                    )
                    st.success(f"✅ {acro.upper()} ajouté (id {new_id}).")

    # ── Modifier ──────────────────────────────────────────────────────────────
    elif page == "✏️ Modifier":
        st.title("Modifier une entrée")
        all_docs = fetch_all_lexique(collection)
        if not all_docs:
            st.info("Base vide.")
        else:
            opts = {f"{r['acronyme']} — {r['signification'][:50]}": r for r in all_docs}
            chosen = opts[st.selectbox("Sélectionner", list(opts.keys()))]
            with st.form("form_edit"):
                new_acro = st.text_input("Acronyme", value=chosen["acronyme"])
                new_sig  = st.text_input("Signification", value=chosen["signification"])
                ok = st.form_submit_button("💾 Sauvegarder")
            if ok:
                collection.update(
                    ids=[chosen["id"]],
                    documents=[build_document(new_acro.upper(), new_sig)],
                    metadatas=[{"acronyme": new_acro.upper(), "signification": new_sig}]
                )
                st.success(f"✅ Entrée `{chosen['id']}` mise à jour.")

    # ── Supprimer ─────────────────────────────────────────────────────────────
    elif page == "🗑️ Supprimer":
        st.title("Supprimer")
        all_docs = fetch_all_lexique(collection)
        if not all_docs:
            st.info("Base vide.")
        else:
            t1, t2 = st.tabs(["Supprimer une entrée", "⚠️ Vider la collection"])
            with t1:
                opts = {f"{r['acronyme']} — {r['signification'][:50]}": r for r in all_docs}
                chosen = opts[st.selectbox("Sélectionner", list(opts.keys()))]
                if st.button("🗑️ Supprimer"):
                    collection.delete(ids=[chosen["id"]])
                    st.success(f"✅ {chosen['acronyme']} supprimé.")
            with t2:
                st.warning(f"Supprime toutes les {collection.count()} entrées.")
                if st.text_input("Tapez CONFIRMER") == "CONFIRMER":
                    if st.button("💣 Vider"):
                        collection.delete(ids=collection.get()["ids"])
                        st.success("Collection vidée.")

    # ── Import JSON ───────────────────────────────────────────────────────────
    elif page == "📥 Import JSON":
        st.title("Import JSON")
        mode = st.radio("Mode", ["Fusionner", "Remplacer tout"])
        uploaded = st.file_uploader("lexique.json", type=["json"])
        if uploaded:
            data = json.load(uploaded)
            st.success(f"{len(data)} entrées détectées.")
            st.dataframe(pd.DataFrame(data).head(10), use_container_width=True, hide_index=True)
            if st.button("📥 Lancer l'import"):
                if mode == "Remplacer tout":
                    ids = collection.get()["ids"]
                    if ids:
                        collection.delete(ids=ids)
                existing_ids = collection.get()["ids"]
                max_id = max([int(i) for i in existing_ids if i.isdigit()], default=-1)
                added = skipped = 0
                prog = st.progress(0)
                for i, entry in enumerate(data):
                    acro = entry.get("acronyme","").upper()
                    sig  = entry.get("signification","")
                    if not acro or not sig:
                        skipped += 1; continue
                    if mode == "Fusionner" and collection.get(where={"acronyme": acro})["ids"]:
                        skipped += 1; prog.progress((i+1)/len(data)); continue
                    max_id += 1
                    collection.add(
                        documents=[build_document(acro, sig)],
                        metadatas=[{"acronyme": acro, "signification": sig}],
                        ids=[str(max_id)]
                    )
                    added += 1
                    prog.progress((i+1)/len(data))
                st.success(f"✅ {added} ajoutées, {skipped} ignorées.")

    # ── Export ────────────────────────────────────────────────────────────────
    elif page == "📤 Export":
        st.title("Exporter le lexique")
        all_docs = fetch_all_lexique(collection)
        if not all_docs:
            st.info("Base vide.")
        else:
            t1, t2 = st.tabs(["JSON", "CSV"])
            with t1:
                data = [{"acronyme": r["acronyme"], "signification": r["signification"]} for r in all_docs]
                st.download_button("⬇️ Télécharger JSON", json.dumps(data, ensure_ascii=False, indent=2).encode(), "lexique.json", "application/json")
            with t2:
                df = pd.DataFrame(all_docs)[["acronyme","signification"]]
                st.download_button("⬇️ Télécharger CSV", df.to_csv(index=False).encode(), "lexique.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION PDF
# ══════════════════════════════════════════════════════════════════════════════

else:
    pdf_col = get_collection(client, PDF_COLLECTION_NAME)

    # ── Dashboard PDF ─────────────────────────────────────────────────────────
    if page == "📊 Dashboard PDF":
        st.title("Dashboard — Documents PDF")
        total = pdf_col.count()
        all_docs = fetch_all_pdf(pdf_col)
        sources = list(set(r["source"] for r in all_docs))
        c1, c2, c3 = st.columns(3)
        for col, val, label in [
            (c1, total, "Chunks indexés"),
            (c2, len(sources), "Documents sources"),
            (c3, "nomic-embed-text", "Modèle"),
        ]:
            col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

        if sources:
            st.markdown("---")
            st.markdown("## Documents indexés")
            summary = []
            for src in sources:
                src_docs = [r for r in all_docs if r["source"] == src]
                pages    = sorted(set(r["page"] for r in src_docs))
                summary.append({
                    "Fichier": src,
                    "Chunks": len(src_docs),
                    "Pages": f"{min(pages)}–{max(pages)}",
                    "Importé le": src_docs[0]["imported_at"][:10] if src_docs else "",
                })
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    # ── Parcourir PDF ─────────────────────────────────────────────────────────
    elif page == "📋 Parcourir PDF":
        st.title("Parcourir les chunks PDF")
        all_docs = fetch_all_pdf(pdf_col)
        if not all_docs:
            st.info("Aucun document indexé.")
        else:
            sources = ["Tous"] + sorted(set(r["source"] for r in all_docs))
            sel_src = st.selectbox("Filtrer par document", sources)
            search  = st.text_input("🔍 Recherche dans le texte")

            filtered = all_docs
            if sel_src != "Tous":
                filtered = [r for r in filtered if r["source"] == sel_src]
            if search:
                filtered = [r for r in filtered if search.lower() in r["extrait"].lower()]

            st.caption(f"{len(filtered)} chunk(s)")
            if filtered:
                st.dataframe(
                    pd.DataFrame(filtered)[["source","page","chunk_idx","extrait"]],
                    use_container_width=True, hide_index=True
                )

    # ── Importer PDF ──────────────────────────────────────────────────────────
    elif page == "📥 Importer PDF":
        st.title("Importer des PDFs")

        if not OCR_AVAILABLE:
            st.warning("⚠️ OCR non disponible (tesseract absent). Seuls les PDFs à texte natif seront traités correctement.")

        uploaded_files = st.file_uploader(
            "Choisir un ou plusieurs PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            mode = st.radio(
                "Si le document est déjà indexé",
                ["Ignorer (garder l'existant)", "Réindexer (remplacer)"]
            )

            chunk_size = st.slider("Taille des chunks (caractères)", 200, 1000, 500, 50)
            overlap    = st.slider("Overlap (caractères)", 0, 200, 100, 10)

            st.markdown("---")
            if st.button("📥 Lancer l'indexation"):
                existing_ids = pdf_col.get()["ids"]
                all_sources  = set(pdf_col.get(include=["metadatas"])["metadatas"][i].get("source","")
                                   for i in range(len(existing_ids))) if existing_ids else set()

                for uploaded in uploaded_files:
                    filename = uploaded.name
                    st.markdown(f"**⏳ {filename}**")

                    # Mode réindexer : supprimer les chunks existants du fichier
                    if mode.startswith("Réindexer") and filename in all_sources:
                        to_del = pdf_col.get(where={"source": filename})["ids"]
                        if to_del:
                            pdf_col.delete(ids=to_del)
                            st.caption(f"  → {len(to_del)} anciens chunks supprimés.")
                    elif mode.startswith("Ignorer") and filename in all_sources:
                        st.caption(f"  → déjà indexé, ignoré.")
                        continue

                    pdf_bytes = uploaded.read()
                    with st.spinner("Extraction du texte…"):
                        chunks = pdf_to_chunks(pdf_bytes, filename)

                    if not chunks:
                        st.warning(f"  → Aucun texte extrait de {filename}.")
                        continue

                    ids = generate_ids(chunks, pdf_col.get()["ids"])

                    prog = st.progress(0)
                    batch_size = 50
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i+batch_size]
                        pdf_col.add(
                            documents=[c["document"] for c in batch],
                            metadatas=[c["metadata"] for c in batch],
                            ids=ids[i:i+batch_size]
                        )
                        prog.progress(min((i + batch_size) / len(chunks), 1.0))

                    st.success(f"  ✅ {len(chunks)} chunks indexés ({len(set(c['metadata']['page'] for c in chunks))} pages).")

    # ── Supprimer PDF ─────────────────────────────────────────────────────────
    elif page == "🗑️ Supprimer PDF":
        st.title("Supprimer des documents PDF")
        all_docs = fetch_all_pdf(pdf_col)
        if not all_docs:
            st.info("Aucun document indexé.")
        else:
            sources = sorted(set(r["source"] for r in all_docs))
            t1, t2 = st.tabs(["Supprimer un document", "⚠️ Vider la collection PDF"])

            with t1:
                sel = st.selectbox("Document à supprimer", sources)
                src_chunks = [r for r in all_docs if r["source"] == sel]
                st.caption(f"{len(src_chunks)} chunks pour ce document.")
                if st.button(f"🗑️ Supprimer « {sel} »"):
                    ids_to_del = [r["id"] for r in src_chunks]
                    pdf_col.delete(ids=ids_to_del)
                    st.success(f"✅ {len(ids_to_del)} chunks supprimés.")

            with t2:
                st.warning(f"Supprime tous les {pdf_col.count()} chunks de la collection PDF.")
                if st.text_input("Tapez CONFIRMER") == "CONFIRMER":
                    if st.button("💣 Vider"):
                        pdf_col.delete(ids=pdf_col.get()["ids"])
                        st.success("Collection PDF vidée.")
