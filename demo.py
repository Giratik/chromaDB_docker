"""
demo.py — Client de démonstration ChromaDB
──────────────────────────────────────────
Script standalone : aucune dépendance au projet chroma-manager.

Prérequis :
    pip install streamlit chromadb ollama

Lancement :
    streamlit run demo.py

Variables d'environnement (optionnelles) :
    CHROMA_HOST   → défaut : localhost
    CHROMA_PORT   → défaut : 8100
    OLLAMA_HOST   → défaut : http://localhost:11434
"""

import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CHROMA_HOST  = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT  = int(os.environ.get("CHROMA_PORT", 8100))
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Registre des collections disponibles
# → pour ajouter les PDFs plus tard : décommenter la ligne documents_pdf


# ─── PAGE ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ChromaDB Demo", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500&display=swap');
:root {
    --surface: #e2e2e2;
    --surface2: #e2e2e282;
    --border: #2a2f40;
    --accent: #5b8dee;
    --accent2: #38e8b0;
    --danger: #e85b5b;
    --warn: #e8b05b;
    --text2: #7a84a0;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg)!important;color:var(--text)!important;}
.stApp{background:var(--bg);}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
h1{font-family:'Space Mono',monospace;color:var(--accent2)!important;letter-spacing:-1px;}
h3{font-family:'Space Mono',monospace;color:var(--accent)!important;font-size:0.95rem!important;}
.stTextInput>div>div>input{background:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:6px!important;}
.stSelectbox>div>div{background:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--text)!important;}
.stButton>button{background:var(--surface2)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:6px!important;font-family:'Space Mono',monospace!important;font-size:0.78rem!important;}
.stButton>button:hover{border-color:var(--accent)!important;color:var(--accent)!important;}
.result-card{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:1rem 1.2rem;margin-bottom:0.75rem;}
.result-card.exact{border-left:3px solid var(--accent2);}
.result-card.semantic{border-left:3px solid var(--accent);}
.result-card.pdf{border-left:3px solid var(--warn, #e8b05b);}
.tag{display:inline-block;padding:1px 8px;border-radius:12px;font-size:0.7rem;font-family:'Space Mono',monospace;margin-right:4px;}
.tag-exact{background:#1a3a2a;color:var(--accent2);}
.tag-semantic{background:#1a2a3a;color:var(--accent);}
.tag-pdf{background:#3a2a1a;color:#e8b05b;}
.score{font-family:'Space Mono',monospace;font-size:0.75rem;color:var(--text2);}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.72rem;font-family:'Space Mono',monospace;}
.badge-ok{background:#1a3a2a;color:var(--accent2);border:1px solid #2a5a3a;}
.badge-err{background:#3a1a1a;color:var(--danger);border:1px solid #5a2a2a;}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)


# ─── CONNEXION ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

@st.cache_resource
def get_embedding_fn():
    return embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_HOST + "/api/embeddings",
        model_name="embeddinggemma"
    )

@st.cache_resource
def get_collection(collection_name: str):
    client = get_client()
    return client.get_collection(
        name=collection_name,
        embedding_function=get_embedding_fn()
    )


# ─── MOTEUR DE RECHERCHE ──────────────────────────────────────────────────────

def search_lexique(collection, query: str, n_results: int = 5, seuil: float = 0.6):
    """
    Reproduit la logique de newer_rag_engine.py :
      1. Match exact sur l'acronyme
      2. Match exact sur la signification
      3. Fallback sémantique vectoriel avec seuil de distance
    Retourne une liste de résultats normalisés.
    """
    results = []

    # 1. Match exact acronyme
    exact = collection.get(where={"acronyme": query.upper()})
    if exact["ids"]:
        for i, doc_id in enumerate(exact["ids"]):
            meta = exact["metadatas"][i]
            results.append({
                "id":           doc_id,
                "acronyme":     meta.get("acronyme", ""),
                "signification":meta.get("signification", ""),
                "document":     exact["documents"][i],
                "type":         "exact_acronyme",
                "distance":     0.0,
            })
        return results

    # 2. Match exact signification
    exact_sig = collection.get(where={"signification": query})
    if exact_sig["ids"]:
        for i, doc_id in enumerate(exact_sig["ids"]):
            meta = exact_sig["metadatas"][i]
            results.append({
                "id":           doc_id,
                "acronyme":     meta.get("acronyme", ""),
                "signification":meta.get("signification", ""),
                "document":     exact_sig["documents"][i],
                "type":         "exact_signification",
                "distance":     0.0,
            })
        return results

    # 3. Recherche sémantique
    vecto = collection.query(query_texts=[query], n_results=n_results)
    for i, doc_id in enumerate(vecto["ids"][0]):
        dist = vecto["distances"][0][i]
        if dist <= seuil:
            meta = vecto["metadatas"][0][i]
            results.append({
                "id":           doc_id,
                "acronyme":     meta.get("acronyme", ""),
                "signification":meta.get("signification", ""),
                "document":     vecto["documents"][0][i],
                "type":         "semantique",
                "distance":     dist,
            })

    return results


def search_pdf(collection, query: str, n_results: int = 5, seuil: float = 0.6):
    vecto = collection.query(query_texts=[query], n_results=n_results)
    # DEBUG — à retirer ensuite
    st.write("distances brutes :", vecto["distances"])
    st.write("nb résultats bruts :", len(vecto["ids"][0]))
    results = []
    for i, doc_id in enumerate(vecto["ids"][0]):
        dist = vecto["distances"][0][i]
        if dist <= seuil:
            meta = vecto["metadatas"][0][i]
            results.append({
                "id":          doc_id,
                "source":      meta.get("source", ""),
                "page":        meta.get("page", ""),
                "chunk_idx":   meta.get("chunk_idx", ""),
                "document":    vecto["documents"][0][i],
                "type":        "pdf",
                "distance":    dist,
            })
    return results


def route_search(query: str, col_type: str, collection, n_results: int, seuil: float):
    """
    Routeur : choisit la bonne fonction de recherche selon le type de collection.
    → Ajouter un nouveau type ici quand de nouvelles collections arrivent.
    """
    if col_type == "lexique":
        return search_lexique(collection, query, n_results, seuil)
    elif col_type == "pdf":
        return search_pdf(collection, query, n_results, seuil)
    return []


# ─── AFFICHAGE DES RÉSULTATS ──────────────────────────────────────────────────

def render_result_lexique(r: dict):
    tag_cls  = "tag-exact" if "exact" in r["type"] else "tag-semantic"
    card_cls = "exact" if "exact" in r["type"] else "semantic"
    label    = "✦ exact" if "exact" in r["type"] else "~ sémantique"
    score    = "" if r["distance"] == 0.0 else f'<span class="score">distance : {r["distance"]:.4f}</span>'
    st.markdown(f"""
    <div class="result-card {card_cls}">
        <span class="tag {tag_cls}">{label}</span> {score}<br>
        <strong style="font-size:1.2rem">{r['acronyme']}</strong>
        &nbsp;→&nbsp; {r['signification']}
        <div style="color:var(--text2);font-size:0.78rem;margin-top:0.4rem">{r['document']}</div>
    </div>
    """, unsafe_allow_html=True)


def render_result_pdf(r: dict):
    score = f'<span class="score">distance : {r["distance"]:.4f}</span>'
    st.markdown(f"""
    <div class="result-card pdf">
        <span class="tag tag-pdf">📄 PDF</span> {score}<br>
        <strong>{r['source']}</strong> — page {r['page']}
        <div style="color:var(--text2);font-size:0.78rem;margin-top:0.4rem">{r['document']}</div>
    </div>
    """, unsafe_allow_html=True)


def render_results(results: list, col_type: str):
    if not results:
        st.info("Aucun résultat pour cette requête.")
        return
    for r in results:
        if col_type == "lexique":
            render_result_lexique(r)
        elif col_type == "pdf":
            render_result_pdf(r)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 ChromaDB Demo")

    # Statut connexion
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
        st.markdown(f"<span class='badge badge-ok'>● {CHROMA_HOST}:{CHROMA_PORT}</span>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"<span class='badge badge-err'>✕ {CHROMA_HOST}:{CHROMA_PORT}</span>", unsafe_allow_html=True)
        st.error(f"ChromaDB inaccessible : `{e}`")
        st.stop()

    st.markdown("---")

    # Collections chargées dynamiquement depuis le serveur
    available = [c.name for c in client.list_collections()]
    if not available:
        st.warning("Aucune collection trouvée dans ChromaDB.")
        st.stop()

    # Icône automatique selon le nom
    def _detect_type(name):
        """Détecte le type en lisant une entrée réelle de la collection."""
        try:
            sample = client.get_collection(name).get(limit=1, include=["metadatas"])
            if not sample["metadatas"]:
                return "lexique"
            meta = sample["metadatas"][0]
            if "acronyme" in meta:
                return "lexique"
            if "source" in meta and "page" in meta:
                return "pdf"
        except Exception:
            pass
        return "lexique"

    def _label(name):
        t = _detect_type(name)
        if t == "pdf":      return f"📄 {name}"
        if t == "lexique":  return f"📚 {name}"
        return f"🗂️ {name}"

    col_name = st.radio("Collection", available, format_func=_label, label_visibility="collapsed")
    col_cfg  = {
        "name": col_name,
        "type": _detect_type(col_name),
        "description": f"Collection `{col_name}`",
    }
    st.markdown("---")
    st.markdown("**Paramètres**")
    n_results = st.slider("Nb résultats max", 1, 20, 5)
    seuil     = st.slider("Seuil distance sémantique", 0.1, 1.0, 0.6, 0.05,
                           help="Plus la valeur est basse, plus la recherche est stricte.")

    st.markdown("---")
    st.markdown(f"<div style='color:var(--text2);font-size:0.75rem;'>Collection<br><code style='color:var(--accent)'>{col_cfg['name']}</code></div>", unsafe_allow_html=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.title("Démo — Recherche ChromaDB")
st.markdown(f"*{col_cfg['description']}*")

# Chargement collection
try:
    collection = get_collection(col_cfg["name"])
    st.caption(f"{collection.count()} entrées dans la collection.")
except Exception as e:
    st.error(f"Collection introuvable : **{col_cfg['name']}**\n\n`{e}`")
    st.stop()

st.markdown("---")

# Champ de recherche
query = st.text_input(
    "Requête",
    placeholder="Ex: CODIR, comité de direction, réunion stratégique…",
    label_visibility="collapsed"
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    search_clicked = st.button("🔍 Rechercher")

if search_clicked and query.strip():
    with st.spinner("Recherche en cours…"):
        results = route_search(
            query.strip(),
            col_cfg["type"],
            collection,
            n_results,
            seuil
        )

    st.markdown(f"### {len(results)} résultat(s) pour « {query} »")
    render_results(results, col_cfg["type"])

elif search_clicked and not query.strip():
    st.warning("Entrez une requête.")