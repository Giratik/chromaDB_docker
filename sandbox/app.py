"""
app.py — Interface Streamlit RAG (frontend uniquement)
──────────────────────────────────────────────────────
Toute la logique métier est dans backend.py.

Lancement :
    streamlit run app.py
"""

import streamlit as st
import backend as rag

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Integrated", page_icon="🤖", layout="wide")

# ─── STYLING ──────────────────────────────────────────────────────────────────
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
h2,h3{font-family:'Space Mono',monospace;color:var(--accent)!important;}
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


# ─── CLIENTS CACHÉS ───────────────────────────────────────────────────────────

@st.cache_resource
def get_chroma_client():
    return rag.make_chroma_client()

@st.cache_resource
def get_ollama_client():
    return rag.make_ollama_client()


# ─── COMPOSANTS UI ────────────────────────────────────────────────────────────

def render_chunk_card(chunk: dict):
    """Affiche une card pour un chunk dans le panneau de visualisation."""
    chunk_type = chunk["type"]
    source = chunk["source"]
    doc = chunk["document"]
    hybrid = chunk["hybrid_score"]
    vecto = chunk["vecto_distance"]
    bm25 = chunk["bm25_score"]
    doc_date = chunk.get("doc_date", "")
    rerank = chunk.get("rerank_score", 0.0)
    

    if chunk_type == "pdf":
        tag_cls, card_cls, label = "tag-pdf", "pdf", "📄 PDF"
    elif chunk_type == "lexique":
        tag_cls, card_cls, label = "tag-semantic", "semantic", "📚 Lexique"
    else:
        tag_cls, card_cls, label = "tag-semantic", "semantic", "📋 Document"

    date_badge = f"<span class='score'>📅 {doc_date}</span> " if doc_date else ""

    st.markdown(f"""
    <div class="result-card {card_cls}">
        <span class="tag {tag_cls}">{label}</span>
        {date_badge}<span class="score">H:{hybrid:.3f} | V:{vecto:.4f} | B:{bm25:.3f} | R:{rerank:.3f}</span><br>
        <strong>{source}</strong>
        <div style="color:var(--text2);font-size:0.78rem;margin-top:0.4rem;line-height:1.4;">{doc}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

def render_sidebar(chroma_client, ollama_client) -> dict:
    """Rend la sidebar et retourne la config sélectionnée."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration RAG")

        try:
            collections = rag.list_collections(chroma_client)
            models = rag.list_generative_models(ollama_client)
        except Exception as e:
            st.error(f"Erreur d'initialisation : {e}")
            st.stop()

        from config import CHROMA_HOST, CHROMA_PORT
        st.markdown(
            f"<span class='badge badge-ok'>● {CHROMA_HOST}:{CHROMA_PORT}</span>",
            unsafe_allow_html=True,
        )

        if not collections:
            st.warning("Aucune collection trouvée.")
            st.stop()
        if not models:
            st.warning("Aucun modèle génératif trouvé.")
            st.stop()

        selected_collection = st.selectbox("Collection ChromaDB", collections)
        selected_model = st.selectbox("Modèle LLM Ollama", models)

        try:
            collection = rag.get_collection(chroma_client, selected_collection)
            doc_dates = rag.list_doc_dates(collection)
        except Exception:
            doc_dates = []

        st.markdown("---")
        selected_doc_date = st.selectbox(
            "Filtrer par date du document",
            ["Toutes"] + doc_dates,
            help="Si une date est sélectionnée, seuls les chunks issus de documents de cette date seront recherchés.",
        )
        selected_doc_date = "" if selected_doc_date == "Toutes" else selected_doc_date

        st.markdown("---")
        n_results = st.slider("Chunks à injecter", 1, 500, 250)
        seuil = st.slider("Seuil de distance (cosine)", 0.1, 1.0, 0.7, 0.05)

        st.markdown("---")
        st.markdown("**🔬 Stratégie de recherche**")
        use_hyde = st.toggle("HyDE (réponse hypothétique)", value=True)
        use_expansion = st.toggle("Query expansion (synonymes)", value=True)
        alpha = st.slider("Vectoriel ← → BM25", 0.0, 1.0, 0.5, 0.05)

        st.markdown("---")
        st.markdown("**🎯 Reranking**")
        use_reranker = st.toggle("Reranker (bge-reranker-v2-gemma)", value=True)

        st.markdown("---")
        if st.button("🗑️ Effacer la conversation"):
            st.session_state.messages = []
            st.rerun()

    return {
        "collection": selected_collection,
        "model": selected_model,
        "doc_date_filter": selected_doc_date,
        "n_results": n_results,
        "seuil": seuil,
        "use_hyde": use_hyde,
        "use_expansion": use_expansion,
        "alpha": alpha,
        "use_reranker": use_reranker,
    }


# ─── COLONNE CHAT ─────────────────────────────────────────────────────────────

def render_chat(cfg: dict, chroma_client, ollama_client):
    st.markdown("### 💬 Conversation")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Posez une question sur vos documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                collection = rag.get_collection(chroma_client, cfg["collection"])
            except Exception as e:
                st.error(f"Impossible de charger la collection : {e}")
                return

            with st.status("🔍 Recherche dans les documents...", expanded=True) as status:
                contexts, sources, detailed_chunks = rag.retrieve_context_hybrid(
                    collection=collection,
                    query=prompt,
                    ollama_client=ollama_client,
                    model=cfg["model"],
                    n_results=cfg["n_results"],
                    seuil=cfg["seuil"],
                    alpha=cfg["alpha"],
                    use_hyde=cfg["use_hyde"],
                    use_expansion=cfg["use_expansion"],
                    use_reranker=cfg["use_reranker"],
                    doc_date_filter=cfg.get("doc_date_filter", ""),
                )

                if not contexts:
                    status.update(label="Aucun document pertinent trouvé.", state="error")
                    context_str = "Aucun contexte pertinent trouvé."
                else:
                    nb_queries = 1 + (3 if cfg["use_expansion"] else 0) + (1 if cfg["use_hyde"] else 0)
                    status.update(
                        label=f"{len(contexts)} extraits (sur {nb_queries} requêtes)",
                        state="complete",
                    )
                    context_str = "\n\n---\n\n".join(contexts)

                st.session_state.last_chunks = detailed_chunks

            placeholder = st.empty()
            full_response = ""
            system_prompt = rag.build_system_prompt(context_str)

            try:
                for token in rag.stream_answer(ollama_client, cfg["model"], system_prompt, prompt):
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Erreur avec Ollama : {e}")


# ─── COLONNE CHUNKS ───────────────────────────────────────────────────────────

def render_chunks_panel():
    st.markdown("### 📦 Chunks Récupérés")

    chunks = st.session_state.get("last_chunks", [])
    if not chunks:
        st.info("Posez une question pour voir les chunks récupérés ici.")
        return

    st.caption(f"{len(chunks)} chunk(s) récupérés")
    st.markdown("---")

    for i, chunk in enumerate(chunks):
        with st.expander(f"**Chunk {i+1}** — {chunk['source'][:40]}...", expanded=(i == 0)):
            render_chunk_card(chunk)
            st.markdown("**Scores détaillés :**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Hybride", f"{chunk['hybrid_score']:.3f}")
            c2.metric("Vectoriel", f"{chunk['vecto_distance']:.4f}")
            c3.metric("BM25", f"{chunk['bm25_score']:.3f}")
            c4 = st.columns(4)[3]
            c4.metric("Rerank", f"{chunk.get('rerank_score', 0.0):.3f}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    chroma_client = get_chroma_client()
    ollama_client = get_ollama_client()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    cfg = render_sidebar(chroma_client, ollama_client)

    st.title("🤖 RAG Intégré — Chatbot + Visualisation Chunks")

    col_chat, col_chunks = st.columns([1.2, 1])

    with col_chat:
        render_chat(cfg, chroma_client, ollama_client)

    with col_chunks:
        render_chunks_panel()


if __name__ == "__main__":
    main()